"""CPU regression tests for absolute-color supervision and legacy-cache guards."""
import sqlite3
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from tools.epr061_cache import chain_common as C
from tools.epr061_cache.chain_codes import Solver
from veraretouch_sprf.readout import mixed_codes as MC
from veraretouch_sprf.readout import multistage_loss as ML


def features(colors, geometry=None):
    weights = torch.softmax(colors[..., :2], dim=-1)
    return torch.cat(((weights.unsqueeze(-1) * colors.unsqueeze(-2)).flatten(-2),
                      weights, colors, torch.ones_like(colors[..., :1])), dim=-1)


class AbsoluteCodeTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        g = torch.Generator().manual_seed(20260919)
        self.z = torch.rand(2, 120, 3, generator=g, dtype=torch.float64)
        self.beta = torch.rand(2, 120, generator=g, dtype=torch.float64)
        self.beta[:, :10] = 0
        self.beta[:, 10:15] = 1e-10
        self.phi = features(self.z)
        self.d = self.phi.shape[-1]
        self.identity = torch.zeros(3, self.d, dtype=torch.float64)
        self.identity[:, -4:-1] = torch.eye(3)
        code = self.identity + 0.04 * torch.randn(2, 3, self.d, generator=g,
                                                 dtype=torch.float64)
        self.prev = self.z + self.beta[..., None] * (
            self.phi @ code.transpose(-1, -2) - self.z)
        self.solver = Solver.__new__(Solver)
        self.solver.device = 'cpu'
        self.solver.glut = C.Glut.__new__(C.Glut)
        self.solver.glut.size = self.d
        self.solver.glut.ridge = 1e-3
        self.solver.glut.eye = torch.eye(self.d, dtype=torch.float64)

    def test_solution_matches_augmented_least_squares(self):
        gram, rhs = self.solver.statistics(self.phi, self.z, self.prev, self.beta)
        got, _ = self.solver.solve_absolute(gram, rhs)
        for i in range(2):
            h = self.beta[i, :, None] * self.phi[i]
            t = self.prev[i] - (1 - self.beta[i, :, None]) * self.z[i]
            lam = 1e-3 * (h.square().sum() / self.d)
            design = torch.cat((h, lam.sqrt() * torch.eye(self.d)))
            target = torch.cat((t, torch.zeros(self.d, 3)))
            expected = torch.linalg.lstsq(design, target, driver='gelsd').solution.T
            torch.testing.assert_close(got[i], expected, rtol=1e-8, atol=1e-9)
        self.assertTrue(torch.isfinite(got).all())

    def test_pixel_subsets_and_empty_support(self):
        rows = torch.arange(0, 120, 2)
        sub = self.solver.statistics(self.phi, self.z, self.prev, self.beta, rows)
        direct = C.absolute_statistics(self.phi[:, rows], self.z[:, rows],
                                        self.prev[:, rows], self.beta[:, rows])
        for a, b in zip(sub, direct):
            torch.testing.assert_close(a, b)
        gram, rhs = self.solver.statistics(self.phi, self.z, self.z,
                                           torch.zeros_like(self.beta))
        codes, empty = self.solver.solve_absolute(gram, rhs)
        self.assertTrue(empty.all())
        torch.testing.assert_close(codes, self.identity.expand_as(codes))

    def test_residual_cache_cannot_be_used_as_absolute(self):
        residual = torch.full_like(self.identity, .01).expand(2, -1, -1)
        legacy = self.z + self.beta[..., None] * (self.phi @ residual.transpose(-1, -2))
        wrong = self.z + self.beta[..., None] * (
            self.phi @ residual.transpose(-1, -2) - self.z)
        torch.testing.assert_close(legacy - wrong, self.beta[..., None] * self.z)
        self.assertGreater(float((legacy - wrong).abs().mean()), .1)
        # An identity shift converts the unregularized function, but rebuilding
        # is required to match the new ridge objective and scaled-mask convention.
        converted = residual + self.identity
        correct = self.z + self.beta[..., None] * (
            self.phi @ converted.transpose(-1, -2) - self.z)
        torch.testing.assert_close(correct, legacy)

    def test_cached_metric_uses_training_executor(self):
        codes, _ = self.solver.solve_absolute(*self.solver.statistics(
            self.phi, self.z, self.prev, self.beta))
        metric = self.solver.render_error(self.phi, self.z, self.prev, self.beta,
                                           {'fit': codes})['fit'][0]
        rendered = self.z + self.beta[..., None] * (
            self.phi @ codes.transpose(-1, -2) - self.z)
        expected = (rendered - self.prev).abs().amax(-1).mean(-1) * 255
        torch.testing.assert_close(metric.double(), expected, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(rendered[:, :10], self.z[:, :10])

    def test_current_local_solver_and_loss_already_agree(self):
        z, prev, a = self.z[0].float(), self.prev[0].float(), self.beta[0].float()
        with patch.object(MC, 'glut_features', features):
            code = MC.solve_support_code(z, prev, a, None, ridge=1e-3)
            rendered = MC.apply_support(code, z, a, None)
        pix = ML.ChainPixelL1.__new__(ML.ChainPixelL1)
        pix.basis = type('Basis', (), {'geometry': None})()
        with patch.object(ML, 'glut_features', features):
            prediction = code.detach().requires_grad_(True)
            actual = pix._chunk_sum(z, prev, a, prediction) / z.numel()
            actual.backward()
        torch.testing.assert_close(actual.detach(), (rendered - prev).abs().mean())
        self.assertTrue(torch.isfinite(prediction.grad).all())
        # Global and all-one masked paths are the same objective.
        with patch.object(MC, 'glut_features', features):
            global_code = MC.solve_support_code(z, prev, None, None)
            ones_code = MC.solve_support_code(z, prev, torch.ones_like(a), None)
        torch.testing.assert_close(global_code, ones_code)

    def test_strength_absorption_is_not_double_applied(self):
        strength = .4
        code = self.identity * 1.1
        absorbed = (1 - strength) * self.identity + strength * code
        original = self.z + (strength * self.beta)[..., None] * (self.phi @ code.T - self.z)
        support_only = self.z + self.beta[..., None] * (self.phi @ absorbed.T - self.z)
        torch.testing.assert_close(original, support_only)

    def test_legacy_resume_rejected_before_pack_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            index = root / 'old.index.sqlite'
            with sqlite3.connect(index) as db:
                db.execute('CREATE TABLE meta(k TEXT PRIMARY KEY,v TEXT)')
            pack = root / 'old.tar'
            pack.write_bytes(b'preserve legacy payload')
            with patch.object(C, 'SCRATCH', root / 'scratch'):
                with self.assertRaisesRegex(ValueError, 'code semantics'):
                    C.ResumableTar('old', out=root, resume=False,
                                   meta={'code_semantics': C.ABSOLUTE_CODE_SEMANTICS})
            self.assertEqual(pack.read_bytes(), b'preserve legacy payload')

    def test_new_contract_is_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            index = Path(directory) / 'new.sqlite'
            with sqlite3.connect(index) as db:
                db.execute('CREATE TABLE meta(k TEXT PRIMARY KEY,v TEXT)')
                db.execute('INSERT INTO meta VALUES(?,?)',
                           ('code_semantics', C.ABSOLUTE_CODE_SEMANTICS))
            C.check_cache_semantics(index, C.ABSOLUTE_CODE_SEMANTICS)


if __name__ == '__main__':
    unittest.main()
