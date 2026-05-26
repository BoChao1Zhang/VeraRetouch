import pytest
import torch

from llava.model.VeraRetouch import VeraRetouchForCausalLLM_Unified


class _FakeVeraRetouch:
    retouch_token_light_idx = 11
    retouch_token_colortemp_idx = 12
    retouch_token_colormixer_idx = 13

    def __init__(self):
        self.retouch_head = torch.nn.Linear(18, 12, bias=False)
        with torch.no_grad():
            self.retouch_head.weight.copy_(torch.eye(12, 18))

    _retouch_token_positions = VeraRetouchForCausalLLM_Unified._retouch_token_positions
    _resolve_hidden_layers = VeraRetouchForCausalLLM_Unified._resolve_hidden_layers
    _retouch_masks_as_tensor = VeraRetouchForCausalLLM_Unified._retouch_masks_as_tensor
    extract_retouch_hidden_bank = VeraRetouchForCausalLLM_Unified.extract_retouch_hidden_bank
    project_hidden_bank_to_sa_lut_exec = VeraRetouchForCausalLLM_Unified.project_hidden_bank_to_sa_lut_exec


def _generation_hidden_states(sequence_length=6, layer_count=5, hidden_dim=6):
    hidden_states = []
    for step in range(sequence_length):
        layers = []
        for layer in range(layer_count):
            value = torch.full((1, step + 1, hidden_dim), float(layer * 100 + step))
            layers.append(value)
        hidden_states.append(tuple(layers))
    return hidden_states


def test_extract_retouch_hidden_bank_uses_selected_layers_and_token_order():
    fake = _FakeVeraRetouch()
    output_ids = torch.tensor([[1, 11, 2, 12, 3, 13]])
    hidden_states = _generation_hidden_states()

    bank = fake.extract_retouch_hidden_bank(output_ids, hidden_states, selected_layers=[-2, -1])

    assert bank.shape == (1, 2, 3, 6)
    assert torch.all(bank[0, 0, :, 0] == torch.tensor([301.0, 303.0, 305.0]))
    assert torch.all(bank[0, 1, :, 0] == torch.tensor([401.0, 403.0, 405.0]))


def test_project_hidden_bank_to_exec_latents_applies_retouch_mask():
    fake = _FakeVeraRetouch()
    output_ids = torch.tensor([[1, 11, 2, 12, 3, 13]])
    hidden_states = _generation_hidden_states()
    hidden_bank = fake.extract_retouch_hidden_bank(output_ids, hidden_states, selected_layers=[-1])

    z_exec, flat = fake.project_hidden_bank_to_sa_lut_exec(
        hidden_bank,
        retouch_masks=torch.tensor([[1.0, 0.0, 1.0]]),
    )

    assert z_exec.shape == (1, 3, 4)
    assert flat.shape == (1, 12)
    assert flat[:, 4:8].abs().sum() == 0
    assert flat[:, :4].abs().sum() > 0
    assert flat[:, 8:].abs().sum() > 0


def test_extract_retouch_hidden_bank_errors_when_token_missing():
    fake = _FakeVeraRetouch()
    output_ids = torch.tensor([[1, 11, 2, 12, 3, 99]])
    hidden_states = _generation_hidden_states()

    with pytest.raises(ValueError, match="was not found"):
        fake.extract_retouch_hidden_bank(output_ids, hidden_states)
