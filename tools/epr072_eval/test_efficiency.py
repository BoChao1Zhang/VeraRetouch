"""CPU-only checks of geometry and safe scheduling; no model inference."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from tools.epr072_eval import efficiency as E, queue_efficiency as Q


class GeometryTests(unittest.TestCase):
    def test_sizes(self):
        self.assertEqual(E.test_size((1536,1024),512),(768,512))
        self.assertEqual(E.test_size((1024,1536),1024),(1024,1536))
        for shape in [(1599,899),(900,1600),(1000,1000)]:
            out=E.test_size(shape,512)
            self.assertEqual(min(out),512)
            self.assertTrue(all(v%64==0 for v in out))

    def test_busy_is_retryable(self):
        with patch.object(E,'other_processes',return_value=[123]):
            with self.assertRaises(E.DeviceBusy):E.require_exclusive()


class QueueTests(unittest.TestCase):
    def launch(self,child,pids):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(Q,'ROOT',Path(directory)), patch.object(Q,'wait_idle'), \
                 patch.object(Q.subprocess,'Popen',return_value=child), \
                 patch.object(Q,'processes',return_value=pids), patch.object(Q.time,'sleep'):
                return Q.launch(['unused'],'test',{})

    def test_conflict_stops_only_owned_child(self):
        child=Mock(pid=100);child.poll.return_value=None
        self.assertEqual(self.launch(child,{100,200}),'retry')
        child.terminate.assert_called_once_with()
        child.kill.assert_not_called()

    def test_failing_job_does_not_abort_queue(self):
        child=Mock(pid=100,returncode=1);child.poll.return_value=1
        self.assertEqual(self.launch(child,set()),'failed')
        child.terminate.assert_not_called()

    def test_retryable_exit(self):
        child=Mock(pid=100,returncode=75);child.poll.return_value=75
        self.assertEqual(self.launch(child,set()),'retry')

    def test_success(self):
        child=Mock(pid=100,returncode=0);child.poll.return_value=0
        self.assertEqual(self.launch(child,set()),'complete')


if __name__=='__main__':unittest.main()
