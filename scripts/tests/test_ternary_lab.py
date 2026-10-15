"""Independent small-oracle checks for the research verifier and campaign gate."""
import importlib.util
import pathlib
import unittest
import subprocess
import sys
import tempfile

ROOT = pathlib.Path(__file__).resolve().parents[2]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class IntegerOracleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            cls.verify = load("verify_ternary", "verify-ternary-lab.py")
        except ModuleNotFoundError as error:
            raise unittest.SkipTest(str(error)) from error
        cls.campaign = load("run_ternary", "run-ternary-lab.py")

    def test_signed_truncation_matches_contract(self):
        self.assertEqual(self.verify.trunc(-3, 2), -1)
        self.assertEqual(self.verify.trunc(3, 2), 1)

    def test_all_tiny_network_states_and_global_optimum(self):
        losses = []
        for a in (-1, 0, 1):
            for b in (-1, 0, 1):
                for c in (-1, 0, 1):
                    model = dict(inputs=1, hidden=1, outputs=2, input_shift=0,
                                 output_shift=0, trits=[a, b, c])
                    out = self.verify.forward(model, [3])
                    self.assertEqual(out, [max(0, 3*a)*b, max(0, 3*a)*c])
                    losses.append((out[0]-256)**2+out[1]**2)
        self.assertEqual(min(losses), 253**2)

    def test_vacuous_tuning_rejected_before_artifact_creation(self):
        with tempfile.TemporaryDirectory() as folder:
            output = pathlib.Path(folder) / "campaign"
            run = subprocess.run([sys.executable, str(ROOT / "scripts/run-ternary-lab.py"),
                                  "--binary", "unused", "--data", "unused", "--output", str(output),
                                  "--tune-steps", "2000"], capture_output=True, text=True)
            self.assertEqual(run.returncode, 2)
            self.assertIn("at least 3176 steps", run.stderr)
            self.assertFalse(output.exists())

    def test_unpaired_results_are_rejected(self):
        left = [dict(seed=i, evaluation_count=100, loss_sum=1, correct=1) for i in range(5)]
        with self.assertRaises(ValueError):
            self.campaign.paired(left, list(reversed(left)))

    def test_paired_gate_rejects_inconclusive_or_accuracy_regression(self):
        base = [dict(loss_sum=100, evaluation_count=100, correct=90)]*5
        self.assertFalse(self.campaign.paired(base, base)["advance"])
        better = [dict(loss_sum=50+i, evaluation_count=100, correct=90) for i in range(5)]
        self.assertTrue(self.campaign.paired(better, base)["advance"])
        for row in better:
            row["correct"] = 88
        self.assertFalse(self.campaign.paired(better, base)["advance"])


if __name__ == "__main__":
    unittest.main()
