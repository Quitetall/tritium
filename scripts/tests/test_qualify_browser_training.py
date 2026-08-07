import json
from pathlib import Path
import runpy
import shutil
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE = runpy.run_path(ROOT / "scripts" / "qualify-browser-training.py")
SUPPORT = runpy.run_path(
    ROOT / "scripts" / "tests" / "test_verify_browser_training_receipt.py"
)
assemble = MODULE["assemble"]
qualify = MODULE["qualify"]
validate = MODULE["validate_receipt"]
QualificationError = MODULE["QualificationError"]
require_clean_revision = MODULE["require_clean_revision"]


def lane_fragments(root: Path):
    fixture = SUPPORT["BrowserTrainingReceiptTests"]()
    _, archive, receipt = fixture.fixture(root)
    paths = []
    for lane in receipt["lanes"]:
        directory = root / f"input-{lane['engine']}"
        directory.mkdir()
        source_trace = root / lane["trace"]["file"]
        destination = directory / source_trace.name
        shutil.copyfile(source_trace, destination)
        fragment = directory / "lane.json"
        fragment.write_text(json.dumps(lane), encoding="utf-8")
        paths.append(fragment)
    return archive, tuple(paths)


class BrowserTrainingQualificationTests(unittest.TestCase):
    def test_clean_revision_rejects_nonignored_untracked_source(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            subprocess.run(["git", "init", "-q"], cwd=root, check=True)
            subprocess.run(
                ["git", "config", "user.email", "test@tritium.invalid"],
                cwd=root,
                check=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Tritium Test"],
                cwd=root,
                check=True,
            )
            (root / "tracked.txt").write_text("tracked\n")
            subprocess.run(["git", "add", "tracked.txt"], cwd=root, check=True)
            subprocess.run(["git", "commit", "-qm", "fixture"], cwd=root, check=True)
            revision = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            require_clean_revision(root, revision)
            (root / "untracked.txt").write_text("source injection\n")
            with self.assertRaisesRegex(QualificationError, "clean exact"):
                require_clean_revision(root, revision)

    def test_assembles_revalidated_distinct_traces_and_strict_receipt(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archive, lanes = lane_fragments(root)
            stage = root / "stage"
            receipt = assemble(
                stage,
                archive=archive,
                lane_paths=lanes,
                source_revision="a" * 40,
                release="1.1.0-rc.0",
                run_id="browser-aggregate-1",
            )
            self.assertEqual(
                [lane["trace"]["file"] for lane in receipt["lanes"]],
                [
                    "traces/chrome.trace.json",
                    "traces/firefox.trace.json",
                    "traces/safari.trace.json",
                ],
            )
            self.assertEqual(
                validate(stage / "receipt.json", "a" * 40, "1.1.0-rc.0", archive),
                receipt,
            )

    def test_rejects_wrong_lane_order_before_publication(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archive, lanes = lane_fragments(root)
            with self.assertRaisesRegex(ValueError, "ordered"):
                assemble(
                    root / "stage",
                    archive=archive,
                    lane_paths=(lanes[1], lanes[0], lanes[2]),
                    source_revision="a" * 40,
                    release="1.1.0-rc.0",
                    run_id="browser-aggregate-1",
                )
            self.assertFalse((root / "stage" / "receipt.json").exists())

    def test_dirty_source_failure_leaves_no_output(self):
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archive, lanes = lane_fragments(root)
            output = root / "published"
            with mock.patch.dict(
                qualify.__globals__,
                {
                    "require_clean_revision": mock.Mock(
                        side_effect=QualificationError("dirty tracked source")
                    )
                },
            ):
                with self.assertRaisesRegex(QualificationError, "dirty"):
                    qualify(
                        output,
                        repo=ROOT,
                        archive=archive,
                        lane_paths=lanes,
                        source_revision="a" * 40,
                        release="1.1.0-rc.0",
                        run_id="browser-aggregate-1",
                    )
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
