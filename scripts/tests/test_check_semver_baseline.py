"""Regression coverage for the stable Rust API baseline selector."""

import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
STABLE_TAG = re.compile(r"v(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)")


def test_semver_gate_selects_latest_reachable_stable_release_not_old_pre_freeze_tag():
    completed = subprocess.run(
        ["bash", "scripts/check-semver.sh", "--print-baseline"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    tags = subprocess.check_output(
        ["git", "tag", "--merged", "HEAD", "--sort=-version:refname"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    expected = next(
        tag
        for tag in tags
        if STABLE_TAG.fullmatch(tag)
        and subprocess.check_output(
            ["git", "rev-parse", f"{tag}^{{commit}}"], cwd=ROOT, text=True
        ).strip()
        != head
    )
    assert completed.stdout.strip() == expected
    assert completed.stderr == ""


def test_semver_selector_skips_stable_tag_pointing_at_current_head():
    command = r'''
source scripts/check-semver.sh
HEAD_VALUE=current
git() {
  if [[ "$1" == "tag" ]]; then
    printf 'v1.1.0\nv1.0.0\n'
  elif [[ "$1" == "rev-parse" && "$2" == "HEAD" ]]; then
    printf '%s\n' "$HEAD_VALUE"
  elif [[ "$1" == "rev-parse" && "$2" == "v1.1.0^{commit}" ]]; then
    printf 'current\n'
  else
    printf 'previous\n'
  fi
}
latest_stable_baseline
HEAD_VALUE=advanced
latest_stable_baseline
'''
    completed = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.splitlines() == ["v1.0.0", "v1.1.0"]
    assert completed.stderr == ""


def test_published_baseline_version_is_reported_and_overridable():
    """The real baseline is a PUBLISHED version, not the never-published v1.0.0 tag."""
    completed = subprocess.run(
        ["bash", "scripts/check-semver.sh", "--print-baseline-version"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    assert completed.stdout.strip() == "1.1.0-rc.0"
    assert completed.stderr == ""

    overridden = subprocess.run(
        ["bash", "scripts/check-semver.sh", "--print-baseline-version"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "TRITIUM_SEMVER_BASELINE_VERSION": "1.2.0"},
    )
    assert overridden.stdout.strip() == "1.2.0"


def test_unknown_mode_is_rejected_before_any_check_runs():
    """A typo must fail loudly rather than silently pick report or block."""
    completed = subprocess.run(
        ["bash", "scripts/check-semver.sh"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env={**os.environ, "TRITIUM_SEMVER_MODE": "bogus"},
    )
    assert completed.returncode == 2
    assert "must be report or block" in completed.stderr


def test_semver_selector_rejects_leading_zero_components():
    command = r'''
source scripts/check-semver.sh
git() {
  if [[ "$1" == "tag" ]]; then
    printf 'v01.2.3\nv1.02.3\nv1.2.03\nv1.2.3\n'
  elif [[ "$1" == "rev-parse" && "$2" == "HEAD" ]]; then
    printf 'current\n'
  else
    printf 'previous\n'
  fi
}
latest_stable_baseline
'''
    completed = subprocess.run(
        ["bash", "-c", command],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )

    assert completed.stdout.strip() == "v1.2.3"
    assert completed.stderr == ""
