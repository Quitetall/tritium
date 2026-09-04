#!/usr/bin/env python3
"""Generate the deterministic v1 stable API delta for release documentation."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import re
import shlex
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JSON = ROOT / "docs/generated/api-diff-v1.0-v1.1.json"
DEFAULT_MARKDOWN = ROOT / "docs/generated/api-diff-v1.0-v1.1.md"
class ApiDiffError(RuntimeError):
    """The public API report could not be generated or admitted."""


def pyo3_exports(source: str) -> set[str]:
    marker = source.find("#[pymodule]")
    if marker < 0:
        raise ApiDiffError("baseline has no #[pymodule] registration")
    opening = source.find("{", marker)
    if opening < 0:
        raise ApiDiffError("baseline #[pymodule] has no body")
    depth = 0
    closing = -1
    for index in range(opening, len(source)):
        if source[index] == "{":
            depth += 1
        elif source[index] == "}":
            depth -= 1
            if depth == 0:
                closing = index
                break
    if closing < 0:
        raise ApiDiffError("baseline #[pymodule] body is unclosed")
    body = source[opening + 1 : closing]
    active_lines = [line.split("//", 1)[0] for line in body.splitlines()]
    active_body = "\n".join(active_lines)
    if "{" in active_body or "}" in active_body:
        raise ApiDiffError("baseline uses conditional or nested PyO3 registration")
    if "#[" in active_body:
        raise ApiDiffError("baseline uses attributed or conditional PyO3 registration")
    classes = set(
        re.findall(
            r"(?m)^\s*m\.add_class::<([A-Za-z_][A-Za-z0-9_]*)>\(\)\?;\s*$",
            active_body,
        )
    )
    functions = set(
        re.findall(
            r"(?m)^\s*m\.add_function\(wrap_pyfunction!\("
            r"([A-Za-z_][A-Za-z0-9_]*)\s*,\s*m\)\?\)\?;\s*$",
            active_body,
        )
    )
    registrations = len(
        re.findall(r"(?m)^\s*m\.add_(?:class|function)", active_body)
    )
    if registrations != len(classes) + len(functions):
        raise ApiDiffError("baseline uses an unsupported PyO3 registration form")
    if re.search(r'(?m)^\s*m\.add\((?!\s*"__doc__")', active_body):
        raise ApiDiffError("baseline uses an unsupported PyO3 registration form")
    allowed_m_lines = re.compile(
        r"^\s*(?:m\.add_class::<[A-Za-z_][A-Za-z0-9_]*>\(\)\?;|"
        r"m\.add_function\(wrap_pyfunction!\([A-Za-z_][A-Za-z0-9_]*\s*,\s*m\)\?\)\?;|"
        r'm\.add\(\s*"__doc__"\s*,.*\)\?;)\s*$'
    )
    for line in active_lines:
        if re.search(r"\bm\b", line) and allowed_m_lines.fullmatch(line) is None:
            raise ApiDiffError("baseline uses an unsupported PyO3 registration helper")
    if re.search(r"#\[(?:pyclass|pyfunction|pyo3)\([^\]]*\bname\s*=", source):
        raise ApiDiffError("baseline uses a renamed PyO3 export")
    return classes | functions


def _string_list(node: ast.AST) -> set[str]:
    if not isinstance(node, (ast.List, ast.Tuple)):
        raise ApiDiffError("__all__ must be assigned or extended with a literal list")
    values: set[str] = set()
    for element in node.elts:
        if not isinstance(element, ast.Constant) or not isinstance(element.value, str):
            raise ApiDiffError("__all__ must contain only literal strings")
        values.add(element.value)
    return values


def python_all(source: str) -> set[str]:
    tree = ast.parse(source)
    exports: set[str] = set()
    assigned = False

    def apply(statements: list[ast.stmt], *, guarded: bool = False) -> None:
        nonlocal exports, assigned
        for node in statements:
            references_all = any(
                isinstance(child, ast.Name) and child.id == "__all__"
                for child in ast.walk(node)
            )
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "__all__"
                for target in node.targets
            ):
                if guarded or len(node.targets) != 1:
                    raise ApiDiffError("__all__ assignment must be a single module-level write")
                exports = _string_list(node.value)
                assigned = True
                continue
            if (
                isinstance(node, ast.Expr)
                and isinstance(node.value, ast.Call)
                and isinstance(node.value.func, ast.Attribute)
                and isinstance(node.value.func.value, ast.Name)
                and node.value.func.value.id == "__all__"
            ):
                call = node.value
                if call.func.attr != "extend" or len(call.args) != 1 or call.keywords:
                    raise ApiDiffError("unsupported __all__ mutation")
                if not assigned:
                    raise ApiDiffError("__all__.extend precedes its module-level assignment")
                exports |= _string_list(call.args[0])
                continue
            if isinstance(node, ast.Try):
                apply(node.body, guarded=True)
                for branch in [*node.handlers, *node.orelse, *node.finalbody]:
                    branch_body = branch.body if isinstance(branch, ast.ExceptHandler) else [branch]
                    if any(
                        isinstance(child, ast.Name) and child.id == "__all__"
                        for item in branch_body
                        for child in ast.walk(item)
                    ):
                        raise ApiDiffError("__all__ fallback mutation is environment-dependent")
                continue
            if references_all:
                raise ApiDiffError("unsupported nested or dynamic __all__ construction")

    apply(tree.body)
    if not assigned or not exports:
        raise ApiDiffError("current package defines no literal __all__ exports")
    return exports


def semver_crates(root: Path) -> tuple[str, ...]:
    source = (root / "scripts/check-semver.sh").read_text(encoding="utf-8")
    assignments = tuple(
        re.finditer(
            r"(?ms)^CHECKED_CRATES=\(\s*(?P<body>.*?)^\)",
            source,
        )
    )
    if len(assignments) != 1:
        raise ApiDiffError("SemVer gate has no unique CHECKED_CRATES assignment")
    try:
        crates = tuple(
            shlex.split(assignments[0].group("body"), comments=True, posix=True)
        )
    except ValueError as error:
        raise ApiDiffError("SemVer gate has an invalid CHECKED_CRATES assignment") from error
    if any(re.fullmatch(r"tritium-[a-z0-9-]+", crate) is None for crate in crates):
        raise ApiDiffError("SemVer gate CHECKED_CRATES scope is not literal")
    if not crates or len(crates) != len(set(crates)):
        raise ApiDiffError("SemVer gate has no unique frozen-crate scope")
    return crates


def report_identity(report: dict[str, object]) -> str:
    payload = {key: value for key, value in report.items() if key != "report_id"}
    return "sha256:" + hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def build_report(
    *,
    baseline: str,
    candidate_version: str,
    baseline_exports: set[str],
    current_exports: set[str],
    frozen_crates: tuple[str, ...] = (),
) -> dict[str, object]:
    removed = sorted(baseline_exports - current_exports)
    if removed:
        raise ApiDiffError(f"v1 Python API names were removed: {', '.join(removed)}")
    retained = sorted(baseline_exports & current_exports)
    added = sorted(current_exports - baseline_exports)
    if not frozen_crates:
        frozen_crates = (
            "tritium-core", "tritium-spec", "tritium-format", "tritium-runtime",
            "tritium-cpu", "tritium-quantize", "tritium-testkit",
        )
    report: dict[str, object] = {
        "schema": "tritium.api-diff.v1",
        "baseline": baseline,
        "candidate_version": candidate_version,
        "rust": {
            "frozen_crates": list(frozen_crates),
            "verification": f"./scripts/check-semver.sh {baseline}",
            "result": "run-required",
        },
        "python": {"retained": retained, "added": added, "removed": removed},
        "scope": {
            "c_abi": "separate cargo test -p tritium-ffi gate",
            "evolving_rust": "not covered by the stable SemVer promise",
            "runtime_receipt": "not produced by this structural report",
        },
    }
    report["report_id"] = report_identity(report)
    return report


def _git(root: Path, *arguments: str) -> str:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=root, text=True, stderr=subprocess.PIPE
        )
    except subprocess.CalledProcessError as error:
        message = error.stderr.strip() or str(error)
        raise ApiDiffError(message) from error


def _workspace_version(root: Path) -> str:
    cargo = (root / "Cargo.toml").read_text(encoding="utf-8")
    match = re.search(r"(?m)^version\s*=\s*\"([^\"]+)\"", cargo)
    if match is None:
        raise ApiDiffError("workspace version is missing")
    return match.group(1)


def repository_report(root: Path, baseline: str) -> dict[str, object]:
    baseline_source = _git(root, "show", f"{baseline}:crates/tritium-py/src/lib.rs")
    current_source = (root / "crates/tritium-py/python/tritium/__init__.py").read_text(
        encoding="utf-8"
    )
    baseline_exports = pyo3_exports(baseline_source)
    if not baseline_exports:
        raise ApiDiffError(f"{baseline} has no registered Python exports")
    return build_report(
        baseline=baseline,
        candidate_version=_workspace_version(root),
        baseline_exports=baseline_exports,
        current_exports=python_all(current_source),
        frozen_crates=semver_crates(root),
    )


def render_json(report: dict[str, object]) -> str:
    return json.dumps(report, indent=2, sort_keys=True) + "\n"


def render_markdown(report: dict[str, object]) -> str:
    python = report["python"]
    rust = report["rust"]
    assert isinstance(python, dict)
    assert isinstance(rust, dict)
    frozen_crates = rust["frozen_crates"]
    assert isinstance(frozen_crates, list)
    retained = ", ".join(f"`{name}`" for name in python["retained"])
    added = "\n".join(f"- `{name}`" for name in python["added"])
    return f"""# Generated v1.0 to v1.1 API diff

Report identity: `{report['report_id']}`

This is a structural source report for candidate `{report['candidate_version']}`
against `{report['baseline']}`. It is not a package-install or runtime receipt.

## Frozen Rust tier

The {len(frozen_crates)} frozen crates require a green cargo-semver-checks run:

```sh
{report['rust']['verification']}
```

## Python root namespace

Retained v1 names: {retained}.

Added in v1.1:

{added}

Removed v1 names: none. Generation fails if this changes.

## Boundaries

- C ABI: {report['scope']['c_abi']}.
- Evolving Rust: {report['scope']['evolving_rust']}.
- Runtime evidence: {report['scope']['runtime_receipt']}.
"""


def write_outputs(report: dict[str, object], json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(render_json(report), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")


def check_outputs(report: dict[str, object], json_path: Path, markdown_path: Path) -> None:
    expected = ((json_path, render_json(report)), (markdown_path, render_markdown(report)))
    for path, value in expected:
        if not path.is_file() or path.read_text(encoding="utf-8") != value:
            raise ApiDiffError(f"generated API diff is stale: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default="v1.0.0")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    try:
        report = repository_report(ROOT, args.baseline)
        if args.check:
            check_outputs(report, args.json, args.markdown)
        else:
            write_outputs(report, args.json, args.markdown)
    except (OSError, SyntaxError, ApiDiffError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
