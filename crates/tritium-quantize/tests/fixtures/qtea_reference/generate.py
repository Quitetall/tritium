"""Generate tiny CPU goldens from the actual MIT QTEA author code; no network.

Usage: python generate.py /path/containing/qtea > golden.json
The caller supplies the two unmodified files from the pinned revision. This
script verifies their hashes before importing them. It does not fit any model,
load datasets, call CUDA, or execute the full feedback driver.
"""

import hashlib
import json
from pathlib import Path
import sys
import types

PIN = "e39bd8c0297e3427daeb404427c74cb6e202b683"
HASHES = {
    "qtea.py": "1a3abb5021dca5c928d34b744b52d3498a7f82efcccfc7b160701577e24480c1",
    "quantizer.py": "5071b111fed5c08d06fba0e7e07e07471e6feea83eee194cee71d302322a5f34",
}


def main():
    root = Path(sys.argv[1]).resolve()
    sources = {}
    for name, digest in HASHES.items():
        source = (root / "qtea" / name).read_bytes()
        if hashlib.sha256(source).hexdigest() != digest:
            raise ValueError(f"pinned reference hash mismatch: {name}")
        sources[name] = source
    # Execute only the exact bytes just authenticated. Do not execute a caller's
    # package __init__.py or re-open unchecked files through import resolution.
    package_name = "_tritium_qtea_oracle"
    package = types.ModuleType(package_name)
    package.__path__ = []
    sys.modules[package_name] = package
    for name in ("quantizer.py", "qtea.py"):
        module_name = package_name + "." + name.removesuffix(".py")
        module = types.ModuleType(module_name)
        module.__package__ = package_name
        sys.modules[module_name] = module
        exec(compile(sources[name], str(root / "qtea" / name), "exec"), module.__dict__)
    reference = sys.modules[package_name + ".qtea"]
    quantizer = sys.modules[package_name + ".quantizer"]
    import torch

    torch.set_num_threads(1)
    groups = []
    inputs = [
        [[-1, 0, 1], [-10, 0, 10]],
        [[-5, -0.5, 0.25, 2, 7], [3, -8, 0.1, 0, 4]],
        [[-3, -2, -1, 4], [-12, -8, -4, 16]],
        [[(row + 2) * (col % 13) - 7 + (col % 3) * 0.0625
          for col in range(128)] for row in range(4)],
    ]
    for matrix in inputs:
        for iterations in (0, 1, 4):
            weights = torch.tensor(matrix, dtype=torch.float32)
            group = quantizer.TernaryGroup(0.96, iterations).fit(weights)
            rows, columns = weights.shape
            column_cases = []
            for offset in (0.0, 0.25, -1.0):
                column = weights[:, -1:] + offset
                for rounds in (0, 1, 3):
                    driver = object.__new__(reference.QTEA)
                    driver.config = reference.QTEAConfig(rescale_iters=rounds)
                    codes, multiplier = driver._rescale_refine(column - group.beta, group)
                    values = group.alpha * multiplier * codes + group.beta
                    column_cases.append({
                        "column": column.flatten().tolist(), "iterations": rounds,
                        "trits": codes.flatten().to(torch.int8).tolist(),
                        "multiplier": multiplier, "values": values.flatten().tolist(),
                    })
            groups.append({
                "weights": weights.flatten().tolist(), "rows": rows, "columns": columns,
                "iterations": iterations, "coefficient": 0.96,
                "alpha": group.alpha.flatten().tolist(), "beta": group.beta.flatten().tolist(),
                "delta": group.delta.flatten().tolist(), "refinements": column_cases,
            })
    # Every adjacent positive finite FP8 midpoint (including zero/subnormals),
    # both signs and exact representable values. Peak448 forces column scale1.
    finite_codes = torch.arange(127, dtype=torch.uint8).view(torch.float8_e4m3fn)
    levels = finite_codes.float().tolist()
    samples = list(levels)
    for left, right in zip(levels, levels[1:]):
        midpoint = torch.tensor((left + right) / 2, dtype=torch.float32)
        samples.extend((torch.nextafter(midpoint, torch.tensor(float("-inf"))).item(),
                        midpoint.item(),
                        torch.nextafter(midpoint, torch.tensor(float("inf"))).item()))
    rounding = []
    for magnitude in samples:
        for sign in (1.0, -1.0):
            residual = torch.tensor([448, 0, 0, 0, sign * magnitude, 0, 0, 0])
            driver = object.__new__(reference.QTEA)
            driver.config = reference.QTEAConfig()
            dense, index, value, scale = driver._sparse_residual(residual)
            codes = (value * scale).to(torch.float8_e4m3fn).view(torch.uint8)
            rounding.append([sign * magnitude, codes.tolist()[1], dense.tolist()[4]])
    sparse = []
    for residual in ([0.0] * 8, [-0.0, 0.0, 0.0, 0.0],
                     [1, -1, 0, 0, 0, -3, 3, 0], [0, 1e-12, 0, 0]):
        residual = torch.tensor(residual, dtype=torch.float32)
        driver = object.__new__(reference.QTEA)
        driver.config = reference.QTEAConfig()
        dense, index, value, scale = driver._sparse_residual(residual)
        codes = (value * scale).to(torch.float8_e4m3fn).view(torch.uint8)
        sparse.append({"residual": residual.tolist(), "indices": index.tolist(),
                       "codes": codes.tolist(), "scale": scale.item(), "dense": dense.tolist()})
    ranking = []
    for ratio in (0.0, 0.125, 0.25, 0.5, 0.625, 0.875, 1.0):
        weights = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8],
                                [1, -1, 2, -2, 3, -3, 4, -4]], dtype=torch.float32)
        diagonal = torch.tensor([1, 3, 2, 4, 6, 5, 8, 7], dtype=torch.float32)
        driver = object.__new__(reference.QTEA)
        driver.columns = 8
        driver.config = reference.QTEAConfig(group_size=2, salient_ratio=ratio)
        salient, rest = driver._select_salient(weights, diagonal)
        ranking.append({"weights": weights.flatten().tolist(), "rows": 2, "columns": 8,
                        "diagonal": diagonal.tolist(), "group_size": 2, "ratio": ratio,
                        "order": torch.cat([salient, rest]).tolist(),
                        "salient_count": len(salient)})
    result = {"source": PIN, "source_sha256": HASHES, "torch": torch.__version__,
              "scope": "tiny CPU arithmetic only; not model or solver qualification",
              "groups": groups, "rounding": rounding, "sparse": sparse, "ranking": ranking}
    # One case per line keeps the generated fixture reviewable without thousands
    # of lines of array punctuation. Python's encoder preserves negative zero.
    def encode(value):
        return json.dumps(value, allow_nan=False, separators=(",", ":"))
    print("{")
    for index, (key, value) in enumerate(result.items()):
        comma = "," if index + 1 < len(result) else ""
        if isinstance(value, list):
            print(f"  {encode(key)}: [")
            for row, case in enumerate(value):
                print("    " + encode(case) + ("," if row + 1 < len(value) else ""))
            print("  ]" + comma)
        else:
            print(f"  {encode(key)}: {encode(value)}{comma}")
    print("}")


if __name__ == "__main__":
    main()
