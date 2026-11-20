"""Execute the pinned author full-tensor driver on tiny CPU tensors, no network.

Uses actual lower-LDL ordering and full feedback. No model/data download or
fallback is admitted. The only supplied source files are hash-authenticated by
the existing arithmetic generator; caller package initialization is not executed.
"""
import contextlib
import io
import json
from pathlib import Path
import runpy
import sys


def main():
    loader = runpy.run_path(str(Path(__file__).with_name("generate.py")))
    reference, _ = loader["load_reference"](sys.argv[1])
    import scipy
    import torch
    torch.set_num_threads(1)
    assert not torch.cuda.is_initialized()
    cases = []
    for columns in (8, 12):
        for width in (2, 4):
            for ratio in (0.0, 0.25, 0.5, 1.0):
                for decay in (0.0, 1.0):
                    for mode in ("normal", "dead", "pivot"):
                        if mode == "pivot" and ratio != 0.0:
                            continue
                        dead = mode == "dead"
                        weights = torch.tensor([
                            [(((row+1)*(col+3)) % 17-8)*0.35+col*0.19+row*0.07
                             for col in range(columns)] for row in range(4)], dtype=torch.float32)
                        if mode == "pivot":
                            weights[:, 0] *= 1000
                        activations = torch.eye(columns, dtype=torch.float32)
                        activations[0, 0] = 0.1
                        activations[0, 1] = 1.0
                        activations[1, 1] = 0.2
                        activations[0, 2] = 0.4
                        activations[2, 2] = 2.0
                        for i in range(3, columns):
                            activations[i, i] = 0.5+i*0.1
                            activations[i-1, i] = 0.2
                        if dead:
                            activations[:, 0] = 0
                        gram = activations.double().t().matmul(activations.double()) / columns
                        damping = 0.0 if mode == "pivot" else 0.007
                        config = reference.QTEAConfig(group_size=width, salient_ratio=ratio,
                            itf_iters=1, rescale_iters=2, damp=damping, decay=decay)
                        layer = torch.nn.Linear(columns, 4, bias=False, dtype=torch.float32)
                        layer.weight.data.copy_(weights)
                        driver = reference.QTEA(layer, config)
                        driver.hessian.copy_(gram.float())
                        # Ask the actual reference for its order before consuming H.
                        h = gram.float().clone()
                        w = weights.clone()
                        zeros = torch.diag(h) == 0
                        h[zeros, zeros] = 1
                        w[:, zeros] = 0
                        salient, rest = driver._select_salient(w, torch.diag(h))
                        order = torch.cat([salient, driver._ldl_order(h, rest)])
                        log = io.StringIO()
                        with contextlib.redirect_stdout(log):
                            payload = driver.quantize()
                        assert "falling back" not in log.getvalue()
                        assert "using its diagonal" not in log.getvalue()
                        assert not torch.cuda.is_initialized()
                        cases.append({"rows": 4, "columns": columns, "group_size": width,
                            "ratio": ratio, "decay": decay, "dead": dead, "damping": damping,
                            "pivot_changed": not torch.equal(order, torch.cat([salient, rest])),
                            "weights": weights.flatten().tolist(),
                            "activations": activations.flatten().tolist(),
                            "order": order.tolist(), "trits": payload["ternary"].flatten().tolist(),
                            "column_v": payload["col_v"].tolist(),
                            "dense": layer.weight.data.flatten().tolist(),
                            "salient_columns": payload["salient_cols"].tolist(),
                            "residual_indices": payload["residual_index"].tolist()})
    result = {"source": loader["PIN"], "source_sha256": loader["HASHES"],
              "torch": torch.__version__, "scipy": scipy.__version__,
              "scope": "tiny CPU full tensor arithmetic; not model/codec/backend qualification",
              "cases": cases}
    print("{")
    for key in ("source", "source_sha256", "torch", "scipy", "scope"):
        print("  " + json.dumps(key) + ": " + json.dumps(result[key]) + ",")
    print('  "cases": [')
    for i, case in enumerate(cases):
        print("    " + json.dumps(case, allow_nan=False, separators=(",", ":"))
              + ("," if i+1 < len(cases) else ""))
    print("  ]\n}")


if __name__ == "__main__":
    main()
