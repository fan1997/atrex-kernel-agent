#!/usr/bin/env python3
"""One-shape remote diagnostic for a repository staging payload."""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path


def main() -> int:
    stage = Path.cwd()
    root = stage / "runtime"
    request = json.loads((stage / "request.json").read_text(encoding="utf-8"))
    sys.path[:0] = [
        str(root),
        *[str(root / item) for item in request.get("python_roots", [])],
    ]
    os.chdir(root)
    try:
        from input import _make_inputs
        from kernel import Model

        shapes = json.loads((root / "shapes.json").read_text(encoding="utf-8"))
        shape_id = sorted(
            shapes, key=lambda value: int(value) if value.isdigit() else value
        )[0]
        inputs = _make_inputs(**shapes[shape_id]["input_kwargs"])
        output = Model().eval()(**inputs)
        import torch

        torch.cuda.synchronize()
        print(
            f"PASS shape={shape_id} output={tuple(output.shape)} dtype={output.dtype}"
        )
        return 0
    except Exception:
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
