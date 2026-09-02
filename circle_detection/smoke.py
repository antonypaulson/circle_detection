"""Short train + infer on synthetic data to verify the install."""

from __future__ import annotations

import tempfile
from pathlib import Path

from circle_detection.infer import run_inference
from circle_detection.train import train


def run_smoke(workdir: str | None = None) -> Path:
    root = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="circle-detection-smoke-"))
    root.mkdir(parents=True, exist_ok=True)
    model_path = root / "smoke.keras"

    print("Smoke test: training 2 steps on 16 synthetic images...")
    train(
        model_path=str(model_path),
        steps=2,
        batch_size=8,
        samples=16,
        eval_samples=8,
        seed=0,
    )
    print("Smoke test: running inference on 8 synthetic images...")
    run_inference(model_path=str(model_path), samples=8, seed=1, save_dir=str(root / "overlays"))
    print(f"Smoke test passed. Artifacts in {root}")
    return root


def main(argv=None) -> None:
    run_smoke()


if __name__ == "__main__":
    main()
