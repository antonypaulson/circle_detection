"""Backward-compatible entrypoint. Prefer: python -m circle_detection train"""

from circle_detection.train import main

if __name__ == "__main__":
    main()
