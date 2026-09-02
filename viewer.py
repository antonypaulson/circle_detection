"""Backward-compatible entrypoint. Prefer: python -m circle_detection view"""

from circle_detection.viewer import main

if __name__ == "__main__":
    main()
