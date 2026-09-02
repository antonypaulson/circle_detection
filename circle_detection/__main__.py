"""CLI: python -m circle_detection {train,infer,view,smoke}"""

from __future__ import annotations

import argparse

from circle_detection import infer, smoke, train, viewer


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(
        prog="circle-detection",
        description="Detect circles against noise using a small CNN.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    train.add_arguments(sub.add_parser("train", help="Train on synthetic noisy circles"))
    infer.add_arguments(sub.add_parser("infer", help="Score a trained model on synthetic circles"))
    viewer.add_arguments(sub.add_parser("view", help="Show or save detection overlays"))
    sub.add_parser("smoke", help="Tiny train + infer to verify the install")

    args = parser.parse_args(argv)
    if args.command == "train":
        train.run(args)
    elif args.command == "infer":
        infer.run(args)
    elif args.command == "view":
        viewer.run(args)
    elif args.command == "smoke":
        smoke.run_smoke()
    else:  # pragma: no cover
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
