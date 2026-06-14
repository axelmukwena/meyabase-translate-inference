"""Command-line entrypoint (non-HuggingFace).

Usage:  uv run cli.py --direction en-ng "Hello, how are you?"
"""
from __future__ import annotations

import argparse

from translation.config import DIRECTIONS
from translation.core import Translator, extract_translations


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Translate text with meyabase models.")
    parser.add_argument("text", help="Text to translate.")
    parser.add_argument(
        "--direction",
        "-d",
        required=True,
        choices=sorted(DIRECTIONS),
        help="Translation direction.",
    )
    args = parser.parse_args(argv)

    translator = Translator()
    prediction = translator.translate(args.text, direction=args.direction)
    print(extract_translations(prediction))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
