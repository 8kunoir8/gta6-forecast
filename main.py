"""gta6-trophy-rarity-forecast-01 entry: count-space structural forecast.

Reads every <id>.in in MAKEFUN_INPUT_DIR (default /input), writes <id>.out to
MAKEFUN_OUTPUT_DIR (default /output): {"trophies":[{"id","q":[5 quantiles]}]}.
Standard library only, deterministic. Model in model.py beside this file.
"""

import json
import os
import sys
from pathlib import Path

# The grader runs `python3 -P -s /bot/main.py`: -P leaves /bot off sys.path.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import forecast_case

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
