"""gta6-trophy-rarity-forecast-01 entry: count-space structural forecast.

Reads every <id>.in in MAKEFUN_INPUT_DIR (default /input), writes <id>.out to
MAKEFUN_OUTPUT_DIR (default /output): {"trophies":[{"id","q":[5 quantiles]}]}.
Standard library only, deterministic. Model in model.py beside this file.

The grader runs `python3 -P -s /bot/main.py`; -P leaves /bot off sys.path, so
main.py restores its directory first and model.py imports from there.
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from model import forecast_case

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))


def _sanity(rows):
    """Clamp to the output contract: five finite 0..100 non-decreasing quantiles."""
    clean = []
    for row in rows["trophies"]:
        q = sorted(round(min(100.0, max(0.0, float(x))), 4) for x in row["q"])[:5]
        clean.append({"id": row["id"], "q": q})
    return {"trophies": clean}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = _sanity(forecast_case(case))
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
