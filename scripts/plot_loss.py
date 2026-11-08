"""Draw the loss curves from a saved loss history as an SVG image.

    python scripts/plot_loss.py results/qlora_r16
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from tooltuned.charts import loss_curve_svg  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path, help="Folder holding loss_history.json.")
    folder = parser.parse_args().folder
    history = json.loads((folder / "loss_history.json").read_text(encoding="utf-8"))
    output = folder / "loss_curves.svg"
    output.write_text(loss_curve_svg(history["train"], history["validation"]), encoding="utf-8")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
