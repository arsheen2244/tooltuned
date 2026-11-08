"""Draw the training and validation loss curves as a small SVG file, with no plotting library.

SVG is a text format for drawings that GitHub shows inside a README. The colours follow
the light/dark palette from the dataviz guide, and switch with the reader's theme.
"""

from dataclasses import dataclass

Point = tuple[float, float]

WIDTH, HEIGHT = 720, 400
Y_MAX = 0.15
X_TICK_STEP = 200
Y_TICKS = (0.0, 0.05, 0.10, 0.15)

STYLE = """
  .surface { fill: #fcfcfb; }
  .ink { fill: #0b0b0b; }
  .ink2 { fill: #52514e; }
  .muted { fill: #898781; }
  .grid { stroke: #e1e0d9; stroke-width: 1; }
  .axis { stroke: #c3c2b7; stroke-width: 1; }
  .train { stroke: #2a78d6; fill: none; stroke-width: 2; stroke-linejoin: round; }
  .train-fill { fill: #2a78d6; }
  .val { stroke: #eb6834; fill: none; stroke-width: 2; stroke-linejoin: round; }
  .val-dot { fill: #eb6834; stroke: #fcfcfb; stroke-width: 2; }
  @media (prefers-color-scheme: dark) {
    .surface { fill: #1a1a19; }
    .ink { fill: #f0efec; }
    .ink2 { fill: #c3c2b7; }
    .grid { stroke: #2c2c2a; }
    .axis { stroke: #383835; }
    .train { stroke: #3987e5; }
    .train-fill { fill: #3987e5; }
    .val { stroke: #d95926; }
    .val-dot { fill: #d95926; stroke: #1a1a19; }
  }
  text { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif; }
"""


@dataclass(frozen=True)
class PlotBox:
    """The rectangle, in pixels, where the data is drawn."""

    left: float = 64
    top: float = 84
    width: float = 560
    height: float = 260

    def x(self, step: float, x_max: float) -> float:
        return self.left + step / x_max * self.width

    def y(self, loss: float) -> float:
        return self.top + self.height - min(loss, Y_MAX) / Y_MAX * self.height


def polyline(points: list[Point], box: PlotBox, x_max: float, css_class: str) -> str:
    coords = " ".join(f"{box.x(x, x_max):.1f},{box.y(y):.1f}" for x, y in points)
    return f'<polyline class="{css_class}" points="{coords}"/>'


def dots(points: list[Point], box: PlotBox, x_max: float) -> str:
    return "".join(
        f'<circle class="val-dot" cx="{box.x(x, x_max):.1f}" cy="{box.y(y):.1f}" r="5"/>'
        for x, y in points
    )


def axes(box: PlotBox, x_max: float) -> str:
    """Gridlines, tick labels and axis titles."""
    parts = []
    for tick in Y_TICKS:
        y = box.y(tick)
        css = "axis" if tick == 0 else "grid"
        parts.append(f'<line class="{css}" x1="{box.left}" x2="{box.left + box.width}" y1="{y:.1f}" y2="{y:.1f}"/>')
        parts.append(f'<text class="muted" font-size="11" x="{box.left - 8}" y="{y + 4:.1f}" text-anchor="end">{tick:.2f}</text>')
    for step in range(0, int(x_max) + 1, X_TICK_STEP):
        x = box.x(step, x_max)
        parts.append(f'<text class="muted" font-size="11" x="{x:.1f}" y="{box.top + box.height + 18}" text-anchor="middle">{step}</text>')
    parts.append(f'<text class="ink2" font-size="12" x="{box.left + box.width / 2}" y="{HEIGHT - 12}" text-anchor="middle">Update step (2 epochs = 1,226 updates)</text>')
    parts.append(f'<text class="ink2" font-size="12" transform="translate(16 {box.top + box.height / 2}) rotate(-90)" text-anchor="middle">Loss</text>')
    return "".join(parts)


def header() -> str:
    """Title and a legend row, so identity never relies on colour alone."""
    return (
        '<text class="ink" font-size="16" font-weight="600" x="16" y="28">Training and validation loss, QLoRA r=16</text>'
        '<rect class="train-fill" x="16" y="46" width="16" height="3" rx="1.5"/>'
        '<text class="ink2" font-size="12" x="38" y="51">Training loss (every 10 updates)</text>'
        '<line class="val" x1="246" x2="266" y1="47.5" y2="47.5"/>'
        '<circle class="val-dot" cx="256" cy="47.5" r="5"/>'
        '<text class="ink2" font-size="12" x="272" y="51">Validation loss (487 held-out rows)</text>'
    )


def end_label(last: Point, box: PlotBox, x_max: float) -> str:
    x, y = box.x(last[0], x_max) + 10, box.y(last[1])
    return (
        f'<text class="ink" font-size="12" font-weight="600" x="{x:.1f}" y="{y - 4:.1f}">Validation</text>'
        f'<text class="ink2" font-size="12" x="{x:.1f}" y="{y + 11:.1f}">{last[1]:.4f}</text>'
    )


def loss_curve_svg(train: list[Point], validation: list[Point]) -> str:
    """Return a complete SVG drawing of both loss curves on one shared loss axis."""
    box = PlotBox()
    x_max = max(x for x, _ in train + validation)
    body = "".join(
        [
            f'<rect class="surface" width="{WIDTH}" height="{HEIGHT}" rx="8"/>',
            header(),
            axes(box, x_max),
            polyline(train, box, x_max, "train"),
            polyline(validation, box, x_max, "val"),
            dots(validation, box, x_max),
            end_label(validation[-1], box, x_max),
        ]
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH} {HEIGHT}" '
        f'width="{WIDTH}" height="{HEIGHT}" role="img" aria-labelledby="t d">'
        '<title id="t">Training and validation loss</title>'
        '<desc id="d">Training loss falls from 0.14 to about 0.01 over 1,226 updates. '
        f"Validation loss falls from {validation[0][1]:.4f} to {validation[-1][1]:.4f} "
        "and stays flat at the end, so the model did not overfit.</desc>"
        f"<style>{STYLE}</style>{body}</svg>\n"
    )
