"""Tests for the loss-curve SVG."""

import xml.etree.ElementTree as ET

import pytest

from tooltuned.charts import HEIGHT, Y_MAX, PlotBox, loss_curve_svg

SVG = "{http://www.w3.org/2000/svg}"
TRAIN = [(10, 0.14), (20, 0.05), (30, 0.02)]
VALIDATION = [(20, 0.03), (30, 0.019)]


def parsed() -> ET.Element:
    return ET.fromstring(loss_curve_svg(TRAIN, VALIDATION))


def test_svg_is_valid_xml_with_a_title_for_screen_readers() -> None:
    root = parsed()
    assert root.tag == f"{SVG}svg"
    assert root.find(f"{SVG}title").text == "Training and validation loss"


def test_one_line_per_series_and_a_dot_per_validation_point() -> None:
    root = parsed()
    lines = root.findall(f"{SVG}polyline")
    assert [line.get("class") for line in lines] == ["train", "val"]
    assert len(lines[0].get("points").split()) == len(TRAIN)
    dots = [c for c in root.iter(f"{SVG}circle") if c.get("class") == "val-dot"]
    assert len(dots) == len(VALIDATION) + 1  # plus the legend's marker


def test_end_label_shows_the_final_validation_loss() -> None:
    texts = [t.text for t in parsed().iter(f"{SVG}text")]
    assert "0.0190" in texts


def test_dark_mode_colours_are_included() -> None:
    assert "prefers-color-scheme: dark" in loss_curve_svg(TRAIN, VALIDATION)


@pytest.mark.parametrize(("loss", "expected"), [(0.0, 344.0), (Y_MAX, 84.0), (1.0, 84.0)])
def test_y_scale_maps_loss_into_the_plot_and_clips_large_values(
    loss: float, expected: float
) -> None:
    assert PlotBox().y(loss) == pytest.approx(expected)


def test_x_scale_spans_the_plot_width() -> None:
    box = PlotBox()
    assert box.x(0, 100) == box.left
    assert box.x(100, 100) == box.left + box.width
    assert box.top + box.height < HEIGHT
