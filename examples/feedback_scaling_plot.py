"""
Plot feedback scaling curves on one Cartesian coordinate system.

The chart intentionally uses one shared linear Y axis for both curves:
raw audit context grows linearly with rows, while compact trust state stays
near the X axis because it grows by learned bucket count.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from examples.feedback_scaling_benchmark import (
    DEFAULT_SCALES,
    ScalingResult,
    run_feedback_scaling_benchmark,
)

_HERE = os.path.dirname(__file__)
_ROOT = os.path.normpath(os.path.join(_HERE, ".."))
DEFAULT_OUTPUT = os.path.join(_ROOT, "data", "feedback_scaling_curves.svg")


def build_svg(
    results: list[ScalingResult],
    width: int = 920,
    height: int = 560,
) -> str:
    if not results:
        raise ValueError("results must not be empty")

    margin_left = 86
    margin_right = 34
    margin_top = 48
    margin_bottom = 72
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    max_rows = max(result.rows for result in results)
    max_tokens = max(result.raw_audit_tokens for result in results)
    y_max = _nice_upper_bound(max_tokens)

    def x_scale(rows: int) -> float:
        return margin_left + (rows / max_rows) * plot_width

    def y_scale(tokens: float) -> float:
        return margin_top + plot_height - (tokens / y_max) * plot_height

    audit_points = [(x_scale(result.rows), y_scale(result.raw_audit_tokens)) for result in results]
    trust_points = [(x_scale(result.rows), y_scale(result.trust_profile_tokens)) for result in results]
    audit_polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in audit_points)
    trust_polyline = " ".join(f"{x:.1f},{y:.1f}" for x, y in trust_points)

    x_axis_y = margin_top + plot_height
    y_axis_x = margin_left
    y_ticks = _ticks(y_max, 5)
    x_ticks = [result.rows for result in results]

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        "<title id=\"title\">Feedback Compression Scaling</title>",
        "<desc id=\"desc\">Raw audit context tokens and compact trust profile tokens plotted against feedback rows on one linear Cartesian coordinate system.</desc>",
        "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>",
        f'<text x="{margin_left}" y="28" font-family="Arial, sans-serif" font-size="20" font-weight="700" fill="#1f2933">Feedback compression scaling</text>',
        f'<text x="{margin_left}" y="48" font-family="Arial, sans-serif" font-size="12" fill="#52606d">same linear Y axis, estimated tokens = bytes / 4</text>',
    ]

    for tick in y_ticks:
        y = y_scale(tick)
        parts.append(f'<line x1="{y_axis_x}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}" stroke="#e4e7eb" stroke-width="1"/>')
        parts.append(f'<text x="{margin_left - 10}" y="{y + 4:.1f}" text-anchor="end" font-family="Arial, sans-serif" font-size="11" fill="#52606d">{_format_tick(tick)}</text>')

    for tick in x_ticks:
        x = x_scale(tick)
        parts.append(f'<line x1="{x:.1f}" y1="{x_axis_y}" x2="{x:.1f}" y2="{x_axis_y + 6}" stroke="#9aa5b1" stroke-width="1"/>')
        parts.append(f'<text x="{x:.1f}" y="{x_axis_y + 24}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#52606d">{tick:,}</text>')

    parts.extend(
        [
            f'<line x1="{y_axis_x}" y1="{margin_top}" x2="{y_axis_x}" y2="{x_axis_y}" stroke="#323f4b" stroke-width="1.5"/>',
            f'<line x1="{y_axis_x}" y1="{x_axis_y}" x2="{width - margin_right}" y2="{x_axis_y}" stroke="#323f4b" stroke-width="1.5"/>',
            f'<text x="{margin_left + plot_width / 2:.1f}" y="{height - 22}" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#323f4b">feedback rows</text>',
            f'<text x="22" y="{margin_top + plot_height / 2:.1f}" transform="rotate(-90 22 {margin_top + plot_height / 2:.1f})" text-anchor="middle" font-family="Arial, sans-serif" font-size="13" fill="#323f4b">estimated tokens</text>',
            f'<polyline points="{audit_polyline}" fill="none" stroke="#d64545" stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round"/>',
            f'<polyline points="{trust_polyline}" fill="none" stroke="#2563eb" stroke-width="3.2" stroke-linejoin="round" stroke-linecap="round"/>',
        ]
    )

    for result, (x, y) in zip(results, audit_points):
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#d64545"/>')
        if result.rows in {results[0].rows, results[-1].rows}:
            parts.append(f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#7f1d1d">{result.raw_audit_tokens:.0f}</text>')

    for result, (x, y) in zip(results, trust_points):
        parts.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="4.5" fill="#2563eb"/>')
        if result.rows in {results[0].rows, results[-1].rows}:
            parts.append(f'<text x="{x:.1f}" y="{y - 10:.1f}" text-anchor="middle" font-family="Arial, sans-serif" font-size="11" fill="#1e3a8a">{result.trust_profile_tokens:.0f}</text>')

    legend_x = margin_left + 18
    legend_y = margin_top + 22
    parts.extend(
        [
            f'<rect x="{legend_x - 12}" y="{legend_y - 18}" width="284" height="56" rx="6" fill="#ffffff" stroke="#d9e2ec"/>',
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 28}" y2="{legend_y}" stroke="#d64545" stroke-width="3.2"/>',
            f'<text x="{legend_x + 38}" y="{legend_y + 4}" font-family="Arial, sans-serif" font-size="12" fill="#323f4b">raw audit context tokens</text>',
            f'<line x1="{legend_x}" y1="{legend_y + 24}" x2="{legend_x + 28}" y2="{legend_y + 24}" stroke="#2563eb" stroke-width="3.2"/>',
            f'<text x="{legend_x + 38}" y="{legend_y + 28}" font-family="Arial, sans-serif" font-size="12" fill="#323f4b">learned trust profile tokens</text>',
        ]
    )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def write_svg(path: str, svg: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)


def _nice_upper_bound(value: float) -> int:
    if value <= 0:
        return 1
    magnitude = 10 ** (len(str(int(value))) - 1)
    return int(((value + magnitude - 1) // magnitude) * magnitude)


def _ticks(max_value: int, count: int) -> list[int]:
    return [round(max_value * index / count) for index in range(count + 1)]


def _format_tick(value: int) -> str:
    if value >= 1_000:
        return f"{value // 1_000}k"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot feedback scaling benchmark curves as SVG."
    )
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--scales",
        default=",".join(str(scale) for scale in DEFAULT_SCALES),
        help="Comma-separated row counts, default: 100,500,1000,5000,10000",
    )
    args = parser.parse_args()
    scales = tuple(int(item.strip()) for item in args.scales.split(",") if item.strip())
    results = run_feedback_scaling_benchmark(scales)
    svg = build_svg(results)
    write_svg(args.output, svg)
    print(f"Wrote feedback scaling plot: {args.output}")


if __name__ == "__main__":
    main()
