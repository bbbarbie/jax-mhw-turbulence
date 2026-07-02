#!/usr/bin/env python3
"""Generate inexpensive summary PNG figures from parsed result tables.

The renderer intentionally uses only the Python standard library so the archive
can regenerate figures in a minimal environment without matplotlib or Pillow.
"""

from __future__ import annotations

import argparse
import csv
import math
import struct
import zlib
from collections import defaultdict
from pathlib import Path


Color = tuple[int, int, int]


class Canvas:
    def __init__(self, width: int = 960, height: int = 640, background: Color = (255, 255, 255)):
        self.width = width
        self.height = height
        self.pixels = bytearray(background * (width * height))

    def set_pixel(self, x: int, y: int, color: Color) -> None:
        if 0 <= x < self.width and 0 <= y < self.height:
            idx = 3 * (y * self.width + x)
            self.pixels[idx:idx + 3] = bytes(color)

    def line(self, x0: int, y0: int, x1: int, y1: int, color: Color, width: int = 1) -> None:
        dx = abs(x1 - x0)
        sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0)
        sy = 1 if y0 < y1 else -1
        err = dx + dy
        while True:
            for ox in range(-(width // 2), width // 2 + 1):
                for oy in range(-(width // 2), width // 2 + 1):
                    self.set_pixel(x0 + ox, y0 + oy, color)
            if x0 == x1 and y0 == y1:
                break
            e2 = 2 * err
            if e2 >= dy:
                err += dy
                x0 += sx
            if e2 <= dx:
                err += dx
                y0 += sy

    def rect(self, x0: int, y0: int, x1: int, y1: int, color: Color, fill: bool = False) -> None:
        if fill:
            for y in range(min(y0, y1), max(y0, y1) + 1):
                for x in range(min(x0, x1), max(x0, x1) + 1):
                    self.set_pixel(x, y, color)
        else:
            self.line(x0, y0, x1, y0, color)
            self.line(x1, y0, x1, y1, color)
            self.line(x1, y1, x0, y1, color)
            self.line(x0, y1, x0, y0, color)

    def circle(self, cx: int, cy: int, radius: int, color: Color) -> None:
        r2 = radius * radius
        for y in range(cy - radius, cy + radius + 1):
            for x in range(cx - radius, cx + radius + 1):
                if (x - cx) ** 2 + (y - cy) ** 2 <= r2:
                    self.set_pixel(x, y, color)

    def text(self, x: int, y: int, text: str, color: Color = (0, 0, 0), scale: int = 2) -> None:
        cx = x
        for char in text.upper():
            glyph = FONT.get(char, FONT[" "])
            for gy, row in enumerate(glyph):
                for gx, bit in enumerate(row):
                    if bit == "1":
                        self.rect(
                            cx + gx * scale,
                            y + gy * scale,
                            cx + (gx + 1) * scale - 1,
                            y + (gy + 1) * scale - 1,
                            color,
                            fill=True,
                        )
            cx += 6 * scale

    def save_png(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = bytearray()
        stride = self.width * 3
        for y in range(self.height):
            raw.append(0)
            raw.extend(self.pixels[y * stride:(y + 1) * stride])

        def chunk(kind: bytes, data: bytes) -> bytes:
            return (
                struct.pack(">I", len(data))
                + kind
                + data
                + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
            )

        data = b"".join(
            [
                b"\x89PNG\r\n\x1a\n",
                chunk(b"IHDR", struct.pack(">IIBBBBB", self.width, self.height, 8, 2, 0, 0, 0)),
                chunk(b"IDAT", zlib.compress(bytes(raw), 9)),
                chunk(b"IEND", b""),
            ]
        )
        path.write_bytes(data)
        print(f"Wrote {path}")


FONT = {
    " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
    "-": ["00000", "00000", "00000", "11110", "00000", "00000", "00000"],
    ".": ["00000", "00000", "00000", "00000", "00000", "01100", "01100"],
    ",": ["00000", "00000", "00000", "00000", "00000", "01100", "01000"],
    "=": ["00000", "11110", "00000", "11110", "00000", "00000", "00000"],
    "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
    "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
    "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
    "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
    "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
    "5": ["11111", "10000", "10000", "11110", "00001", "00001", "11110"],
    "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
    "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
    "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
    "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
    "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
    "B": ["11110", "10001", "10001", "11110", "10001", "10001", "11110"],
    "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
    "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
    "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
    "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
    "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
    "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
    "K": ["10001", "10010", "10100", "11000", "10100", "10010", "10001"],
    "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
    "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
    "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
    "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
    "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
    "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
    "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
    "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
    "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
    "V": ["10001", "10001", "10001", "10001", "10001", "01010", "00100"],
    "W": ["10001", "10001", "10001", "10101", "10101", "10101", "01010"],
    "X": ["10001", "10001", "01010", "00100", "01010", "10001", "10001"],
    "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
}


def load_rows(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise SystemExit(f"Missing table: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def map_linear(value: float, vmin: float, vmax: float, lo: int, hi: int) -> int:
    if math.isclose(vmin, vmax):
        return (lo + hi) // 2
    return int(round(lo + (value - vmin) * (hi - lo) / (vmax - vmin)))


def axes(canvas: Canvas, title: str, note: str = "") -> tuple[int, int, int, int]:
    left, top, right, bottom = 110, 80, 900, 540
    grid = (225, 225, 225)
    black = (0, 0, 0)
    for i in range(6):
        y = top + i * (bottom - top) // 5
        canvas.line(left, y, right, y, grid)
    for i in range(6):
        x = left + i * (right - left) // 5
        canvas.line(x, top, x, bottom, grid)
    canvas.rect(left, top, right, bottom, black)
    canvas.text(left, 25, title, black, scale=2)
    if note:
        canvas.text(left, 575, note, (70, 70, 70), scale=1)
    return left, top, right, bottom


def direct_ad_plot(rows: list[dict[str, str]], field: str, title: str, output: Path) -> None:
    canvas = Canvas()
    left, top, right, bottom = axes(canvas, title, "X LOG AD STEPS, Y LINEAR NORMALIZED GRADIENT")
    by_alpha: dict[str, list[dict[str, str]]] = defaultdict(list)
    values: list[float] = []
    for row in rows:
        by_alpha[row["alpha"]].append(row)
        values.append(float(row[field]))
    ymin, ymax = min(values), max(values)
    pad = max(abs(ymin), abs(ymax), 1e-18) * 0.08
    ymin -= pad
    ymax += pad
    xmin = math.log10(min(int(row["ad_steps"]) for row in rows))
    xmax = math.log10(max(int(row["ad_steps"]) for row in rows))
    colors = {"0.2": (37, 99, 235), "0.8": (220, 38, 38)}
    for alpha, group in sorted(by_alpha.items(), key=lambda item: float(item[0])):
        group.sort(key=lambda row: int(row["ad_steps"]))
        points = []
        for row in group:
            x = map_linear(math.log10(int(row["ad_steps"])), xmin, xmax, left, right)
            y = map_linear(float(row[field]), ymin, ymax, bottom, top)
            points.append((x, y))
            canvas.circle(x, y, 5, colors.get(alpha, (0, 0, 0)))
        for (x0, y0), (x1, y1) in zip(points, points[1:]):
            canvas.line(x0, y0, x1, y1, colors.get(alpha, (0, 0, 0)), width=2)
    canvas.text(650, 95, "BLUE A=0.2", (37, 99, 235), scale=2)
    canvas.text(650, 125, "RED A=0.8", (220, 38, 38), scale=2)
    canvas.save_png(output)


def box_distribution(rows: list[dict[str, str]], field: str, title: str, output: Path) -> None:
    canvas = Canvas()
    left, top, right, bottom = axes(canvas, title, "BOXPLOTS SHOW NORMALIZED PER-STEP GRADIENTS")
    groups: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        groups[row["alpha"]].append(float(row[field]))
    values = [value for group in groups.values() for value in group]
    ymin, ymax = min(values), max(values)
    pad = max(abs(ymin), abs(ymax), 1e-18) * 0.08
    ymin -= pad
    ymax += pad
    colors = [(37, 99, 235), (220, 38, 38)]
    for idx, alpha in enumerate(sorted(groups, key=float)):
        data = sorted(groups[alpha])
        q1 = data[len(data) // 4]
        med = data[len(data) // 2]
        q3 = data[(3 * len(data)) // 4]
        low, high = data[0], data[-1]
        cx = left + (idx + 1) * (right - left) // 3
        box_w = 80
        color = colors[idx % len(colors)]
        y_low = map_linear(low, ymin, ymax, bottom, top)
        y_high = map_linear(high, ymin, ymax, bottom, top)
        y_q1 = map_linear(q1, ymin, ymax, bottom, top)
        y_q3 = map_linear(q3, ymin, ymax, bottom, top)
        y_med = map_linear(med, ymin, ymax, bottom, top)
        canvas.line(cx, y_low, cx, y_high, color, width=2)
        canvas.line(cx - 35, y_low, cx + 35, y_low, color, width=2)
        canvas.line(cx - 35, y_high, cx + 35, y_high, color, width=2)
        canvas.rect(cx - box_w // 2, y_q3, cx + box_w // 2, y_q1, color)
        canvas.line(cx - box_w // 2, y_med, cx + box_w // 2, y_med, color, width=3)
        canvas.text(cx - 55, bottom + 18, f"A={alpha}", color, scale=2)
    canvas.save_png(output)


def finite_difference_plot(rows: list[dict[str, str]], output: Path) -> None:
    canvas = Canvas()
    left, top, right, bottom = axes(canvas, "FINITE DIFFERENCE SENSITIVITY", "SATURATED FD VALUES FROM EXISTING SUMMARY")
    values = [float(row["finite_difference_sensitivity"]) for row in rows]
    ymin, ymax = min(values + [0.0]), max(values + [0.0])
    pad = max(abs(ymin), abs(ymax), 1e-18) * 0.12
    ymin -= pad
    ymax += pad
    zero_y = map_linear(0.0, ymin, ymax, bottom, top)
    canvas.line(left, zero_y, right, zero_y, (0, 0, 0), width=2)
    for idx, row in enumerate(rows):
        value = float(row["finite_difference_sensitivity"])
        cx = left + (idx + 1) * (right - left) // 3
        y = map_linear(value, ymin, ymax, bottom, top)
        color = (37, 99, 235) if value >= 0 else (220, 38, 38)
        canvas.rect(cx - 65, min(y, zero_y), cx + 65, max(y, zero_y), color, fill=True)
        canvas.text(cx - 80, bottom + 18, f"A={row['baseline_alpha']}", color, scale=2)
    canvas.save_png(output)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--direct-ad", type=Path, default=Path("results/summary/tables/direct_ad_window_sweep.csv"))
    parser.add_argument("--ensemble", type=Path, default=Path("results/summary/tables/ensemble_ad_results.csv"))
    parser.add_argument("--finite-difference", type=Path, default=Path("results/summary/tables/finite_difference_summary.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results/summary/figures"))
    args = parser.parse_args()

    direct_rows = load_rows(args.direct_ad)
    ensemble_rows = load_rows(args.ensemble)
    fd_rows = load_rows(args.finite_difference)
    direct_ad_plot(direct_rows, "grad_alpha_per_step", "DIRECT AD GRAD ALPHA PER STEP", args.output_dir / "direct_ad_grad_alpha_vs_steps.png")
    direct_ad_plot(direct_rows, "grad_kappa_per_step", "DIRECT AD GRAD KAPPA PER STEP", args.output_dir / "direct_ad_grad_kappa_vs_steps.png")
    box_distribution(ensemble_rows, "grad_alpha_per_step", "ENSEMBLE GRAD ALPHA DISTRIBUTION", args.output_dir / "ensemble_grad_alpha_distribution.png")
    box_distribution(ensemble_rows, "grad_kappa_per_step", "ENSEMBLE GRAD KAPPA DISTRIBUTION", args.output_dir / "ensemble_grad_kappa_distribution.png")
    finite_difference_plot(fd_rows, args.output_dir / "finite_difference_sensitivity.png")


if __name__ == "__main__":
    main()
