"""Минимальные построители SVG-графиков без внешних библиотек.

Дашборд должен открываться из файла на любой машине, в том числе без доступа
в интернет и на отечественной ОС, поэтому в нём нет ни одной внешней
зависимости: графики — обычный инлайновый SVG.

Палитра и правила разметки взяты из валидированного набора: три категориальных
оттенка, проходящих проверку на цветовую слепоту по всем парам, тонкие штрихи,
подписи значений прямо у данных (не только цветом), обязательная таблица рядом
с каждым графиком.
"""
from __future__ import annotations

import html
from dataclasses import dataclass

import numpy as np

SERIES = ("var(--series-1)", "var(--series-2)", "var(--series-3)")


def _esc(s) -> str:
    return html.escape(str(s))


def _nice_ticks(lo: float, hi: float, n: int = 5) -> list[float]:
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return [lo]
    span = hi - lo
    raw = span / max(n - 1, 1)
    mag = 10 ** np.floor(np.log10(raw))
    for m in (1, 2, 2.5, 5, 10):
        if raw <= m * mag:
            step = m * mag
            break
    else:
        step = 10 * mag
    start = np.ceil(lo / step) * step
    ticks = []
    v = start
    while v <= hi + 1e-9:
        ticks.append(round(float(v), 10))
        v += step
    return ticks


@dataclass
class Axes:
    w: int = 560
    h: int = 300
    pad_l: int = 62
    pad_r: int = 22
    pad_t: int = 18
    pad_b: int = 46

    @property
    def x0(self): return self.pad_l

    @property
    def x1(self): return self.w - self.pad_r

    @property
    def y0(self): return self.h - self.pad_b

    @property
    def y1(self): return self.pad_t


def line_chart(x, series: list[tuple[str, list[float]]], *, x_label="", y_label="",
               ax: Axes | None = None, label_every: int = 2, y_min=None, y_max=None,
               x_is_category=False) -> str:
    """Линейный график. Точки подписаны напрямую — идентичность не только цветом."""
    ax = ax or Axes()
    xs = np.asarray(x, dtype=float)
    all_y = np.concatenate([np.asarray(v, dtype=float) for _, v in series])
    finite = all_y[np.isfinite(all_y)]
    lo = y_min if y_min is not None else float(finite.min())
    hi = y_max if y_max is not None else float(finite.max())
    if hi - lo < 1e-9:
        hi, lo = hi + 1, lo - 1
    pad = (hi - lo) * 0.14
    lo, hi = lo - pad, hi + pad
    xlo, xhi = float(np.nanmin(xs)), float(np.nanmax(xs))
    if xhi - xlo < 1e-12:
        xhi = xlo + 1

    def px(v): return ax.x0 + (v - xlo) / (xhi - xlo) * (ax.x1 - ax.x0)
    def py(v): return ax.y0 - (v - lo) / (hi - lo) * (ax.y0 - ax.y1)

    parts = [f'<svg viewBox="0 0 {ax.w} {ax.h}" class="chart" role="img">']
    for t in _nice_ticks(lo, hi):
        y = py(t)
        parts.append(f'<line x1="{ax.x0}" y1="{y:.1f}" x2="{ax.x1}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{ax.x0-9}" y="{y+4:.1f}" class="tick tick-y">{t:g}</text>')
    for i, xv in enumerate(xs):
        if i % label_every == 0 or i == len(xs) - 1:
            parts.append(f'<text x="{px(xv):.1f}" y="{ax.y0+20}" class="tick tick-x">{xv:g}</text>')
    parts.append(f'<line x1="{ax.x0}" y1="{ax.y0}" x2="{ax.x1}" y2="{ax.y0}" class="axis"/>')

    for si, (name, ys) in enumerate(series):
        col = SERIES[si % len(SERIES)]
        pts = [(px(a), py(b)) for a, b in zip(xs, ys) if np.isfinite(b)]
        d = " ".join(("M" if k == 0 else "L") + f"{a:.1f},{b:.1f}" for k, (a, b) in enumerate(pts))
        parts.append(f'<path d="{d}" fill="none" stroke="{col}" stroke-width="2" '
                     f'stroke-linejoin="round" stroke-linecap="round"/>')
        for k, (a, b) in enumerate(pts):
            parts.append(f'<circle cx="{a:.1f}" cy="{b:.1f}" r="4" fill="{col}" '
                         f'stroke="var(--surface-1)" stroke-width="2"/>')
        if pts:
            a, b = pts[-1]
            parts.append(f'<text x="{a-6:.1f}" y="{b-11:.1f}" class="serieslabel" '
                         f'text-anchor="end">{_esc(name)}</text>')
    if y_label:
        parts.append(f'<text x="14" y="{(ax.y0+ax.y1)/2:.0f}" class="axislabel" '
                     f'transform="rotate(-90 14 {(ax.y0+ax.y1)/2:.0f})">{_esc(y_label)}</text>')
    if x_label:
        parts.append(f'<text x="{(ax.x0+ax.x1)/2:.0f}" y="{ax.h-8}" class="axislabel" '
                     f'text-anchor="middle">{_esc(x_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def bar_chart(labels: list[str], values: list[float], *, y_label="", ax: Axes | None = None,
              series_idx: int = 0, value_fmt="{:.0f}", horizontal=False) -> str:
    ax = ax or Axes(h=280, pad_b=64)
    vals = np.asarray(values, dtype=float)
    hi = float(np.nanmax(vals)) * 1.18 or 1.0
    col = SERIES[series_idx % len(SERIES)]
    parts = [f'<svg viewBox="0 0 {ax.w} {ax.h}" class="chart" role="img">']
    if horizontal:
        n = len(labels)
        bh = (ax.y0 - ax.y1) / max(n, 1) * 0.62
        step = (ax.y0 - ax.y1) / max(n, 1)
        for i, (lb, v) in enumerate(zip(labels, vals)):
            y = ax.y1 + i * step + (step - bh) / 2
            w = (v / hi) * (ax.x1 - ax.x0)
            parts.append(f'<rect x="{ax.x0}" y="{y:.1f}" width="{max(w,1):.1f}" '
                         f'height="{bh:.1f}" rx="4" fill="{col}"/>')
            parts.append(f'<text x="{ax.x0-9}" y="{y+bh*0.68:.1f}" class="tick tick-y">'
                         f'{_esc(lb)}</text>')
            parts.append(f'<text x="{ax.x0+w+7:.1f}" y="{y+bh*0.68:.1f}" class="barvalue">'
                         f'{value_fmt.format(v)}</text>')
    else:
        n = len(labels)
        bw = (ax.x1 - ax.x0) / max(n, 1) * 0.6
        step = (ax.x1 - ax.x0) / max(n, 1)
        for t in _nice_ticks(0, hi):
            y = ax.y0 - t / hi * (ax.y0 - ax.y1)
            parts.append(f'<line x1="{ax.x0}" y1="{y:.1f}" x2="{ax.x1}" y2="{y:.1f}" class="grid"/>')
            parts.append(f'<text x="{ax.x0-9}" y="{y+4:.1f}" class="tick tick-y">{t:g}</text>')
        for i, (lb, v) in enumerate(zip(labels, vals)):
            x = ax.x0 + i * step + (step - bw) / 2
            h = (v / hi) * (ax.y0 - ax.y1)
            parts.append(f'<rect x="{x:.1f}" y="{ax.y0-h:.1f}" width="{bw:.1f}" '
                         f'height="{max(h,1):.1f}" rx="4" fill="{col}"/>')
            parts.append(f'<text x="{x+bw/2:.1f}" y="{ax.y0-h-7:.1f}" class="barvalue" '
                         f'text-anchor="middle">{value_fmt.format(v)}</text>')
            parts.append(f'<text x="{x+bw/2:.1f}" y="{ax.y0+18:.1f}" class="tick tick-x" '
                         f'text-anchor="middle">{_esc(lb)}</text>')
        parts.append(f'<line x1="{ax.x0}" y1="{ax.y0}" x2="{ax.x1}" y2="{ax.y0}" class="axis"/>')
    if y_label:
        parts.append(f'<text x="{(ax.x0+ax.x1)/2:.0f}" y="{ax.h-8}" class="axislabel" '
                     f'text-anchor="middle">{_esc(y_label)}</text>')
    parts.append("</svg>")
    return "".join(parts)


def grouped_bar(labels: list[str], groups: list[tuple[str, list[float]]], *,
                y_label="", ax: Axes | None = None, value_fmt="{:.1f}") -> str:
    ax = ax or Axes(w=760, h=320, pad_b=86)
    arr = np.array([g[1] for g in groups], dtype=float)
    hi = float(np.nanmax(arr)) * 1.2 or 1.0
    n, k = len(labels), len(groups)
    step = (ax.x1 - ax.x0) / max(n, 1)
    bw = step * 0.78 / k
    parts = [f'<svg viewBox="0 0 {ax.w} {ax.h}" class="chart" role="img">']
    for t in _nice_ticks(0, hi):
        y = ax.y0 - t / hi * (ax.y0 - ax.y1)
        parts.append(f'<line x1="{ax.x0}" y1="{y:.1f}" x2="{ax.x1}" y2="{y:.1f}" class="grid"/>')
        parts.append(f'<text x="{ax.x0-9}" y="{y+4:.1f}" class="tick tick-y">{t:g}</text>')
    for i, lb in enumerate(labels):
        base = ax.x0 + i * step + step * 0.11
        for j, (gname, vals) in enumerate(groups):
            v = float(vals[i])
            x = base + j * (bw + 2)
            h = (v / hi) * (ax.y0 - ax.y1) if np.isfinite(v) else 0
            parts.append(f'<rect x="{x:.1f}" y="{ax.y0-h:.1f}" width="{bw-2:.1f}" '
                         f'height="{max(h,1):.1f}" rx="3" fill="{SERIES[j%len(SERIES)]}"/>')
            parts.append(f'<text x="{x+(bw-2)/2:.1f}" y="{ax.y0-h-6:.1f}" class="barvalue tiny" '
                         f'text-anchor="middle">{value_fmt.format(v)}</text>')
        parts.append(f'<text x="{ax.x0+i*step+step/2:.1f}" y="{ax.y0+18:.1f}" '
                     f'class="tick tick-x" text-anchor="middle">{_esc(lb)}</text>')
    parts.append(f'<line x1="{ax.x0}" y1="{ax.y0}" x2="{ax.x1}" y2="{ax.y0}" class="axis"/>')
    legend = " ".join(
        f'<span class="lg"><i style="background:{SERIES[j%len(SERIES)]}"></i>{_esc(g[0])}</span>'
        for j, g in enumerate(groups))
    parts.append("</svg>")
    return f'<div class="legend">{legend}</div>' + "".join(parts)


def calibration_chart(declared: list[float], observed: list[float], ns: list[int],
                      ax: Axes | None = None) -> str:
    """Диаграмма калибровки: заявленный риск против наблюдаемой частоты."""
    ax = ax or Axes(w=520, h=380, pad_b=56)
    hi = max(max(declared), max(observed)) * 1.25
    def px(v): return ax.x0 + v / hi * (ax.x1 - ax.x0)
    def py(v): return ax.y0 - v / hi * (ax.y0 - ax.y1)
    parts = [f'<svg viewBox="0 0 {ax.w} {ax.h}" class="chart" role="img">']
    for t in _nice_ticks(0, hi):
        parts.append(f'<line x1="{ax.x0}" y1="{py(t):.1f}" x2="{ax.x1}" y2="{py(t):.1f}" class="grid"/>')
        parts.append(f'<text x="{ax.x0-9}" y="{py(t)+4:.1f}" class="tick tick-y">{t:g}</text>')
        parts.append(f'<text x="{px(t):.1f}" y="{ax.y0+18:.1f}" class="tick tick-x" '
                     f'text-anchor="middle">{t:g}</text>')
    parts.append(f'<line x1="{px(0):.1f}" y1="{py(0):.1f}" x2="{px(hi):.1f}" y2="{py(hi):.1f}" '
                 f'class="refline"/>')
    parts.append(f'<text x="{px(hi)-8:.1f}" y="{py(hi)+18:.1f}" class="serieslabel" '
                 f'text-anchor="end">идеальная калибровка</text>')
    for d, o, n in zip(declared, observed, ns):
        r = 5 + 7 * (n / max(ns)) ** 0.5
        parts.append(f'<circle cx="{px(d):.1f}" cy="{py(o):.1f}" r="{r:.1f}" '
                     f'fill="var(--series-1)" stroke="var(--surface-1)" stroke-width="2"/>')
        parts.append(f'<text x="{px(d)+r+5:.1f}" y="{py(o)+4:.1f}" class="barvalue tiny">'
                     f'n={n}</text>')
    parts.append(f'<line x1="{ax.x0}" y1="{ax.y0}" x2="{ax.x1}" y2="{ax.y0}" class="axis"/>')
    parts.append(f'<text x="{(ax.x0+ax.x1)/2:.0f}" y="{ax.h-8}" class="axislabel" '
                 f'text-anchor="middle">заявленный риск, %</text>')
    parts.append(f'<text x="14" y="{(ax.y0+ax.y1)/2:.0f}" class="axislabel" '
                 f'transform="rotate(-90 14 {(ax.y0+ax.y1)/2:.0f})">наблюдаемая доля нарушений, %</text>')
    parts.append("</svg>")
    return "".join(parts)
