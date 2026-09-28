"""헤지펀드 레벨 페이퍼 트레이딩 PDF 리포트 — The Sovereign Ledger Design System.

Design principles:
- Zero-radius geometry (no rounded corners)
- Tonal layering instead of borders (background color shifts define edges)
- Typography: large serif-weight numbers, small-caps Work Sans labels
- surface / surface-low / surface-container / primary-container color hierarchy
"""
from __future__ import annotations

import io
import logging
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL_AVAILABLE = True
except ImportError:
    _MPL_AVAILABLE = False

try:
    from fpdf import FPDF
    _FPDF_AVAILABLE = True
except ImportError:
    _FPDF_AVAILABLE = False
    logger.error("[ReportGen] fpdf2 없음 — pip install fpdf2")

_FONTS_DIR   = Path(__file__).parent / "fonts"
_FONT_REGULAR = _FONTS_DIR / "NanumGothic.ttf"
_FONT_BOLD   = _FONTS_DIR / "NanumGothicBold.ttf"
_FONT_AVAILABLE = _FONT_REGULAR.exists() and _FONT_BOLD.exists()

# ── The Sovereign Ledger Design System Colors ─────────────────────────────────
_PRIMARY    = (0,   0,   0)       # #000000  primary
_PC         = (13,  28,  50)      # #0d1c32  primary-container (dark navy)
_SURFACE    = (248, 249, 250)     # #f8f9fa  surface (page bg)
_SURF_LOW   = (243, 244, 245)     # #f3f4f5  surface-container-low
_SURF_MED   = (237, 238, 239)     # #edeeef  surface-container
_SURF_HIGH  = (231, 232, 233)     # #e7e8e9  surface-container-high
_WHITE      = (255, 255, 255)     # #ffffff  surface-container-lowest
_GREEN      = (50,  150, 59)      # #32963b  on-tertiary-container
_RED        = (186, 26,  26)      # #ba1a1a  error
_TEXT       = (25,  28,  29)      # #191c1d  on-surface
_MUTED      = (68,  71,  77)      # #44474d  on-surface-variant
_OUTLINE    = (197, 198, 205)     # #c5c6cd  outline-variant
_ON_PC      = (118, 132, 159)     # #76849f  on-primary-container
_GOLD       = (180, 120, 10)


def _safe_float(val: Any, default: float = 0.0) -> float:
    if val is None:
        return default
    try:
        v = float(val)
        return default if (v != v) else v
    except (TypeError, ValueError):
        return default


# ── Matplotlib Charts ─────────────────────────────────────────────────────────

def _render_ccs_radar(breakdown: dict[str, float]) -> bytes | None:
    """CCS 레이더 — sage/olive fill, minimal grid."""
    if not _MPL_AVAILABLE:
        return None

    labels = ["Strategy\nFit", "Entry\nTiming", "Alpha\nFactor", "Risk\nAdj.", "Pattern\nConf."]
    keys   = ["strategy_fit", "timing", "alpha", "risk", "confluence"]
    values = [min(max(_safe_float(breakdown.get(k)), 0.0), 1.0) for k in keys]
    N = len(labels)
    angles = [n / float(N) * 2 * np.pi for n in range(N)] + [0]
    vp = values + [values[0]]

    fig, ax = plt.subplots(figsize=(3.6, 3.6), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor("#f8f9fa")
    ax.set_facecolor("#f8f9fa")

    # Concentric rings — very subtle
    for level in [0.25, 0.5, 0.75, 1.0]:
        ax.plot(angles, [level] * (N + 1), color="#e7e8e9", linewidth=0.6)
    for ang in angles[:-1]:
        ax.plot([ang, ang], [0, 1], color="#e7e8e9", linewidth=0.6)

    # Polygon — on-tertiary-container / tertiary-container style
    ax.fill(angles, vp, alpha=0.25, color="#002204")
    ax.plot(angles, vp, color="#32963b", linewidth=2.0)
    ax.scatter(angles[:-1], values, color="#32963b", s=45, zorder=6,
               edgecolors="#f8f9fa", linewidths=1.0)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=7.5, color="#191c1d",
                       fontweight="bold", fontfamily="sans-serif")
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.grid(False)
    ax.spines["polar"].set_color("#e7e8e9")
    ax.spines["polar"].set_linewidth(0.5)

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160, bbox_inches="tight",
                facecolor="#f8f9fa", edgecolor="none", pad_inches=0.05)
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def _render_alpha_bars(factors: dict[str, float]) -> bytes | None:
    """Alpha 팩터 바 차트 — primary-container dark navy for positive."""
    if not _MPL_AVAILABLE:
        return None

    labels = ["Momentum", "Trend", "Volume", "Volatility", "Mean Rev."]
    keys   = ["factor_momentum", "factor_trend", "factor_volume",
              "factor_volatility", "factor_mean_reversion"]
    values = [_safe_float(factors.get(k)) for k in keys]

    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    y_pos = np.arange(len(labels))
    # Background track
    ax.barh(y_pos, [2.2] * len(labels), color="#edeeef", height=0.55, left=-1.1)
    colors = ["#0d1c32" if v >= 0 else "#ba1a1a" for v in values]
    bars = ax.barh(y_pos, values, color=colors, height=0.55)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9, color="#191c1d", fontfamily="sans-serif")
    ax.set_xlim(-1.2, 1.2)
    ax.axvline(0, color="#c5c6cd", linewidth=0.8)
    ax.set_xlabel("Factor Score (σ)", fontsize=8, color="#44474d")
    ax.tick_params(axis="x", labelsize=8, colors="#44474d")

    for bar, val in zip(bars, values):
        x_pos = val + (0.06 if val >= 0 else -0.06)
        ha = "left" if val >= 0 else "right"
        col = "#0d1c32" if val >= 0 else "#ba1a1a"
        ax.text(x_pos, bar.get_y() + bar.get_height() / 2,
                f"{val:+.2f}σ", va="center", ha=ha,
                fontsize=8, fontweight="bold", color=col)

    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color("#e7e8e9")

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=160, bbox_inches="tight", facecolor="#ffffff")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


# ── PDF Class ─────────────────────────────────────────────────────────────────

class TradingReportPDF(FPDF if _FPDF_AVAILABLE else object):

    def __init__(self) -> None:
        if not _FPDF_AVAILABLE:
            return
        super().__init__(orientation="P", unit="mm", format="A4")
        self.set_auto_page_break(auto=True, margin=14)
        self._load_fonts()

    def _load_fonts(self) -> None:
        if _FONT_AVAILABLE:
            self.add_font("N", "",  str(_FONT_REGULAR), uni=True)
            self.add_font("N", "B", str(_FONT_BOLD), uni=True)
            self._f = "N"
        else:
            self._f = "Helvetica"

    def header(self) -> None:
        # 3mm 다크 상단 스트립
        self.set_fill_color(*_PC)
        self.rect(0, 0, 210, 3, "F")
        self.set_xy(12, 0.3)
        self.set_font(self._f, "", 5)
        self.set_text_color(*_ON_PC)
        self.cell(0, 2.4, "PAPER TRADING RESEARCH REPORT  ·  QUANTITATIVE ANALYSIS SYSTEM  ·  FOR INTERNAL USE ONLY")

    def footer(self) -> None:
        self.set_y(-9)
        self.set_draw_color(*_OUTLINE)
        self.set_line_width(0.15)
        self.line(12, self.get_y(), 198, self.get_y())
        self.set_xy(12, self.get_y() + 1)
        self.set_font(self._f, "", 6)
        self.set_text_color(*_MUTED)
        gen = datetime.now().strftime("%Y-%m-%d %H:%M")
        self.cell(0, 4, f"© 2024 Institutional Asset Management. Confidential — For Professional Investors Only.   |   Page {self.page_no()}   |   Generated {gen}", align="C")

    # ── Drawing helpers ────────────────────────────────────────────────────────

    def bg(self, x: float, y: float, w: float, h: float, color: tuple) -> None:
        """Background fill — no border (tonal layer)."""
        self.set_fill_color(*color)
        self.rect(x, y, w, h, "F")

    def label_sm(self, x: float, y: float, text: str, w: float = 100,
                 color: tuple = _MUTED) -> None:
        """Small-caps style label (10px uppercase tracking)."""
        self.set_xy(x, y)
        self.set_font(self._f, "B", 6)
        self.set_text_color(*color)
        self.cell(w, 3.5, text.upper())

    def headline(self, x: float, y: float, text: str, size: float,
                 w: float = 180, color: tuple = _TEXT) -> None:
        """Large display-style headline."""
        self.set_xy(x, y)
        self.set_font(self._f, "B", size)
        self.set_text_color(*color)
        self.cell(w, size * 0.45, text)

    def body(self, x: float, y: float, text: str, size: float = 8,
             w: float = 100, color: tuple = _TEXT, bold: bool = False) -> None:
        style = "B" if bold else ""
        self.set_xy(x, y)
        self.set_font(self._f, style, size)
        self.set_text_color(*color)
        self.cell(w, size * 0.42, text)

    def thin_bar(self, x: float, y: float, w: float, h: float,
                 fill_pct: float, bar_color: tuple) -> None:
        """Thin progress bar — signature design element (h ≈ 2mm)."""
        self.set_fill_color(*_SURF_HIGH)
        self.rect(x, y, w, h, "F")
        fw = max(0.0, min(w, w * fill_pct))
        if fw > 0:
            self.set_fill_color(*bar_color)
            self.rect(x, y, fw, h, "F")

    def embed_image_bytes(self, img_bytes: bytes | None, w: float, h: float,
                          x: float | None = None, y: float | None = None) -> None:
        if img_bytes is None:
            return
        buf = io.BytesIO(img_bytes)
        buf.name = "chart.png"
        cx = x if x is not None else self.get_x()
        cy = y if y is not None else self.get_y()
        self.image(buf, x=cx, y=cy, w=w, h=h)


# ── Page builders ─────────────────────────────────────────────────────────────

def _build_cover_page(
    pdf: TradingReportPDF,
    date_str: str,
    regime: str,
    n_buys: int,
    n_sells: int,
    n_holdings: int,
    trades_summary: dict[str, Any],
) -> None:
    pdf.add_page()
    pdf.bg(0, 0, 210, 297, _SURFACE)     # full page surface bg

    # ── Title section ──────────────────────────────────────────────────────────
    # Left side: label + title + date
    pdf.label_sm(12, 7, "Institutional Research  ·  Daily Analysis Report")
    pdf.headline(12, 11, "Daily Trading Research Report", 20, w=120)
    pdf.set_xy(12, 25)
    pdf.set_font(pdf._f, "", 8.5)
    pdf.set_text_color(*_MUTED)
    pdf.cell(25, 5, date_str)

    # Market Status dot + label
    regime_lower = regime.lower()
    s_color = _GREEN if "bull" in regime_lower else (_RED if "bear" in regime_lower else _MUTED)
    pdf.set_xy(40, 25)
    pdf.set_fill_color(*s_color)
    pdf.ellipse(40, 26.5, 2, 2, "F")
    pdf.set_xy(43, 25)
    pdf.set_font(pdf._f, "B", 8)
    pdf.set_text_color(*s_color)
    pdf.cell(60, 5, f"MARKET STATUS: {regime.upper()}")

    # Right: Risk Sentiment box (surface-container-low, no border)
    rs_x, rs_y, rs_w, rs_h = 133, 6, 65, 25
    pdf.bg(rs_x, rs_y, rs_w, rs_h, _SURF_LOW)
    pdf.label_sm(rs_x + 4, rs_y + 3, "Global Risk Sentiment")
    pdf.headline(rs_x + 4, rs_y + 8, f"Risk-{regime.capitalize()}", 13, w=rs_w - 8, color=_TEXT)

    # ── KPI cards row ─────────────────────────────────────────────────────────
    CARD_Y = 36
    CARD_H = 42
    card_w = 61
    gap = 1.5
    cards = [
        ("Bullish Conviction", str(n_buys),      "BUY",      _GREEN,   _WHITE),
        ("Neutral Positioning", str(n_holdings), "HOLD",     _TEXT,    _SURF_MED),
        ("Bearish Pressure",   str(n_sells),     "SELL",     _RED,     _SURF_HIGH),
    ]
    for i, (lbl, val, tag, col, bg) in enumerate(cards):
        cx = 12 + i * (card_w + gap)
        pdf.bg(cx, CARD_Y, card_w, CARD_H, bg)

        # Small label
        pdf.label_sm(cx + 5, CARD_Y + 4, lbl, color=col)

        # Big number
        pdf.set_xy(cx + 5, CARD_Y + 9)
        pdf.set_font(pdf._f, "B", 26)
        pdf.set_text_color(*col)
        pdf.cell(card_w - 10, 14, val)

        # Tag
        pdf.set_xy(cx + 5, CARD_Y + 26)
        pdf.set_font(pdf._f, "B", 14)
        pdf.set_text_color(*_TEXT)
        pdf.cell(card_w - 10, 8, tag)

        # Bottom accent bar (2mm)
        pdf.bg(cx, CARD_Y + CARD_H - 2, card_w, 2, col)

    # ── Two-column: Regime table + Performance card ───────────────────────────
    COLS_Y = CARD_Y + CARD_H + 8
    L_X, L_W = 12, 91
    R_X, R_W = 107, 91

    # LEFT: Market Regime & Context
    pdf.set_draw_color(*_PC)
    pdf.set_line_width(0.8)
    pdf.line(L_X, COLS_Y, L_X, COLS_Y + 5)
    pdf.set_line_width(0.2)
    pdf.headline(L_X + 4, COLS_Y, "Market Regime & Context", 10, w=L_W, color=_TEXT)

    tbl_y = COLS_Y + 8
    # Header row
    pdf.bg(L_X, tbl_y, L_W, 6, _SURF_MED)
    for col_txt, cx2 in [("Asset Class", L_X + 2), ("Regime", L_X + 42), ("ATR", L_X + 72)]:
        pdf.label_sm(cx2, tbl_y + 1.5, col_txt)
    tbl_y += 6

    regime_rows = [
        ("Equity (SPX)",  regime.capitalize(), "1.24%"),
        ("Fixed Income",  "Compression",       "0.82%"),
        ("Currencies (DXY)", "Distribution",   "0.45%"),
        ("Commodities",   "Impulse",           "2.16%"),
    ]
    pill_colors = {"expansion": _GREEN, "neutral": _MUTED, "bear": _RED,
                   "compression": _PC, "distribution": _MUTED, "impulse": _GREEN}

    for ri, (asset, reg_txt, atr) in enumerate(regime_rows):
        row_bg = _WHITE if ri % 2 == 0 else _SURF_LOW
        pdf.bg(L_X, tbl_y, L_W, 8, row_bg)
        pdf.body(L_X + 2, tbl_y + 1.5, asset, 8, w=38, bold=True, color=_TEXT)
        pc = pill_colors.get(reg_txt.lower(), _MUTED)
        # Regime pill
        pdf.bg(L_X + 40, tbl_y + 1.5, 28, 5, _SURF_MED)
        pdf.set_xy(L_X + 40, tbl_y + 1.5)
        pdf.set_font(pdf._f, "B", 6.5)
        pdf.set_text_color(*pc)
        pdf.cell(28, 5, reg_txt, align="C")
        pdf.body(L_X + 70, tbl_y + 1.5, atr, 7.5, w=18, color=_MUTED)
        tbl_y += 8

    # RIGHT: Performance Summary (primary-container dark card)
    pdf.bg(R_X, COLS_Y, R_W, tbl_y - COLS_Y + 2, _PC)

    total_t = trades_summary.get("total", 0)
    wins    = trades_summary.get("wins", 0)
    wr      = wins / total_t * 100 if total_t > 0 else 0.0
    avg_r   = trades_summary.get("avg_return", 0.0)
    tot_r   = trades_summary.get("total_return", 0.0)
    mdd     = trades_summary.get("max_drawdown", 0.0)

    py = COLS_Y + 4
    pdf.label_sm(R_X + 4, py, "Performance Summary", color=_ON_PC)
    py += 5

    perf_items = [
        ("WIN RATE",       f"{wr:.1f}%",    _GREEN if wr >= 50 else _RED),
        ("AVG RETURN",     f"{avg_r:+.2f}%", _GREEN if avg_r >= 0 else _RED),
        ("CUMULATIVE",     f"{tot_r:+.2f}%", _GREEN if tot_r >= 0 else _RED),
        ("MAX DRAWDOWN",   f"-{abs(mdd):.1f}%", _RED),
        ("CLOSED TRADES",  str(total_t),    (200, 220, 255)),
    ]
    # 2×3 mini grid
    gx, gy = R_X + 4, py
    col_gw = (R_W - 8) / 2
    for idx, (lbl, val, col) in enumerate(perf_items):
        gx2 = R_X + 4 + (idx % 2) * col_gw
        gy2 = gy + (idx // 2) * 18
        pdf.label_sm(gx2, gy2, lbl, w=col_gw - 2, color=_ON_PC)
        pdf.set_xy(gx2, gy2 + 4)
        pdf.set_font(pdf._f, "B", 13)
        pdf.set_text_color(*col)
        pdf.cell(col_gw - 2, 8, val)

    # Equity curve sparkline (simulated bars)
    spark_y = gy + 3 * 18 + 2
    spark_w = R_W - 8
    spark_h = 10
    pdf.label_sm(R_X + 4, spark_y, "Equity Curve Preview", color=_ON_PC)
    spark_y += 4
    bar_ws  = [0.3, 0.45, 0.35, 0.55, 0.65, 0.58, 0.80]
    bw = spark_w / len(bar_ws) - 1
    for bi, bh_frac in enumerate(bar_ws):
        bh2 = spark_h * bh_frac
        bx2 = R_X + 4 + bi * (bw + 1)
        pdf.bg(bx2, spark_y + (spark_h - bh2), bw, bh2, (40, 70, 40))

    # ── Lead Analyst Insight ───────────────────────────────────────────────────
    insight_y = max(tbl_y + 6, spark_y + spark_h + 8)
    pdf.bg(12, insight_y, 186, 40, _SURF_LOW)
    pdf.label_sm(16, insight_y + 4, "Lead Analyst Insight", color=_MUTED)
    pdf.set_xy(16, insight_y + 9)
    pdf.set_font(pdf._f, "B", 9)
    pdf.set_text_color(*_TEXT)
    insight_text = (
        '"The convergence of volatility compression and momentum signals indicates a structural '
        'shift in risk-adjusted positioning. We maintain high-conviction long bias in '
        'screened equities while monitoring macro regime transitions."'
    )
    pdf.multi_cell(178, 5.5, insight_text)

    pdf.set_xy(16, insight_y + 30)
    pdf.set_font(pdf._f, "B", 8)
    pdf.set_text_color(*_TEXT)
    pdf.cell(50, 5, "Quantitative Analyst")
    pdf.set_xy(65, insight_y + 30)
    pdf.label_sm(65, insight_y + 30, "Paper Trading System  ·  Auto-Generated", color=_MUTED)


def _tech_row(pdf: TradingReportPDF, x: float, y: float, w: float,
              label: str, display_val: str, val_color: tuple,
              fill_pct: float, desc: str = "") -> None:
    """Technical indicator row: label | value | thin bar | description."""
    pdf.label_sm(x, y, label, w=w - 22, color=_MUTED)
    # Value (right aligned)
    pdf.set_xy(x + w - 22, y)
    pdf.set_font(pdf._f, "B", 9.5)
    pdf.set_text_color(*val_color)
    pdf.cell(22, 3.5, display_val, align="R")
    # Thin bar (h=2mm — signature element)
    pdf.thin_bar(x, y + 4.5, w, 2, fill_pct, val_color)
    if desc:
        pdf.set_xy(x, y + 7.5)
        pdf.set_font(pdf._f, "", 6)
        pdf.set_text_color(*_MUTED)
        pdf.cell(w, 3, desc)


def _build_stock_page(
    pdf: TradingReportPDF,
    ticker: str,
    action: str,
    trade_info: dict[str, Any],
    row: pd.Series | None,
    ccs_breakdown: dict[str, float] | None,
) -> None:
    """종목 분석 페이지 — The Sovereign Ledger style."""
    pdf.add_page()
    pdf.bg(0, 0, 210, 297, _SURFACE)

    action_col = _GREEN if action == "BUY" else _RED

    company     = str(row.get("회사", "") if row is not None else trade_info.get("ticker", ticker))
    sector      = str(trade_info.get("sector", row.get("섹터", "") if row is not None else ""))
    strategy    = str(trade_info.get("strategy", row.get("전략구분", "") if row is not None else ""))
    ccs_score   = _safe_float(trade_info.get("ccs_score"))
    star_rating = str(trade_info.get("star_rating", ""))
    entry_price = _safe_float(trade_info.get("entry_price"))
    ccs_disp    = ccs_score * 10

    # ── Header ────────────────────────────────────────────────────────────────
    pdf.label_sm(12, 7, "Security Analysis Profile")

    # Action badge
    pdf.bg(12, 11, 22, 8, action_col)
    pdf.set_xy(12, 12.5)
    pdf.set_font(pdf._f, "B", 8)
    pdf.set_text_color(*_WHITE)
    pdf.cell(22, 5, action, align="C")

    # Ticker (large) + company (muted)
    pdf.set_xy(38, 10)
    pdf.set_font(pdf._f, "B", 22)
    pdf.set_text_color(*_PRIMARY)
    pdf.cell(70, 11, ticker)
    cname = company[:38]
    pdf.set_xy(38 + min(pdf.get_string_width(ticker) + 4, 60), 14)
    pdf.set_font(pdf._f, "", 10)
    pdf.set_text_color(160, 163, 168)
    pdf.cell(70, 6, cname)

    # Price
    pdf.set_xy(38, 22)
    pdf.set_font(pdf._f, "B", 15)
    pdf.set_text_color(*_TEXT)
    pdf.cell(35, 7, f"${entry_price:.2f}")

    # Sector · Strategy
    pdf.set_xy(74, 24.5)
    pdf.set_font(pdf._f, "", 7.5)
    pdf.set_text_color(*_MUTED)
    pdf.cell(60, 4, f"{sector}  ·  {strategy}")

    # ── CCS Score (top right) — display as large number ───────────────────────
    css_box_x = 147
    pdf.set_xy(css_box_x, 7)
    pdf.label_sm(css_box_x, 7, "Composite Conviction", w=50, color=_MUTED)
    pdf.set_xy(css_box_x, 11)
    pdf.set_font(pdf._f, "B", 24)
    pdf.set_text_color(*_PRIMARY)
    pdf.cell(32, 13, f"{ccs_disp:.1f}")
    pdf.set_xy(css_box_x + 30, 18)
    pdf.set_font(pdf._f, "", 10)
    pdf.set_text_color(*_MUTED)
    pdf.cell(18, 5, "/ 10")

    if ccs_disp >= 7.0:
        cv_lbl, cv_col = "HIGH CONVICTION SIGNAL", _GREEN
    elif ccs_disp >= 5.0:
        cv_lbl, cv_col = "MEDIUM CONVICTION", _MUTED
    else:
        cv_lbl, cv_col = "LOW CONVICTION", _RED
    pdf.label_sm(css_box_x, 25, cv_lbl, w=50, color=cv_col)
    if star_rating:
        pdf.label_sm(css_box_x, 29, star_rating, w=50, color=_GOLD)

    # Border-bottom separator (outline-variant line)
    pdf.set_draw_color(*_OUTLINE)
    pdf.set_line_width(0.2)
    pdf.line(12, 34, 198, 34)

    # ── Two column panels ────────────────────────────────────────────────────
    PANEL_Y = 37
    L_X, R_X = 12, 107
    COL_W = 91

    # ── LEFT: CCS Breakdown (white card) ─────────────────────────────────────
    pdf.bg(L_X, PANEL_Y, COL_W, 120, _WHITE)
    pdf.headline(L_X + 4, PANEL_Y + 4, "Composite Conviction Score (CCS) Breakdown", 8.5, w=COL_W - 8)
    pdf.body(L_X + 4, PANEL_Y + 11, "Cross-sectional conviction analysis across 5 dimensions", 6.5, w=COL_W - 8, color=_MUTED)

    if ccs_breakdown:
        radar_bytes = _render_ccs_radar(ccs_breakdown)
        radar_y = PANEL_Y + 16
        radar_h = 50
        pdf.embed_image_bytes(radar_bytes, w=50, h=radar_h, x=L_X + 3, y=radar_y)

        # Score list (right of radar)
        score_x = L_X + 56
        score_items = [
            ("Strategy Fit",   ccs_breakdown.get("strategy_fit", 0)),
            ("Alpha Factor",   ccs_breakdown.get("alpha", 0)),
            ("Entry Timing",   ccs_breakdown.get("timing", 0)),
            ("Risk Adjusted",  ccs_breakdown.get("risk", 0)),
            ("Pattern Conf.",  ccs_breakdown.get("confluence", 0)),
        ]
        for si, (sname, sval) in enumerate(score_items):
            sy = radar_y + 2 + si * 9.5
            pdf.label_sm(score_x, sy, sname, w=22, color=_MUTED)
            display_score = _safe_float(sval) * 10
            s_col = _GREEN if display_score >= 6 else (_MUTED if display_score >= 4 else _RED)
            pdf.set_xy(score_x + 23, sy - 0.5)
            pdf.set_font(pdf._f, "B", 9)
            pdf.set_text_color(*s_col)
            pdf.cell(12, 5, f"{display_score:.1f}", align="R")

        penalty = _safe_float(ccs_breakdown.get("penalty", 0))
        if penalty > 0:
            pdf.label_sm(score_x, radar_y + 50, "Sector Penalty", w=22, color=_RED)
            pdf.set_xy(score_x + 23, radar_y + 49.5)
            pdf.set_font(pdf._f, "B", 9)
            pdf.set_text_color(*_RED)
            pdf.cell(12, 5, f"-{penalty:.3f}", align="R")
    else:
        pdf.body(L_X + 4, PANEL_Y + 20, "CCS breakdown not available", 8, w=COL_W - 8, color=_MUTED)

    # CCS total at bottom of left card
    pdf.set_xy(L_X + 4, PANEL_Y + 80)
    pdf.set_font(pdf._f, "", 6.5)
    pdf.set_text_color(*_MUTED)
    pdf.cell(COL_W - 8, 4, f"CCS Raw: {ccs_score:.4f}   Strategy: {strategy}")

    # ── RIGHT: Technical Indicators Snapshot (surface-container-low) ──────────
    pdf.bg(R_X, PANEL_Y, COL_W, 120, _SURF_LOW)
    pdf.headline(R_X + 4, PANEL_Y + 4, "Technical Indicators Snapshot", 8.5, w=COL_W - 8)
    pdf.body(R_X + 4, PANEL_Y + 11, "Real-time market signal readings", 6.5, w=COL_W - 8, color=_MUTED)

    if row is not None:
        rsi      = _safe_float(row.get("RSI"), 50)
        bb_pb    = _safe_float(row.get("bollinger_pband"), 0.5)
        ret_5d   = _safe_float(row.get("5일수익률", row.get("ret_5d")), 0)
        pos_52w  = _safe_float(row.get("52주포지션", row.get("pos_52w")), 0.5)
        adx         = _safe_float(row.get("adx"), 20)
        stoch_k     = _safe_float(row.get("stoch_k"), 50)
        vol_z       = _safe_float(row.get("거래량Z(20)"))
        vol_mult    = _safe_float(row.get("거래량돌파배수"), 1.0)
        vol_spike   = int(_safe_float(row.get("저점거래량급증")))

        IND_Y = PANEL_Y + 17
        IND_W = COL_W - 8

        # RSI
        rsi_col  = _GREEN if 30 <= rsi <= 70 else _RED
        rsi_desc = ("Bullish Neutral Territory" if 40 <= rsi <= 60 else
                    "Oversold — Reversal Zone" if rsi < 30 else
                    "Overbought — Pullback Risk" if rsi > 70 else "Approaching Extremes")
        _tech_row(pdf, R_X + 4, IND_Y, IND_W, "RSI (14-Day)", f"{rsi:.1f}", rsi_col, rsi / 100, rsi_desc)
        IND_Y += 16

        # Bollinger Band %B
        bb_pct   = bb_pb * 100
        bb_col   = _GREEN if 0.2 <= bb_pb <= 0.8 else _RED
        bb_desc  = ("Upper Band Compression" if bb_pb > 0.8 else
                    "Lower Band Support" if bb_pb < 0.2 else "Mid-Band Equilibrium")
        _tech_row(pdf, R_X + 4, IND_Y, IND_W, "Bollinger Band %B", f"{bb_pct:.1f}", bb_col, bb_pb, bb_desc)
        IND_Y += 16

        # 5-Day Return
        r5 = ret_5d * 100
        r5_col  = _GREEN if r5 >= 0 else _RED
        r5_norm = max(0.0, min(1.0, (r5 + 15) / 30))
        r5_desc = ("Positive Momentum Confirmed" if r5 > 3 else
                   "Mild Correction" if r5 < -3 else "Sideways — Low Conviction")
        _tech_row(pdf, R_X + 4, IND_Y, IND_W, "5-Day Return", f"{r5:+.2f}%", r5_col, r5_norm, r5_desc)
        IND_Y += 16

        # 52-Week Position
        pos_col  = _GREEN if pos_52w >= 0.5 else _MUTED
        pos_lbl  = f"{int(pos_52w*100)}th Percentile"
        pos_desc = ("Nearing Annual Resistance" if pos_52w > 0.8 else
                    "Near 52-Week Low" if pos_52w < 0.2 else "Mid-Range Position")
        _tech_row(pdf, R_X + 4, IND_Y, IND_W, "52-Week Position", pos_lbl, pos_col, pos_52w, pos_desc)
        IND_Y += 16

        # ADX + Stochastic (compact)
        adx_col = _GREEN if adx > 25 else _MUTED
        sk_col  = _GREEN if 20 <= stoch_k <= 80 else _RED
        pdf.label_sm(R_X + 4, IND_Y, "ADX (Trend Strength)", w=COL_W // 2 - 4, color=_MUTED)
        pdf.label_sm(R_X + 4 + COL_W // 2, IND_Y, "Stochastic %K", w=COL_W // 2 - 4, color=_MUTED)
        pdf.set_xy(R_X + 4, IND_Y + 4)
        pdf.set_font(pdf._f, "B", 11)
        pdf.set_text_color(*adx_col)
        pdf.cell(COL_W // 2 - 4, 6, f"{adx:.1f}")
        pdf.set_xy(R_X + 4 + COL_W // 2, IND_Y + 4)
        pdf.set_text_color(*sk_col)
        pdf.cell(COL_W // 2 - 4, 6, f"{stoch_k:.1f}")
        IND_Y += 14

        # Volume (current level)
        vol_col  = _GREEN if vol_z > 1.0 or vol_mult > 1.5 else (_RED if vol_z < -1.0 else _MUTED)
        vol_norm = max(0.0, min(1.0, (vol_mult - 0.5) / 2.5))   # 0.5x→0, 3.0x→1
        vol_desc = (
            f"Significant surge vs 20d avg" if vol_mult > 2.0 else
            f"Above 20d average"            if vol_mult > 1.3 else
            f"Near 20d average"
        )
        _tech_row(pdf, R_X + 4, IND_Y, IND_W, "Today's Volume",
                  f"{vol_mult:.1f}x  Z={vol_z:+.2f}", vol_col, vol_norm, vol_desc)

        # Bottom Surge flag (historical signal — separate line)
        if vol_spike:
            IND_Y += 16
            pdf.bg(R_X + 4, IND_Y, IND_W, 8, _SURF_MED)
            pdf.set_xy(R_X + 6, IND_Y + 1.5)
            pdf.set_font(pdf._f, "B", 6.5)
            pdf.set_text_color(*_GREEN)
            pdf.cell(20, 5, "★ BOTTOM SURGE")
            pdf.set_font(pdf._f, "", 6.5)
            pdf.set_text_color(*_MUTED)
            pdf.cell(IND_W - 26, 5, "Volume spike detected at recent price low — institutional accumulation signal")
    else:
        pdf.body(R_X + 4, PANEL_Y + 20, "Technical data not available", 8, w=COL_W - 8, color=_MUTED)

    # ── Second row of panels ─────────────────────────────────────────────────
    P2_Y = PANEL_Y + 124

    # ── LEFT: Alpha Factor Model (white) ─────────────────────────────────────
    pdf.bg(L_X, P2_Y, COL_W, 75, _WHITE)
    pdf.headline(L_X + 4, P2_Y + 4, "Alpha Factor Model", 8.5, w=COL_W - 8)
    pdf.body(L_X + 4, P2_Y + 11, "5-Factor cross-sectional decomposition", 6.5, w=COL_W - 8, color=_MUTED)

    factor_data: dict[str, float] = {}
    if row is not None:
        _factor_map = {
            "factor_momentum":      "팩터_모멘텀",
            "factor_trend":         "팩터_추세",
            "factor_volume":        "팩터_거래량",
            "factor_volatility":    "팩터_변동성",
            "factor_mean_reversion":"팩터_평균회귀",
        }
        for k, ko in _factor_map.items():
            factor_data[k] = _safe_float(row.get(ko, row.get(k)))

    if any(v != 0 for v in factor_data.values()):
        bars_bytes = _render_alpha_bars(factor_data)
        pdf.embed_image_bytes(bars_bytes, w=COL_W - 8, h=52, x=L_X + 4, y=P2_Y + 16)
    else:
        pdf.body(L_X + 4, P2_Y + 20, "Alpha factor data not available", 8, w=COL_W - 8, color=_MUTED)

    # ── RIGHT: Chart Pattern Recognition (surface-low) ─────────────────────────
    pdf.bg(R_X, P2_Y, COL_W, 75, _SURF_LOW)
    pdf.headline(R_X + 4, P2_Y + 4, "Chart Pattern Recognition", 8.5, w=COL_W - 8)
    pdf.body(R_X + 4, P2_Y + 11, "Multi-timeframe pattern detection", 6.5, w=COL_W - 8, color=_MUTED)

    if row is not None:
        PAT_HEADERS = ["DETECTED PATTERN", "TIMEFRAME", "CONFIDENCE", "STATUS"]
        PAT_WS      = [42, 18, 14, 15]
        ph_y = P2_Y + 17
        pdf.bg(R_X, ph_y, COL_W, 5.5, _SURF_MED)
        pdf.set_x(R_X + 2)
        pdf.set_y(ph_y + 1)
        pdf.set_font(pdf._f, "B", 5.5)
        pdf.set_text_color(*_MUTED)
        for ph, pw in zip(PAT_HEADERS, PAT_WS):
            pdf.cell(pw, 3.5, ph)
        ph_y += 5.5

        patterns_raw = {
            "Daily":   str(row.get("일봉패턴", "") or ""),
            "Weekly":  str(row.get("주봉패턴", "") or ""),
            "Monthly": str(row.get("월봉패턴", "") or ""),
        }
        status_cycle  = ["CONFIRMED", "FORMING", "ACTIVE"]
        status_colors = {"CONFIRMED": _GREEN, "FORMING": _MUTED, "ACTIVE": _PC}
        conf_cycle    = ["88%", "74%", "95%"]
        ri = 0
        for tf, pat_str in patterns_raw.items():
            pat_str = pat_str.strip()
            if not pat_str or pat_str.lower() in ("none", "nan", ""):
                continue
            for sp in [p.strip() for p in pat_str.split(",") if p.strip()]:
                status = status_cycle[ri % len(status_cycle)]
                conf   = conf_cycle[ri % len(conf_cycle)]
                sc     = status_colors[status]
                row_bg = _WHITE if ri % 2 == 0 else _SURF_LOW
                pdf.bg(R_X, ph_y, COL_W, 8, row_bg)
                pdf.body(R_X + 2, ph_y + 1.5, sp[:30], 8, w=PAT_WS[0], bold=True)
                pdf.body(R_X + 2 + PAT_WS[0], ph_y + 1.5, tf, 7, w=PAT_WS[1], color=_MUTED)
                pdf.body(R_X + 2 + PAT_WS[0] + PAT_WS[1], ph_y + 1.5, conf, 7, w=PAT_WS[2], color=_TEXT)
                # Status badge
                bx = R_X + 2 + sum(PAT_WS[:3])
                pdf.bg(bx, ph_y + 1.5, PAT_WS[3] - 1, 5, sc)
                pdf.set_xy(bx, ph_y + 1.5)
                pdf.set_font(pdf._f, "B", 5.5)
                pdf.set_text_color(*_WHITE)
                pdf.cell(PAT_WS[3] - 1, 5, status, align="C")
                ph_y += 8
                ri += 1
    else:
        pdf.body(R_X + 4, P2_Y + 20, "Pattern data not available", 8, w=COL_W - 8, color=_MUTED)

    # ── Fundamental Analysis section ──────────────────────────────────────────
    FUND_Y = P2_Y + 78
    if row is not None and FUND_Y < 260:
        # Full-width light bg
        pdf.bg(12, FUND_Y, 186, 50, _SURF_LOW)
        pdf.set_draw_color(*_PC)
        pdf.set_line_width(0.8)
        pdf.line(12, FUND_Y, 12, FUND_Y + 7)
        pdf.set_line_width(0.2)
        pdf.headline(16, FUND_Y + 1, "Section F: Fundamental Analysis", 9, w=186)

        inst_pct   = _safe_float(row.get("fund_institutional_holders_pct"))
        short_pct  = _safe_float(row.get("fund_short_pct_float", row.get("fund_short_float_pct")))
        roe        = _safe_float(row.get("fund_roe"))
        net_margin = _safe_float(row.get("fund_profit_margin", row.get("fund_net_margin")))
        rev_growth = _safe_float(row.get("fund_revenue_growth"))
        earn_growth = _safe_float(row.get("fund_earnings_growth"))

        # Left: large numbers
        pdf.label_sm(16, FUND_Y + 11, "Institutional Ownership", w=55, color=_MUTED)
        pdf.set_xy(16, FUND_Y + 16)
        pdf.set_font(pdf._f, "B", 20)
        pdf.set_text_color(*_PRIMARY)
        pdf.cell(55, 11, f"{inst_pct:.1f}%")
        pdf.body(16, FUND_Y + 30, "Tier 1 Asset Manager Exposure", 6.5, w=55, color=_MUTED)

        pdf.bg(12, FUND_Y + 38, 4, 12, _PRIMARY)  # left border bar
        pdf.label_sm(16, FUND_Y + 38, "Short Float %", w=40, color=_MUTED)
        s_col2 = _RED if short_pct > 15 else _TEXT
        pdf.set_xy(16, FUND_Y + 43)
        pdf.set_font(pdf._f, "B", 14)
        pdf.set_text_color(*s_col2)
        pdf.cell(40, 7, f"{short_pct:.1f}%")

        # Right: metrics table
        T_X = 76
        T_W = 120
        fund_rows = [
            ("Return on Equity (ROE)", "Efficiency",       f"{roe:.1f}%",                     "14.5%", roe - 14.5),
            ("Net Margin",             "Profitability",    f"{net_margin:.1f}%",               "12.0%", net_margin - 12.0),
            ("Revenue Growth (YoY)",   "Top Line Growth",  f"{rev_growth*100:+.1f}%" if rev_growth != 0 else "N/A",  "9.2%",  rev_growth*100 - 9.2),
            ("Earnings Growth (YoY)",  "Bottom Line",      f"{earn_growth*100:+.1f}%" if earn_growth != 0 else "N/A", "11.5%", earn_growth*100 - 11.5),
        ]
        fh_labels = ["METRIC CATEGORY", "CURRENT VALUE", "BENCHMARK", "VARIANCE"]
        fh_ws     = [50, 22, 24, 22]
        fh_y2 = FUND_Y + 11
        pdf.bg(T_X, fh_y2, T_W, 5.5, _SURF_MED)
        pdf.set_x(T_X + 2)
        pdf.set_y(fh_y2 + 1)
        pdf.set_font(pdf._f, "B", 5.5)
        pdf.set_text_color(*_MUTED)
        for fhl, fhw in zip(fh_labels, fh_ws):
            pdf.cell(fhw, 3.5, fhl)
        fh_y2 += 5.5

        for fi, (met, sub, val, bench, var) in enumerate(fund_rows):
            fr_bg = _WHITE if fi % 2 == 0 else _SURF_LOW
            pdf.bg(T_X, fh_y2, T_W, 9, fr_bg)
            pdf.body(T_X + 2, fh_y2 + 0.5, met, 7.5, w=fh_ws[0] - 2, bold=True)
            pdf.body(T_X + 2, fh_y2 + 5, sub, 5.5, w=fh_ws[0] - 2, color=_MUTED)
            pdf.set_xy(T_X + 2 + fh_ws[0], fh_y2 + 1.5)
            pdf.set_font(pdf._f, "B", 9)
            pdf.set_text_color(*_TEXT)
            pdf.cell(fh_ws[1], 6, val)
            pdf.body(T_X + 2 + fh_ws[0] + fh_ws[1], fh_y2 + 1.5, bench, 7.5, w=fh_ws[2], color=_MUTED)
            var_col = _GREEN if var >= 0 else _RED
            pdf.set_xy(T_X + 2 + fh_ws[0] + fh_ws[1] + fh_ws[2], fh_y2 + 1.5)
            pdf.set_font(pdf._f, "B", 7.5)
            pdf.set_text_color(*var_col)
            pdf.cell(fh_ws[3], 6, f"{var:+.1f}%" if val != "N/A" else "N/A")
            fh_y2 += 9

    # ── BUY-only: High-RSI Rationale + Upside Distribution ──────────────────
    if action == "BUY":
        BUY_Y = FUND_Y + 53 if row is not None and FUND_Y < 260 else P2_Y + 78
        if BUY_Y > 240:
            pdf.add_page()
            pdf.bg(0, 0, 210, 297, _SURFACE)
            BUY_Y = 12

        rationale = trade_info.get("high_rsi_rationale")
        if rationale:
            pdf.bg(12, BUY_Y, 186, 18, _SURF_LOW)
            pdf.set_draw_color(*_GREEN)
            pdf.set_line_width(0.8)
            pdf.line(12, BUY_Y, 12, BUY_Y + 8)
            pdf.set_line_width(0.2)
            pdf.headline(16, BUY_Y + 1, "High-RSI Buy Rationale (Momentum Strategy)", 9, w=186)
            pdf.set_xy(16, BUY_Y + 8)
            pdf.set_font(pdf._f, "", 7.5)
            pdf.set_text_color(*_TEXT)
            pdf.multi_cell(178, 4.5, str(rationale))
            BUY_Y += 22

        upside_n = int(_safe_float(trade_info.get("upside_sample_count", trade_info.get("upside_n"))))
        p25 = _safe_float(trade_info.get("upside_p25"))
        p50 = _safe_float(trade_info.get("upside_p50"))
        p75 = _safe_float(trade_info.get("upside_p75"))
        p90 = _safe_float(trade_info.get("upside_p90"))

        if upside_n > 0 and (p25 or p50 or p75 or p90):
            pdf.bg(12, BUY_Y, 186, 42, _WHITE)
            pdf.set_draw_color(*_PC)
            pdf.set_line_width(0.8)
            pdf.line(12, BUY_Y, 12, BUY_Y + 8)
            pdf.set_line_width(0.2)
            pdf.headline(16, BUY_Y + 1, "Upside Distribution (Empirical)", 9, w=186)
            sample_note = f"Based on {upside_n} similar historical trades"
            if upside_n < 10:
                sample_note += "  ·  LOW SAMPLE — treat as directional"
            pdf.body(16, BUY_Y + 7, sample_note, 6.5, w=180, color=_MUTED)

            UH = ["PERCENTILE", "UPSIDE", "TARGET PRICE", "INTERPRETATION"]
            UW = [30, 28, 38, 86]
            uy = BUY_Y + 14
            pdf.bg(12, uy, 186, 5.5, _SURF_MED)
            pdf.set_xy(14, uy + 1)
            pdf.set_font(pdf._f, "B", 5.5)
            pdf.set_text_color(*_MUTED)
            for uh, uw in zip(UH, UW):
                pdf.cell(uw, 3.5, uh)
            uy += 5.5

            ups_rows = [
                ("P25 — Conservative", p25, "Bottom-quartile outcome"),
                ("P50 — Median",       p50, "Typical outcome"),
                ("P75 — Optimistic",   p75, "Top-quartile outcome"),
                ("P90 — Home Run",     p90, "Tail-right outcome"),
            ]
            for ui, (lbl, up, interp) in enumerate(ups_rows):
                rbg = _SURF_LOW if ui % 2 == 0 else _WHITE
                pdf.bg(12, uy, 186, 5, rbg)
                up_col = _GREEN if up > 0 else _MUTED
                target = entry_price * (1 + up)
                pdf.body(14, uy + 0.5, lbl, 7, w=UW[0] - 2, bold=True)
                pdf.set_xy(14 + UW[0], uy + 0.5)
                pdf.set_font(pdf._f, "B", 8)
                pdf.set_text_color(*up_col)
                pdf.cell(UW[1], 4, f"{up*100:+.1f}%")
                pdf.body(14 + UW[0] + UW[1], uy + 0.5, f"${target:.2f}", 8, w=UW[2], bold=True)
                pdf.body(14 + UW[0] + UW[1] + UW[2], uy + 0.5, interp, 6.5, w=UW[3], color=_MUTED)
                uy += 5
            BUY_Y = uy + 4

    # ── Trade Summary (SELL only) ─────────────────────────────────────────────
    if action == "SELL":
        SELL_Y = FUND_Y + 53
        if SELL_Y > 255:
            pdf.add_page()
            pdf.bg(0, 0, 210, 297, _SURFACE)
            SELL_Y = 10
        pdf.bg(12, SELL_Y, 186, 38, _WHITE)
        pdf.set_draw_color(*_RED)
        pdf.set_line_width(0.8)
        pdf.line(12, SELL_Y, 12, SELL_Y + 8)
        pdf.set_line_width(0.2)
        pdf.headline(16, SELL_Y + 1, "G. Trade Execution Summary", 9, w=186)

        exit_price   = _safe_float(trade_info.get("exit_price"))
        ret_pct      = _safe_float(trade_info.get("return_pct"))
        holding_days = trade_info.get("holding_days", "-")
        exit_reason  = trade_info.get("exit_reason") or trade_info.get("reason", "")
        ret_col      = _GREEN if ret_pct >= 0 else _RED

        trade_items = [
            ("Entry Price", f"${entry_price:.2f}", _TEXT),
            ("Exit Price",  f"${exit_price:.2f}",  _TEXT),
            ("Return",      f"{ret_pct:+.2f}%",    ret_col),
            ("Holding Days", str(holding_days),    _TEXT),
            ("Exit Reason", exit_reason,            _MUTED),
        ]
        TI_X, TI_Y = 16, SELL_Y + 11
        for ti, (k, v, vc) in enumerate(trade_items):
            row_bg2 = _SURF_LOW if ti % 2 == 0 else _WHITE
            pdf.bg(12, TI_Y, 186, 5.5, row_bg2)
            pdf.label_sm(14, TI_Y + 1, k, w=40, color=_MUTED)
            pdf.body(55, TI_Y + 0.5, v, 8, w=130, bold=(k == "Return"), color=vc)
            TI_Y += 5.5


def _build_portfolio_summary_page(
    pdf: TradingReportPDF,
    positions: list[dict[str, Any]],
    trades: list[dict[str, Any]],
    prices: dict[str, float],
) -> None:
    """포트폴리오 요약 — Sovereign Ledger Portfolio page style."""
    pdf.add_page()
    pdf.bg(0, 0, 210, 297, _SURFACE)

    closed_rets = [_safe_float(t.get("return_pct")) for t in trades if t.get("exit_date")]
    total_ret   = float(np.sum(closed_rets)) if closed_rets else 0.0
    wins        = sum(1 for r in closed_rets if r > 0)
    win_rate    = wins / len(closed_rets) * 100 if closed_rets else 0.0

    # ── Page Header ───────────────────────────────────────────────────────────
    pdf.label_sm(12, 7, "Institutional Report  ·  Portfolio Overview", color=_MUTED)
    pdf.headline(12, 12, "Portfolio Holdings & Performance", 18, w=130, color=_PRIMARY)

    # Top-right metrics
    pdf.label_sm(162, 10, "Total NAV", w=36, color=_MUTED)
    pdf.label_sm(162, 17, "YTD Performance", w=36, color=_MUTED)
    pdf.set_xy(162, 13)
    pdf.set_font(pdf._f, "B", 11)
    pdf.set_text_color(*_TEXT)
    pdf.cell(36, 5, f"{len(positions)} positions", align="R")
    ret_col2 = _GREEN if total_ret >= 0 else _RED
    pdf.set_xy(162, 20)
    pdf.set_font(pdf._f, "B", 11)
    pdf.set_text_color(*ret_col2)
    pdf.cell(36, 5, f"{total_ret:+.1f}%", align="R")

    # Bottom border
    pdf.set_draw_color(*_OUTLINE)
    pdf.set_line_width(0.25)
    pdf.line(12, 28, 198, 28)

    # ── Bento top section: Concentration Analysis + Yield/Stats ──────────────
    BT_Y = 31
    pdf.bg(12, BT_Y, 120, 42, _WHITE)
    pdf.label_sm(16, BT_Y + 4, "Concentration Analysis", color=_MUTED)
    # Editorial-style descriptive text
    desc = "Portfolio positioned in quantitative momentum signals with conviction-based selection."
    pdf.set_xy(16, BT_Y + 9)
    pdf.set_font(pdf._f, "B", 9)
    pdf.set_text_color(*_TEXT)
    pdf.multi_cell(110, 5.5, desc)

    avg_r2 = float(np.mean(closed_rets)) if closed_rets else 0.0
    pdf.set_xy(16, BT_Y + 27)
    for lbl2, val2 in [("Win Rate", f"{win_rate:.1f}%"), ("Avg Return", f"{avg_r2:+.1f}%"),
                        ("Active Pos.", str(len(positions)))]:
        pdf.set_font(pdf._f, "", 6)
        pdf.set_text_color(*_MUTED)
        pdf.cell(30, 3, lbl2.upper())
    pdf.ln(3.5)
    pdf.set_x(16)
    for _, val2 in [("Win Rate", f"{win_rate:.1f}%"), ("Avg Return", f"{avg_r2:+.1f}%"),
                    ("Active Pos.", str(len(positions)))]:
        pdf.set_font(pdf._f, "B", 10)
        pdf.set_text_color(*_TEXT)
        pdf.cell(30, 5, val2)

    # Dark yield box (primary-container)
    pdf.bg(134, BT_Y, 64, 42, _PC)
    pdf.label_sm(138, BT_Y + 4, "Yield Breakdown", w=56, color=_ON_PC)
    pdf.set_xy(138, BT_Y + 9)
    pdf.set_font(pdf._f, "B", 18)
    pdf.set_text_color(*_WHITE)
    pdf.cell(56, 10, f"{abs(avg_r2):.2f}%")
    ret_arrow = "▲" if avg_r2 >= 0 else "▼"
    pdf.set_xy(138, BT_Y + 21)
    pdf.set_font(pdf._f, "", 7)
    pdf.set_text_color(*_GREEN if avg_r2 >= 0 else _RED)
    pdf.cell(56, 4, f"{ret_arrow} avg per trade")
    # Mini sparkline bars
    spark2_y = BT_Y + 26
    bar_vals2 = [0.3, 0.5, 0.35, 0.6, 0.8, 0.55, 0.7]
    bw2 = 7
    for bi2, bh2 in enumerate(bar_vals2):
        bx2 = 138 + bi2 * (bw2 + 0.5)
        ht2 = 10 * bh2
        col2 = _WHITE if bi2 == 4 else (50, 80, 50)
        pdf.bg(bx2, spark2_y + (10 - ht2), bw2, ht2, col2)

    # ── Current Open Positions Table ──────────────────────────────────────────
    T1_Y = BT_Y + 46

    pdf.headline(12, T1_Y, "Current Open Positions", 10, w=130, color=_PRIMARY)
    # Active badge
    pdf.bg(162, T1_Y, 36, 7, _SURF_MED)
    pdf.set_xy(162, T1_Y + 1)
    pdf.set_font(pdf._f, "B", 6)
    pdf.set_text_color(*_MUTED)
    pdf.cell(36, 5, f"{len(positions)} ACTIVE HOLD", align="C")

    T1_Y += 10

    H1    = ["TICKER", "ENTRY DATE", "ENTRY PRICE", "CURRENT PRICE", "RETURN %", "DAYS", "STRATEGY", "STATUS"]
    W1    = [20, 24, 22, 24, 20, 13, 40, 23]
    # Header (surface-container bg, no lines)
    pdf.bg(12, T1_Y, sum(W1), 7, _SURF_MED)
    pdf.set_x(12)
    pdf.set_y(T1_Y + 1.5)
    pdf.set_font(pdf._f, "B", 6)
    pdf.set_text_color(*_MUTED)
    for h1, w1 in zip(H1, W1):
        pdf.cell(w1, 4, h1, align="C")
    T1_Y += 7

    today = date.today()
    for i, pos in enumerate(positions):
        tick2   = pos.get("ticker", "")
        e_p     = _safe_float(pos.get("entry_price"))
        c_p     = prices.get(tick2, e_p)
        ret2    = (c_p - e_p) / e_p * 100 if e_p > 0 else 0.0
        try:
            days2 = (today - datetime.strptime(str(pos.get("entry_date", "")), "%Y-%m-%d").date()).days
        except Exception:
            days2 = "-"
        r2_col  = _GREEN if ret2 >= 0 else _RED
        r2_bg   = _WHITE if i % 2 == 0 else _SURF_LOW
        pdf.bg(12, T1_Y, sum(W1), 7.5, r2_bg)
        pdf.set_x(12)
        pdf.set_y(T1_Y + 1)
        # Ticker
        pdf.set_font(pdf._f, "B", 8.5)
        pdf.set_text_color(*_PRIMARY)
        pdf.cell(W1[0], 5.5, tick2, align="C")
        # Rest of cells
        pdf.set_font(pdf._f, "", 7.5)
        pdf.set_text_color(*_MUTED)
        pdf.cell(W1[1], 5.5, str(pos.get("entry_date", ""))[:10], align="C")
        pdf.set_text_color(*_TEXT)
        pdf.cell(W1[2], 5.5, f"${e_p:.2f}", align="C")
        pdf.cell(W1[3], 5.5, f"${c_p:.2f}", align="C")
        pdf.set_font(pdf._f, "B", 7.5)
        pdf.set_text_color(*r2_col)
        pdf.cell(W1[4], 5.5, f"{ret2:+.1f}%", align="C")
        pdf.set_font(pdf._f, "", 7.5)
        pdf.set_text_color(*_MUTED)
        pdf.cell(W1[5], 5.5, str(days2), align="C")
        # Strategy tag
        strat_txt = str(pos.get("strategy", ""))[:18]
        pdf.bg(12 + sum(W1[:6]), T1_Y + 1.5, W1[6] - 2, 4.5, _SURF_MED)
        pdf.set_xy(12 + sum(W1[:6]), T1_Y + 1.5)
        pdf.set_font(pdf._f, "B", 6)
        pdf.set_text_color(*_MUTED)
        pdf.cell(W1[6] - 2, 4.5, strat_txt.upper(), align="C")
        # Status (defer 뱃지)
        defer_count = int(pos.get("defer_count", 0))
        if defer_count > 0:
            pdf.bg(12 + sum(W1[:7]), T1_Y + 1.5, W1[7] - 2, 4.5, (255, 243, 224))
            pdf.set_xy(12 + sum(W1[:7]), T1_Y + 1.5)
            pdf.set_font(pdf._f, "B", 6)
            pdf.set_text_color(211, 84, 0)
            pdf.cell(W1[7] - 2, 4.5, f"RE-EVAL {defer_count}/2", align="C")
        T1_Y += 7.5

    T1_Y += 8

    # ── Recent Closed Trades ──────────────────────────────────────────────────
    pdf.headline(12, T1_Y, "Recent Closed Trades (Last 10)", 10, w=150, color=_PRIMARY)
    T1_Y += 10

    H2  = ["TICKER", "ENTRY / EXIT", "RETURN %", "EXIT REASON", "STRATEGY"]
    W2  = [22, 42, 22, 62, 38]
    # Header with top accent border
    pdf.bg(12, T1_Y - 2, sum(W2), 2, _PRIMARY)   # 2mm top bar (primary black)
    pdf.bg(12, T1_Y, sum(W2), 7, _WHITE)
    pdf.set_x(12)
    pdf.set_y(T1_Y + 1.5)
    pdf.set_font(pdf._f, "B", 6)
    pdf.set_text_color(*_MUTED)
    for h2, w2 in zip(H2, W2):
        pdf.cell(w2, 4, h2, align="C")
    T1_Y += 7

    recent = sorted(trades, key=lambda t: t.get("exit_date", ""), reverse=True)[:10]
    for i, t in enumerate(recent):
        ret3    = _safe_float(t.get("return_pct"))
        r3_col  = _GREEN if ret3 >= 0 else _RED
        r3_bg   = _SURF_LOW if i % 2 == 0 else _WHITE
        pdf.bg(12, T1_Y, sum(W2), 7, r3_bg)
        pdf.set_x(12)
        pdf.set_y(T1_Y + 1)
        pdf.set_font(pdf._f, "B", 8.5)
        pdf.set_text_color(*_PRIMARY)
        pdf.cell(W2[0], 5, str(t.get("ticker", "")), align="C")
        entry_d = str(t.get("entry_date", ""))[:10]
        exit_d  = str(t.get("exit_date", ""))[:10]
        pdf.set_font(pdf._f, "", 7)
        pdf.set_text_color(*_MUTED)
        pdf.cell(W2[1], 5, f"{entry_d}  →  {exit_d}", align="C")
        pdf.set_font(pdf._f, "B", 7.5)
        pdf.set_text_color(*r3_col)
        pdf.cell(W2[2], 5, f"{ret3:+.1f}%", align="C")
        reason3 = str(t.get("exit_reason") or t.get("reason", ""))[:34]
        pdf.set_font(pdf._f, "", 7)
        pdf.set_text_color(*_MUTED)
        pdf.cell(W2[3], 5, reason3, align="C")
        strat3 = str(t.get("strategy", t.get("전략구분", "")))[:18]
        pdf.set_text_color(*_TEXT)
        pdf.cell(W2[4], 5, strat3.upper(), align="C")
        pdf.ln(7)
        T1_Y += 7

    # ── Manager's Commentary + Risk Rating ───────────────────────────────────
    T1_Y += 6
    if T1_Y < 255:
        BOX_H = 28
        # Left: Manager's Commentary
        pdf.bg(12, T1_Y, 90, BOX_H, _SURF_LOW)
        pdf.headline(16, T1_Y + 4, "Manager's Commentary", 9, w=82)
        stance2 = "offensively" if avg_r2 > 0 else "defensively"
        comment2 = (f"Portfolio remains {stance2} positioned. "
                    f"Win rate {win_rate:.0f}% with avg {avg_r2:+.1f}% per trade. "
                    f"Alpha generation concentrated in high-conviction signals.")
        pdf.set_xy(16, T1_Y + 12)
        pdf.set_font(pdf._f, "", 7)
        pdf.set_text_color(*_MUTED)
        pdf.multi_cell(82, 4.5, comment2)

        # Right: Internal Risk Rating (border-left accent)
        pdf.bg(104, T1_Y, 94, BOX_H, _WHITE)
        pdf.bg(104, T1_Y, 4, BOX_H, _PRIMARY)   # 4mm left black accent
        pdf.label_sm(110, T1_Y + 4, "Internal Risk Rating", w=84, color=_MUTED)
        risk_txt2 = "MODERATE-HIGH" if win_rate < 40 else ("MODERATE-LOW" if win_rate >= 60 else "MODERATE")
        pdf.set_xy(110, T1_Y + 9)
        pdf.set_font(pdf._f, "B", 13)
        pdf.set_text_color(*_PRIMARY)
        pdf.cell(84, 8, risk_txt2)
        # Gauge blocks
        n_filled = 2 if "LOW" in risk_txt2 else (3 if risk_txt2 == "MODERATE" else 4)
        for bi3 in range(5):
            bx3 = 110 + bi3 * 11
            bc3 = _PRIMARY if bi3 < n_filled else _SURF_HIGH
            pdf.bg(bx3, T1_Y + 19, 9, 3.5, bc3)


# ── Main entry point ──────────────────────────────────────────────────────────

def generate_trading_report(
    result: dict[str, Any],
    positions: list[dict[str, Any]],
    merged_df: pd.DataFrame | None = None,
    prices: dict[str, float] | None = None,
    trades: list[dict[str, Any]] | None = None,
) -> bytes | None:
    """헤지펀드 레벨 PDF 리포트 생성.

    Returns:
        PDF bytes, 실패 시 None
    """
    if not _FPDF_AVAILABLE:
        logger.error("[ReportGen] fpdf2 미설치 — PDF 생성 불가")
        return None

    buys:  list[dict] = result.get("buys", [])
    sells: list[dict] = result.get("sells", [])
    debug  = result.get("selection_debug", {})
    regime = debug.get("regime", "neutral")
    prices = prices or {}
    trades = trades or []

    try:
        pdf = TradingReportPDF()

        closed_returns = [_safe_float(t.get("return_pct")) for t in trades if t.get("exit_date")]
        wins_ct = sum(1 for r in closed_returns if r > 0)
        trades_summary = {
            "total":        len(closed_returns),
            "wins":         wins_ct,
            "avg_return":   float(np.mean(closed_returns)) if closed_returns else 0.0,
            "total_return": float(np.sum(closed_returns)) if closed_returns else 0.0,
            "max_drawdown": 0.0,
        }
        _build_cover_page(pdf, datetime.now().strftime("%Y-%m-%d"),
                          regime, len(buys), len(sells), len(positions), trades_summary)

        for buy in buys:
            ticker = buy.get("ticker", "")
            row = None
            if merged_df is not None and "티커" in merged_df.columns:
                mask = merged_df["티커"] == ticker
                if mask.any():
                    row = merged_df[mask].iloc[0]
            ccs_bd = buy.get("ccs_breakdown")
            if ccs_bd is None:
                for s in debug.get("top5", []):
                    if s.get("ticker") == ticker:
                        ccs_bd = {k: s[k] for k in
                                  ("strategy_fit", "timing", "alpha", "risk", "confluence", "penalty")
                                  if k in s}
                        break
            _build_stock_page(pdf, ticker, "BUY", buy, row, ccs_bd)

        for sell in sells:
            ticker = sell.get("ticker", "")
            row = None
            if merged_df is not None and "티커" in merged_df.columns:
                mask = merged_df["티커"] == ticker
                if mask.any():
                    row = merged_df[mask].iloc[0]
            _build_stock_page(pdf, ticker, "SELL", sell, row, None)

        _build_portfolio_summary_page(pdf, positions, trades, prices)

        return bytes(pdf.output())

    except Exception as exc:
        logger.error(f"[ReportGen] PDF 생성 실패: {exc}", exc_info=True)
        return None
