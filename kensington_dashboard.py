"""
Kensington Health — OR Prioritization Suite
============================================
Combined dashboard merging three tools into a single multi-page application:

  Page 1 — Criteria Extraction (KEI pipeline: encounter + booking + manual lookup)
  Page 2 — BWM Weight Calculator (Best-Worst Method criterion weighting)
  Page 3 — Physician Conflict Resolution (ranked scoring with radar & contribution charts)

Run with:
    streamlit run kensington_dashboard.py

Dependencies:
    pip install streamlit pandas numpy plotly scipy python-dateutil openpyxl
"""

# ── Standard imports ──────────────────────────────────────────────────────────
import io, re, html as html_module, warnings
import math
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.optimize import linprog
import streamlit as st
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")


# ══════════════════════════════════════════════════════════════════════════════
#  BRAND TOKENS  (Kensington Health colour palette)
# ══════════════════════════════════════════════════════════════════════════════
DARK        = "#1D3D4F"   # Primary brand · headers
MID         = "#2A6B7C"   # Secondary brand · accents
ACCENT      = "#2DB89A"   # Kensington green · highlights
DEEP_GREEN  = "#1A9E85"   # Hover / active states
LIGHT_MINT  = "#F0F7F7"   # Page / slide backgrounds
PALE_TEAL   = "#E8F4F4"   # Section fills
TEAL_MIST   = "#D0E8E8"   # Borders · dividers
AMBER       = "#E8965A"   # Survey · in-progress · warnings
CHARCOAL    = "#444444"   # Body text
DARK_GRAY   = "#333333"   # Secondary text
DANGER      = "#C0392B"

KEN_COLORSCALE = [
    [0.0,  LIGHT_MINT],
    [0.25, PALE_TEAL],
    [0.5,  ACCENT],
    [0.75, MID],
    [1.0,  DARK],
]
DEV_COLORSCALE = [
    [0.0,  PALE_TEAL],
    [0.35, TEAL_MIST],
    [0.65, MID],
    [1.0,  DARK],
]
RESP_PALETTE = [DARK, MID, ACCENT, DEEP_GREEN, AMBER, DARK_GRAY, PALE_TEAL]
CI_TABLE = {1: 0.00, 2: 0.44, 3: 1.00, 4: 1.63, 5: 2.30, 6: 3.00, 7: 3.73}


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIG & GLOBAL CSS
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Kensington Health — OR Prioritization Suite",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(f"""
<style>
  html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {{
      background: {DARK};
  }}
  section[data-testid="stSidebar"] * {{
      color: white !important;
  }}
  section[data-testid="stSidebar"] label,
  section[data-testid="stSidebar"] small,
  section[data-testid="stSidebar"] p {{
      color: #cce8e2 !important;
  }}
  section[data-testid="stSidebar"] hr {{
      border-color: {MID};
  }}
  section[data-testid="stSidebar"] textarea,
  section[data-testid="stSidebar"] input {{
      background-color: #ffffff !important;
      color: {DARK} !important;
      border: 1.5px solid {MID} !important;
      border-radius: 8px !important;
  }}
  section[data-testid="stSidebar"] [data-testid="stFileUploader"] {{
      background: #ffffff !important;
      border-radius: 8px !important;
  }}
  section[data-testid="stSidebar"] [data-testid="stFileUploader"] * {{
      color: {DARK} !important;
  }}

  /* ── Main ── */
  .main .block-container {{ padding-top: 1.5rem; }}
  #MainMenu, footer {{ visibility: hidden; }}

  /* ── Cards ── */
  .kcard {{
      background: white;
      border: 1.5px solid {TEAL_MIST};
      border-radius: 12px;
      padding: 20px 24px;
      margin-bottom: 16px;
      box-shadow: 0 2px 8px rgba(29,61,79,0.07);
      height: 100%;
  }}
  .kcard-accent {{ border-left: 5px solid {ACCENT}; }}

  /* ── Section header ── */
  .section-header {{
      font-size: 20px; font-weight: 700;
      color: {DARK};
      border-bottom: 3px solid {ACCENT};
      padding-bottom: 6px;
      margin: 24px 0 16px 0;
  }}

  /* ── Metric boxes ── */
  .metric-box {{
      background: {LIGHT_MINT};
      border-radius: 10px;
      padding: 14px 18px;
      text-align: center;
      border: 1px solid {TEAL_MIST};
  }}
  .metric-box .val {{ font-size: 28px; font-weight: 700; color: {DARK}; }}
  .metric-box .lbl {{ font-size: 11px; color: {CHARCOAL}; text-transform: uppercase;
                      letter-spacing: 0.5px; margin-top: 2px; }}

  /* ── Badges ── */
  .badge {{ display: inline-block; padding: 3px 10px; border-radius: 20px;
            font-size: 12px; font-weight: 600; }}
  .badge-ok   {{ background: #d4f5ec; color: #0f6b52; }}
  .badge-warn {{ background: #fde8d0; color: #7a3d07; }}
  .badge-bad  {{ background: #fdd8d4; color: #7a1b12; }}

  /* ── Window pill ── */
  .window-pill {{
      display: inline-block; background: {PALE_TEAL};
      color: {DARK}; font-size: 0.85rem;
      padding: 6px 14px; border-radius: 8px;
      border: 1px solid {TEAL_MIST}; margin-top: 4px;
  }}

  /* ── Step nums ── */
  .step-num {{
      display: inline-flex; align-items: center; justify-content: center;
      width: 28px; height: 28px; background: {ACCENT}; color: white;
      border-radius: 50%; font-size: 0.8rem; font-weight: 700;
      margin-right: 8px; flex-shrink: 0;
  }}
  .step-label {{
      font-size: 0.95rem; font-weight: 600; color: {DARK};
      display: flex; align-items: center; margin-bottom: 0.6rem;
  }}

  /* ── Buttons ── */
  .stButton > button {{
      background: {ACCENT} !important;
      color: white !important;
      border: none !important;
      border-radius: 6px !important;
      padding: 0.6rem 2rem !important;
      font-weight: 600 !important;
  }}
  .stButton > button:hover {{ background: {DEEP_GREEN} !important; }}

  [data-testid="stDownloadButton"] > button {{
      background: white !important; color: {DARK} !important;
      border: 1px solid {TEAL_MIST} !important;
      border-radius: 6px !important; font-weight: 500 !important;
  }}
  [data-testid="stDownloadButton"] > button:hover {{
      background: {PALE_TEAL} !important;
  }}

  /* ── Metrics ── */
  [data-testid="stMetric"] {{
      background: white; border: 1px solid {TEAL_MIST};
      border-radius: 8px; padding: 1rem;
  }}

  .stAlert {{ border-radius: 6px; }}
  hr {{ border-color: {TEAL_MIST}; }}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SHARED PAGE HEADER
# ══════════════════════════════════════════════════════════════════════════════
def render_page_header(subtitle: str):
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, {DARK} 0%, {MID} 60%, {ACCENT} 100%);
                border-radius: 14px; padding: 22px 30px; margin-bottom: 24px;
                color: white; border: 1px solid {TEAL_MIST}; display:flex; align-items:center; gap:18px;">
      <div style="width:52px; height:52px; background:{LIGHT_MINT}; border-radius:10px;
                  display:flex; align-items:center; justify-content:center;
                  font-size:26px; font-weight:800; color:{DARK}; flex-shrink:0;">K</div>
      <div>
        <div style="font-size:11px; font-weight:700; letter-spacing:0.1em;
                    text-transform:uppercase; color:{ACCENT}; margin-bottom:4px;">
          Kensington Health — OR Prioritization Suite
        </div>
        <h1 style="margin:0; font-size:24px; font-weight:800; color:white;">{subtitle}</h1>
      </div>
    </div>
    """, unsafe_allow_html=True)


def section(title: str):
    st.markdown(f'<div class="section-header">{title}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR NAVIGATION
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown(f"""
    <div style="padding: 14px 0 8px 0;">
      <span style="font-size:22px; font-weight:800; letter-spacing:-0.5px;">🏥 Kensington Health</span><br>
      <span style="font-size:12px; opacity:0.75;">OR Prioritization Suite</span>
    </div>
    <hr style="margin: 10px 0 16px 0;">
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate to:",
        ["📋 Criteria Extraction", "⚖️ BWM Weight Calculator", "🏆 Physician Conflict Resolution"],
        label_visibility="collapsed",
    )

    st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-size:11px; opacity:0.65; line-height:1.7;">
      <b>Workflow overview</b><br>
      1 · <b>Criteria Extraction</b> — compute physician performance metrics from encounter,
      booking, and manual data.<br><br>
      2 · <b>BWM Weights</b> — derive criterion importance weights from stakeholder survey
      responses using the Best-Worst Method.<br><br>
      3 · <b>Conflict Resolution</b> — rank physicians using the extracted criteria and
      calculated weights, with full transparency charts.
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 — CRITERIA EXTRACTION (KEI pipeline)
# ══════════════════════════════════════════════════════════════════════════════

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW_DAYS  = 90
BLOCK_HOURS  = 7.5
GRACE_MIN    = 5

BOOKING_THRESHOLDS = {"Cataract": 16, "Retina": 8, "Cornea": 4, "Glaucoma": 8}
SUBSPECIALTIES = list(BOOKING_THRESHOLDS.keys())

CRITERIA = [
    ("leadership",        "DOVS Leadership",          "manual"),
    ("gft",               "Geographically Full-Time",  "manual"),
    ("subspecialty",      "Subspecialty Coverage",     "manual"),
    ("committee",         "Committee Membership",      "manual"),
    ("scheduling_flex",   "Scheduling Flexibility",    "manual"),
    ("or_days_elsewhere", "OR Time Elsewhere",         "manual"),
    ("avg_efficiency",    "Efficiency & Performance",  "calculated"),
]
CRITERIA_COL_ORDER  = ["physician_id"] + [c[0] for c in CRITERIA]
CRITERIA_RENAME_MAP = {c[0]: c[1] for c in CRITERIA}
CRITERIA_MANUAL_COLS = {"physician_id"} | {c[0] for c in CRITERIA if c[2] == "manual"}

ENCOUNTER_COLS = ["encounter_id", "Surgeon Provider", "Operation Room", "Surgerydate",
                  "Procedure", "Patient In Room Time", "Surgery StartTime", "Surgery StopTime"]
BOOKING_COLS   = ["Physician ID", "Appointment Date", "Room",
                  "First appointment time", "Last appointment time"] + SUBSPECIALTIES


def clean_id(series):
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def extract_encounter(file, window_start, window_end):
    raw = pd.read_excel(file)
    required = {"Surgerydate", "Surgeon Provider", "Surgery StartTime",
                "Surgery StopTime", "Patient In Room Time"}
    if not required.issubset(set(raw.columns)):
        raise ValueError(f"Wrong file for Encounter Data — missing columns: "
                         f"{sorted(required - set(raw.columns))}")
    raw["Surgerydate"] = pd.to_datetime(raw["Surgerydate"], errors="coerce")
    in_window = ((raw["Surgerydate"] >= pd.Timestamp(window_start)) &
                 (raw["Surgerydate"] <= pd.Timestamp(window_end)))
    windowed = raw.loc[in_window].copy()
    if len(windowed) == 0:
        raise ValueError(f"No Encounter data found for {window_start} to {window_end}.")
    missing = [c for c in ENCOUNTER_COLS if c not in windowed.columns]
    if missing:
        raise ValueError(f"Encounter file missing required columns: {missing}")
    clean = windowed[ENCOUNTER_COLS].copy()

    def combine_date_time(date_series, time_series):
        combined = []
        for d, t in zip(date_series, time_series):
            try:
                combined.append(datetime.combine(d, t))
            except Exception:
                combined.append(pd.NaT)
        return pd.Series(combined, index=date_series.index)

    for col in ["Patient In Room Time", "Surgery StartTime", "Surgery StopTime"]:
        clean[col] = combine_date_time(clean["Surgerydate"].dt.date, clean[col])
    clean = clean.dropna(subset=["Surgeon Provider", "Surgery StartTime", "Surgery StopTime"])
    clean["Surgeon Provider"] = clean_id(clean["Surgeon Provider"])
    clean["duration_hr"] = (clean["Surgery StopTime"] - clean["Surgery StartTime"]).dt.total_seconds() / 3600
    return clean


def extract_booking(file, window_start, window_end):
    try:
        raw = pd.read_excel(file, header=1)
        if "Physician ID" not in raw.columns:
            raw = pd.read_excel(file, header=0)
    except Exception:
        raw = pd.read_excel(file, header=0)
    required_booking = {"Physician ID", "Appointment Date", "First appointment time"}
    if not required_booking.issubset(set(raw.columns)):
        raise ValueError(f"Wrong file for Booking Data — missing columns: "
                         f"{sorted(required_booking - set(raw.columns))}")
    raw["Appointment Date"] = pd.to_datetime(raw["Appointment Date"], errors="coerce")
    in_window = ((raw["Appointment Date"] >= pd.Timestamp(window_start)) &
                 (raw["Appointment Date"] <= pd.Timestamp(window_end)))
    windowed = raw.loc[in_window].copy()
    if len(windowed) == 0:
        raise ValueError(f"No Booking data found for {window_start} to {window_end}.")
    missing = [c for c in BOOKING_COLS if c not in windowed.columns]
    if missing:
        raise ValueError(f"Booking file missing required columns: {missing}")
    clean = windowed[BOOKING_COLS].copy()
    clean["First appointment time"] = pd.to_datetime(clean["First appointment time"], errors="coerce")
    clean["Last appointment time"]  = pd.to_datetime(clean["Last appointment time"],  errors="coerce")
    clean[SUBSPECIALTIES] = clean[SUBSPECIALTIES].fillna(0).astype(int)
    clean = clean.dropna(subset=["Physician ID", "Appointment Date"])
    clean["Physician ID"]     = clean_id(clean["Physician ID"])
    clean["Appointment Date"] = pd.to_datetime(clean["Appointment Date"]).dt.date
    return clean


def calc_block_utilization(enc):
    daily = (enc.groupby(["Surgeon Provider", "Surgerydate"])["duration_hr"]
               .sum().reset_index().rename(columns={"duration_hr": "daily_used_hr"}))
    daily["daily_util"] = (daily["daily_used_hr"] / BLOCK_HOURS).clip(upper=1.0)
    result = (daily.groupby("Surgeon Provider")
                   .agg(block_utilization=("daily_util", "mean"), n_or_days=("daily_util", "count"))
                   .reset_index().rename(columns={"Surgeon Provider": "physician_id"}))
    result["block_utilization"] = result["block_utilization"].round(4)
    return result


def calc_ontime_start(enc, booking):
    enc = enc.copy(); booking = booking.copy()
    enc["Surgeon Provider"] = enc["Surgeon Provider"].astype(str).str.strip()
    booking["Physician ID"] = booking["Physician ID"].astype(str).str.strip()
    enc["_date"]     = pd.to_datetime(enc["Surgerydate"]).dt.date
    booking["_date"] = pd.to_datetime(booking["Appointment Date"]).dt.date
    first_cases = (enc.sort_values("Patient In Room Time")
                      .groupby(["Surgeon Provider", "_date"]).first().reset_index()
                      [["Surgeon Provider", "_date", "Patient In Room Time"]])
    merged = first_cases.merge(
        booking[["Physician ID", "_date", "First appointment time"]],
        left_on=["Surgeon Provider", "_date"],
        right_on=["Physician ID", "_date"], how="left")
    merged["delay_min"] = (merged["Patient In Room Time"] - merged["First appointment time"]).dt.total_seconds() / 60
    merged["on_time"]   = merged["delay_min"] <= GRACE_MIN
    result = (merged.groupby("Surgeon Provider")
                    .agg(on_time_pct=("on_time", "mean"),
                         total_first_cases=("on_time", "count"),
                         avg_delay_min=("delay_min", "mean"))
                    .reset_index().rename(columns={"Surgeon Provider": "physician_id"}))
    result["ontime_meets_85pct"] = result["on_time_pct"] >= 0.85
    result["on_time_pct"]        = result["on_time_pct"].round(4)
    result["avg_delay_min"]      = result["avg_delay_min"].round(1)
    return result


def calc_booking_adherence(booking):
    rows = []
    for physician_id, grp in booking.groupby("Physician ID"):
        sub_totals  = grp[SUBSPECIALTIES].sum()
        primary_sub = sub_totals.idxmax()
        if sub_totals[primary_sub] == 0:
            rows.append({"physician_id": physician_id, "booking_adherence": np.nan}); continue
        threshold   = BOOKING_THRESHOLDS[primary_sub]
        active_days = grp[grp[SUBSPECIALTIES].sum(axis=1) > 0]
        adherence   = (active_days[primary_sub] >= threshold).mean() if len(active_days) else np.nan
        rows.append({"physician_id": physician_id,
                     "booking_adherence": round(adherence, 4) if not np.isnan(adherence) else np.nan})
    return pd.DataFrame(rows)


def run_kei_pipeline(enc_file, booking_file, manual_file, window_start, window_end, log):
    log.append(("info", f"Window: **{window_start}** → **{window_end}**"))
    enc     = extract_encounter(enc_file, window_start, window_end)
    log.append(("ok", f"Encounter: {len(enc):,} rows"))
    booking = extract_booking(booking_file, window_start, window_end)
    log.append(("ok", f"Booking: {len(booking):,} rows"))

    enc_ids  = set(enc["Surgeon Provider"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).unique())
    book_ids = set(booking["Physician ID"].astype(str).unique())
    only_enc = enc_ids - book_ids
    if only_enc:
        log.append(("warn", f"{len(only_enc)} physician(s) in Encounter but not in Booking: {sorted(only_enc)}"))

    block      = calc_block_utilization(enc)
    ontime     = calc_ontime_start(enc, booking)
    adherence  = calc_booking_adherence(booking)
    for df_ in [block, ontime, adherence]:
        df_["physician_id"] = clean_id(df_["physician_id"])

    calculated = (block
                  .merge(ontime[["physician_id", "on_time_pct", "ontime_meets_85pct"]], on="physician_id", how="outer")
                  .merge(adherence, on="physician_id", how="outer"))
    calculated["physician_id"]   = clean_id(calculated["physician_id"])
    calculated["avg_efficiency"] = calculated[["block_utilization", "on_time_pct", "booking_adherence"]].mean(axis=1, skipna=False).round(4)

    def _eff_note(row):
        if pd.notna(row["avg_efficiency"]): return ""
        missing = []
        if pd.isna(row["block_utilization"]):  missing.append("no OR cases")
        if pd.isna(row["on_time_pct"]):        missing.append("no booking match")
        if pd.isna(row["booking_adherence"]):  missing.append("no booking data")
        return "Missing: " + ", ".join(missing) if missing else "insufficient data"

    calculated["_efficiency_note"] = calculated.apply(_eff_note, axis=1)

    manual = pd.read_excel(manual_file)
    expected = CRITERIA_MANUAL_COLS
    actual   = set(manual.columns.str.strip())
    missing_cols = expected - actual
    if missing_cols:
        raise ValueError(f"Wrong file for Manual Lookup — missing columns: {sorted(missing_cols)}")
    manual["physician_id"] = clean_id(manual["physician_id"].astype(str))
    log.append(("ok", f"Manual lookup: {len(manual)} physicians"))

    final = calculated.merge(manual, on="physician_id", how="left")
    col_order   = [c for c in CRITERIA_COL_ORDER if c in final.columns]
    display_only = [c for c in ["block_utilization", "on_time_pct", "booking_adherence", "_efficiency_note"]
                    if c in final.columns]
    final = final[col_order + display_only]
    final = final.rename(columns=CRITERIA_RENAME_MAP)
    log.append(("ok", f"Output: {len(final)} physicians × {len(final.columns)} columns ✓"))
    return final


# ── KEI Visualizations ───────────────────────────────────────────────────────

def kei_efficiency_bar(result):
    eff_col = "Efficiency & Performance"
    if eff_col not in result.columns:
        return None
    df_plot = result[["physician_id", eff_col]].dropna(subset=[eff_col]).copy()
    df_plot = df_plot.sort_values(eff_col, ascending=False)
    colors = [ACCENT if v >= 0.7 else AMBER if v >= 0.5 else DANGER
              for v in df_plot[eff_col]]
    fig = go.Figure(go.Bar(
        x=df_plot["physician_id"].astype(str), y=df_plot[eff_col],
        marker_color=colors,
        text=[f"{v:.1%}" for v in df_plot[eff_col]],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Efficiency: %{y:.1%}<extra></extra>",
    ))
    fig.add_hline(y=0.7, line_dash="dash", line_color=DARK, line_width=1.5,
                  annotation_text="Target 70%", annotation_position="top right")
    fig.update_layout(
        title="Efficiency & Performance by Physician",
        xaxis_title="Physician", yaxis_title="Efficiency Score",
        yaxis=dict(tickformat=".0%", range=[0, 1.15]),
        plot_bgcolor=LIGHT_MINT, paper_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=80),
        height=380, font=dict(family="Inter", color=DARK),
        xaxis=dict(tickangle=-35, type="category"),
    )
    fig.update_yaxes(gridcolor=TEAL_MIST)
    return fig


def kei_sub_breakdown(result):
    """Stacked bar: block utilization, on-time start, booking adherence per physician."""
    sub_cols = ["block_utilization", "on_time_pct", "booking_adherence"]
    display_names = ["Block Utilization", "On-Time Start %", "Booking Adherence"]
    available = [c for c in sub_cols if c in result.columns]
    if not available:
        return None
    df_plot = result[["physician_id"] + available].copy()
    df_plot["physician_id"] = df_plot["physician_id"].astype(str)
    fig = go.Figure()
    colors = [ACCENT, MID, DEEP_GREEN]
    for col, name, color in zip(available, display_names[:len(available)], colors):
        fig.add_trace(go.Bar(
            name=name,
            x=df_plot["physician_id"].astype(str),
            y=df_plot[col],
            marker_color=color,
            hovertemplate=f"<b>%{{x}}</b><br>{name}: %{{y:.1%}}<extra></extra>",
        ))
    fig.update_layout(
        barmode="group",
        title="Efficiency Sub-Metrics Breakdown",
        xaxis_title="Physician", yaxis_title="Score",
        yaxis=dict(tickformat=".0%", range=[0, 1.2]),
        plot_bgcolor=LIGHT_MINT, paper_bgcolor="white",
        margin=dict(l=10, r=10, t=50, b=80),
        height=380, font=dict(family="Inter", color=DARK),
        xaxis=dict(tickangle=-35, type="category"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
    )
    fig.update_yaxes(gridcolor=TEAL_MIST)
    return fig



# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 — BWM WEIGHT CALCULATOR
# ══════════════════════════════════════════════════════════════════════════════

def consistency_ratio(xi_star: float, a_BW: float) -> float:
    a_BW_int = max(1, min(9, int(round(float(a_BW)))))
    ci = CI_TABLE.get(a_BW_int, 0)
    return 0.0 if ci == 0.0 else xi_star / ci


def match_criterion(text: str, criteria: list) -> int:
    text = str(text).strip().lower()
    text = re.sub(r'[\u2010-\u2015\u2212]', '-', text)
    for i, c in enumerate(criteria):
        if c.lower() in text or text in c.lower():
            return i
    for i, c in enumerate(criteria):
        words = c.lower().split()
        if any(w in text for w in words if len(w) > 4):
            return i
    raise ValueError(f"Cannot match criterion: '{text}'")


def parse_best_col(col_name: str, criteria: list):
    m  = re.search(r'is (.+?) \(your most important', col_name, re.IGNORECASE)
    m2 = re.search(r'in comparison to (.+?)\??$', col_name, re.IGNORECASE)
    if not m or not m2:
        return None
    try:
        return match_criterion(m.group(1), criteria), match_criterion(m2.group(1), criteria)
    except ValueError:
        return None


def parse_worst_col(col_name: str, criteria: list):
    m  = re.search(r'is (.+?) in comparison to', col_name, re.IGNORECASE)
    m2 = re.search(r'in comparison to (.+?) \(your least', col_name, re.IGNORECASE)
    if not m or not m2:
        return None
    try:
        return match_criterion(m.group(1), criteria), match_criterion(m2.group(1), criteria)
    except ValueError:
        return None


@st.cache_data(show_spinner=False)
def bwm_load_and_parse(file_bytes: bytes, criteria: tuple):
    criteria = list(criteria)
    raw = pd.read_excel(io.BytesIO(file_bytes), header=0)
    n = len(criteria)
    BEST_COL  = raw.columns[1]
    WORST_COL = raw.columns[44] if len(raw.columns) > 44 else raw.columns[-1]
    best_cols, worst_cols = {}, {}
    for col in raw.columns:
        col_l = col.lower()
        if 'most important criterion' in col_l:
            res = parse_best_col(col, criteria)
            if res:
                best_cols[res] = col
        elif 'least important criterion' in col_l:
            res = parse_worst_col(col, criteria)
            if res:
                worst_cols[res] = col
    return raw, n, BEST_COL, WORST_COL, best_cols, worst_cols


def extract_vectors(row, criteria, n, BEST_COL, WORST_COL, best_cols, worst_cols):
    best_idx  = match_criterion(str(row[BEST_COL]).strip(), criteria)
    worst_idx = match_criterion(str(row[WORST_COL]).strip(), criteria)
    a_Bj = np.ones(n); a_jW = np.ones(n)
    for (b, j), col in best_cols.items():
        if b == best_idx:
            val = row[col]
            if pd.notna(val): a_Bj[j] = float(val)
    for (j, w_), col in worst_cols.items():
        if w_ == worst_idx:
            val = row[col]
            if pd.notna(val): a_jW[j] = float(val)
    a_Bj[best_idx] = 1.0; a_jW[worst_idx] = 1.0
    a_BW = a_Bj[worst_idx]
    return best_idx, worst_idx, a_Bj, a_jW, a_BW


def solve_bwm(best_idx, worst_idx, a_Bj, a_jW, n):
    c_obj = np.zeros(n + 1); c_obj[n] = 1.0
    A_ub = []; b_ub = []
    for j in range(n):
        for sign in [1, -1]:
            row_a = np.zeros(n + 1)
            row_a[best_idx] = sign; row_a[j] -= sign * a_Bj[j]; row_a[n] = -1.0
            A_ub.append(row_a); b_ub.append(0.0)
            row_b = np.zeros(n + 1)
            row_b[j] = sign; row_b[worst_idx] -= sign * a_jW[j]; row_b[n] = -1.0
            A_ub.append(row_b); b_ub.append(0.0)
    A_eq = np.zeros((1, n + 1)); A_eq[0, :n] = 1.0
    bounds = [(0.0, 1.0)] * n + [(0.0, None)]
    res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                  A_eq=A_eq, b_eq=[1.0], bounds=bounds, method='highs')
    xi_star = float(res.x[n])
    weights = np.clip(res.x[:n], 0, None)
    weights /= weights.sum()
    return weights, xi_star


@st.cache_data(show_spinner=False)
def bwm_run_pipeline(file_bytes: bytes, criteria_tuple: tuple):
    criteria = list(criteria_tuple)
    raw, n, BEST_COL, WORST_COL, best_cols, worst_cols = bwm_load_and_parse(file_bytes, criteria_tuple)
    results, errors = [], []
    prog = st.progress(0, text="Running BWM-L optimisation…")
    for idx, row in raw.iterrows():
        try:
            b, w_, aBj, ajW, a_BW = extract_vectors(row, criteria, n, BEST_COL, WORST_COL, best_cols, worst_cols)
            wts, xi = solve_bwm(b, w_, aBj, ajW, n)
            cr = consistency_ratio(xi, a_BW)
            rec = {'Respondent': idx + 1, 'Timestamp': row.get('Timestamp', ''),
                   'Best': criteria[b], 'Worst': criteria[w_],
                   'a_BW': int(round(a_BW)), 'xi_star': round(xi, 6), 'CR': round(cr, 6)}
            for c, wt in zip(criteria, wts):
                rec[c] = round(wt, 6)
            results.append(rec)
        except Exception as e:
            errors.append(f"Respondent {idx+1}: {e}")
        prog.progress((idx + 1) / len(raw), text=f"Optimising respondent {idx+1} of {len(raw)}…")
    prog.empty()
    df_ = pd.DataFrame(results)
    return df_, errors, raw


def aggregate_weights(df_, criteria):
    wm = df_[criteria].values.astype(float)
    log_wm = np.log(wm + 1e-12)
    geo = np.exp(log_wm.mean(axis=0)); geo /= geo.sum()
    gsd = np.exp(log_wm.std(axis=0, ddof=1))
    geo_err_upper = geo * (gsd - 1); geo_err_lower = geo * (1 - 1 / gsd)
    ranks = pd.Series(geo, index=criteria).rank(ascending=False).astype(int)
    agg = pd.DataFrame({
        'Criterion': criteria,
        'Geo Mean Weight': geo.round(4),
        'Geo SD Upper': geo_err_upper.round(4),
        'Geo SD Lower': geo_err_lower.round(4),
        'GSD': gsd.round(4),
        'Rank': ranks.values,
    }).sort_values('Rank').reset_index(drop=True)
    return agg, geo, wm


def cr_badge(cr):
    if cr < 0.1:   return f'<span class="badge badge-ok">CR={cr:.3f} — Excellent</span>'
    elif cr < 0.5: return f'<span class="badge badge-warn">CR={cr:.3f} — Moderate</span>'
    else:          return f'<span class="badge badge-bad">CR={cr:.3f} — Low</span>'


def hex_to_rgba(hex_color: str, alpha: float = 0.18) -> str:
    h = hex_color.lstrip('#'); r, g, b = int(h[0:2],16), int(h[2:4],16), int(h[4:6],16)
    return f"rgba({r},{g},{b},{alpha})"


def plot_weight_bar(agg_df):
    colors = px.colors.sample_colorscale(
        [[0, DARK], [0.4, MID], [0.7, ACCENT], [1.0, DEEP_GREEN]],
        [i / (len(agg_df) - 1 or 1) for i in range(len(agg_df))])
    geo_pct = agg_df['Geo Mean Weight'] * 100
    err_upper_pct = agg_df['Geo SD Upper'] * 100
    err_lower_pct = agg_df['Geo SD Lower'] * 100
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=agg_df['Criterion'], x=geo_pct, orientation='h',
        marker_color=colors,
        error_x=dict(type='data', array=err_upper_pct.tolist(),
                     arrayminus=err_lower_pct.tolist(), color='#555', thickness=1.5, width=6),
        hovertemplate='<b>%{y}</b><br>Weight: %{x:.1f}%<extra></extra>',
    ))
    for _, row in agg_df.iterrows():
        x_tip = (row['Geo Mean Weight'] + row['Geo SD Upper']) * 100
        fig.add_annotation(x=x_tip, y=row['Criterion'],
                           text=f"<b>{row['Geo Mean Weight']*100:.1f}%</b>",
                           xanchor='left', yanchor='middle', xshift=6, showarrow=False,
                           font=dict(size=11, color=DARK, family='Inter'))
    n_crit = len(agg_df); equal_pct = 100 / n_crit
    fig.add_vline(x=equal_pct, line_dash='dash', line_color=DARK, line_width=1.5)
    fig.add_annotation(x=equal_pct, y=1.0, yref='paper',
                       text=f"Equal weight (1/{n_crit} = {equal_pct:.0f}%)",
                       xanchor='left', yanchor='bottom', xshift=6, showarrow=False,
                       font=dict(size=10, color=DARK, family='Inter'),
                       bgcolor='rgba(255,255,255,0.75)')
    max_x = ((agg_df['Geo Mean Weight'] + agg_df['Geo SD Upper']) * 100).max()
    fig.update_layout(
        title='Criterion Importance Ranking (Geometric Mean)',
        xaxis_title='Weight (%)', yaxis=dict(autorange='reversed'),
        xaxis=dict(range=[0, max_x * 1.25], ticksuffix='%'),
        plot_bgcolor=LIGHT_MINT, paper_bgcolor='white',
        margin=dict(l=10, r=20, t=50, b=30), height=380,
        font=dict(family='Inter', color=DARK),
    )
    fig.update_xaxes(gridcolor=TEAL_MIST, zeroline=False)
    fig.update_yaxes(gridcolor=TEAL_MIST)
    return fig


def plot_pie(agg_df):
    fig = go.Figure(go.Pie(
        labels=agg_df['Criterion'], values=agg_df['Geo Mean Weight'],
        hole=0.38,
        marker=dict(
            colors=px.colors.sample_colorscale(
                [[0, DARK], [0.4, MID], [0.7, ACCENT], [1.0, DEEP_GREEN]],
                [i / (len(agg_df) - 1 or 1) for i in range(len(agg_df))]),
            line=dict(color='white', width=2)),
        textinfo='percent',
        hovertemplate='<b>%{label}</b><br>%{percent} (%{value:.4f})<extra></extra>',
    ))
    fig.update_layout(
        title='Weight Distribution',
        plot_bgcolor=LIGHT_MINT, paper_bgcolor='white',
        margin=dict(l=10, r=10, t=50, b=10), height=380,
        font=dict(family='Inter', color=DARK),
        legend=dict(orientation='v', font=dict(size=11)),
    )
    return fig


def plot_individual_heatmap(df_, criteria):
    wm = df_[criteria].values.astype(float)
    resp_labels = [f"R{r+1}" for r in range(len(df_))]
    fig = go.Figure(go.Heatmap(
        z=wm, x=criteria, y=resp_labels,
        colorscale=KEN_COLORSCALE,
        text=[[f"{v:.3f}" for v in row] for row in wm],
        texttemplate="%{text}",
        hovertemplate='<b>%{y}</b> — %{x}<br>Weight: %{z:.4f}<extra></extra>',
        showscale=True, colorbar=dict(title='Weight'),
    ))
    fig.update_layout(
        title="Individual Criterion Weights per Respondent",
        plot_bgcolor=LIGHT_MINT, paper_bgcolor='white',
        margin=dict(l=10, r=10, t=50, b=80),
        height=max(280, 70 * len(df_)),
        font=dict(family='Inter', color=DARK),
        xaxis=dict(tickangle=-35),
    )
    return fig


def plot_consistency_bars(df_):
    resp_labels = [f"R{r+1}" for r in range(len(df_))]
    cr_vals = df_['CR'].values
    colors = [DEEP_GREEN if v < 0.1 else AMBER if v < 0.5 else DANGER for v in cr_vals]
    fig = go.Figure(go.Bar(
        x=resp_labels, y=cr_vals, marker_color=colors,
        text=[f"{v:.3f}" for v in cr_vals], textposition='outside',
        hovertemplate='<b>%{x}</b><br>CR = %{y:.4f}<extra></extra>',
    ))
    fig.add_hline(y=0.1, line_dash='dash', line_color=DARK, line_width=2,
                  annotation_text='CR = 0.1 threshold', annotation_position='top right')
    fig.update_layout(
        title='Consistency Ratio (CR) per Respondent',
        xaxis_title='Respondent', yaxis_title='CR (lower = more consistent)',
        plot_bgcolor=LIGHT_MINT, paper_bgcolor='white',
        margin=dict(l=10, r=10, t=50, b=30), height=320,
        font=dict(family='Inter', color=DARK),
    )
    fig.update_yaxes(gridcolor=TEAL_MIST)
    return fig


def plot_deviation_heatmap(df_, criteria, raw, n, BEST_COL, WORST_COL, best_cols, worst_cols):
    R_ = len(df_); wm = df_[criteria].values.astype(float)
    dev_matrix = np.zeros((R_, n))
    raw_idx_to_pos = {int(row["Respondent"]) - 1: pos
                      for pos, (_, row) in enumerate(df_.iterrows())}
    for idx, row in raw.iterrows():
        if idx not in raw_idx_to_pos: continue
        result_pos = raw_idx_to_pos[idx]
        try:
            b_idx, w_idx, a_Bj, a_jW, _ = extract_vectors(row, criteria, n, BEST_COL, WORST_COL, best_cols, worst_cols)
            w = wm[result_pos]
            for j in range(n):
                d1 = abs(w[b_idx] - a_Bj[j] * w[j]); d2 = abs(w[j] - a_jW[j] * w[w_idx])
                dev_matrix[result_pos, j] = max(d1, d2)
        except Exception:
            pass
    resp_labels = [f"R{r+1}" for r in range(R_)]
    dmax = dev_matrix.max() if dev_matrix.max() > 0 else 1.0
    fig = go.Figure(go.Heatmap(
        z=dev_matrix, x=criteria, y=resp_labels,
        colorscale=DEV_COLORSCALE,
        hovertemplate='<b>%{y}</b> — %{x}<br>Deviation: %{z:.4f}<extra></extra>',
        colorbar=dict(title='Deviation'), showscale=True,
    ))
    for r in range(R_):
        for j in range(n):
            text_color = DARK if dev_matrix[r, j] / dmax < 0.55 else "white"
            fig.add_annotation(x=criteria[j], y=resp_labels[r],
                               text=f"{dev_matrix[r, j]:.2f}", showarrow=False,
                               font=dict(color=text_color, size=12, family='Inter'))
    fig.update_layout(
        title='Within-Response Consistency (lower = more consistent)',
        plot_bgcolor=LIGHT_MINT, paper_bgcolor='white',
        margin=dict(l=10, r=10, t=50, b=80),
        height=max(280, 70 * R_),
        font=dict(family='Inter', color=DARK),
        xaxis=dict(tickangle=-35),
    )
    return fig


def plot_radar(df_, criteria, respondent_idx=None):
    fig = go.Figure()
    categories = criteria + [criteria[0]]
    if respondent_idx is not None:
        row = df_.iloc[respondent_idx]
        vals = [row[c] for c in criteria] + [row[criteria[0]]]
        fig.add_trace(go.Scatterpolar(r=vals, theta=categories, fill='toself',
                                     name=f"R{respondent_idx+1}",
                                     line=dict(color=ACCENT, width=2),
                                     fillcolor=hex_to_rgba(ACCENT, 0.20)))
    else:
        palette = [RESP_PALETTE[i % len(RESP_PALETTE)] for i in range(len(df_))]
        for i, (_, row) in enumerate(df_.iterrows()):
            vals = [row[c] for c in criteria] + [row[criteria[0]]]
            fig.add_trace(go.Scatterpolar(r=vals, theta=categories, fill='toself',
                                         name=f"R{i+1}",
                                         line=dict(color=palette[i], width=2),
                                         fillcolor=hex_to_rgba(palette[i], 0.15)))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 0.6], gridcolor=TEAL_MIST, linecolor=TEAL_MIST),
            angularaxis=dict(gridcolor=TEAL_MIST, linecolor=TEAL_MIST),
            bgcolor=LIGHT_MINT,
        ),
        showlegend=True, title='Criterion Weight Profiles',
        paper_bgcolor='white', margin=dict(l=40, r=40, t=60, b=40), height=420,
        font=dict(family='Inter', color=DARK),
    )
    return fig


def plot_bwm_kendall_gauge(W_k):
    gauge = go.Figure(go.Indicator(
        mode="gauge+number", value=W_k,
        number={'font': {'size': 48, 'color': DARK, 'family': 'Inter'}, 'valueformat': '.3f'},
        gauge=dict(
            axis=dict(range=[0, 1], tickwidth=1, tickcolor=DARK,
                      tickfont=dict(color=DARK, size=11), dtick=0.2),
            bar=dict(color=DARK, thickness=0.25),
            bgcolor=LIGHT_MINT, borderwidth=1, bordercolor=TEAL_MIST,
            steps=[dict(range=[0, 0.4], color="#e8e8e8"),
                   dict(range=[0.4, 0.7], color="#fde8c8"),
                   dict(range=[0.7, 1.0], color=PALE_TEAL)],
            threshold=dict(line=dict(color="rgba(0,0,0,0)", width=0), thickness=0, value=0.7),
        ),
        title={'text': "Kendall's W — Inter-rater Concordance<br>"
                       "<span style='font-size:12px;color:#666'>Weak < 0.4 · Moderate 0.4–0.7 · Strong > 0.7</span>",
               'font': {'size': 15, 'color': DARK, 'family': 'Inter'}},
    ))
    angle_deg = 180 - 0.7 * 180; angle_rad = math.radians(angle_deg)
    cx, cy = 0.5, 0.195; r_arc = 0.375
    r_inner = r_arc - 0.07; r_outer = r_arc + 0.07
    x0 = cx + r_inner * math.cos(angle_rad); y0 = cy + r_inner * math.sin(angle_rad)
    x1 = cx + r_outer * math.cos(angle_rad); y1 = cy + r_outer * math.sin(angle_rad)
    xl = cx + (r_outer + 0.06) * math.cos(angle_rad); yl = cy + (r_outer + 0.06) * math.sin(angle_rad)
    gauge.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                    xref="paper", yref="paper",
                    line=dict(color=ACCENT, width=3, dash="dash"))
    gauge.add_annotation(x=xl, y=yl, xref="paper", yref="paper",
                         text="<b>threshold = 0.7</b>", showarrow=False,
                         xanchor="left", yanchor="middle",
                         font=dict(size=11, color=ACCENT, family="Inter"),
                         bgcolor="rgba(255,255,255,0.85)",
                         bordercolor=ACCENT, borderwidth=1, borderpad=4)
    gauge.update_layout(height=340, margin=dict(l=30, r=30, t=60, b=30),
                        paper_bgcolor='white', font=dict(family='Inter', color=DARK))
    return gauge


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 — PHYSICIAN CONFLICT RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════

def validate_data(df_):
    if df_.empty: return False, "Uploaded file is empty."
    if df_.shape[1] < 2: return False, "Need at least one identifier column and one criterion column."
    return True, ""


def check_data_requirements(df_):
    if df_.shape[0] < 2: return False, "Need at least 2 physicians."
    if df_.shape[1] < 3: return False, "Need at least 2 criteria columns."
    return True, ""


def preprocess_data(df_):
    """Transpose so index = criteria, columns = physicians."""
    df_ = df_.copy()
    id_col = df_.columns[0]
    df_.set_index(id_col, inplace=True)
    return df_.T.apply(pd.to_numeric, errors='coerce')


def normalize_minmax(series, higher_is_better=True):
    lo, hi = series.min(), series.max()
    if hi == lo: return pd.Series([0.5] * len(series), index=series.index)
    norm = (series - lo) / (hi - lo)
    return norm if higher_is_better else 1 - norm


def calculate_score(df_processed, method, weights, orientations):
    df_norm = df_processed.apply(
        lambda row: normalize_minmax(row, orientations.get(row.name, True)), axis=1
    )
    contributions = pd.DataFrame(index=df_norm.index, columns=df_norm.columns, dtype=float)
    total_weight  = sum(weights.values())
    for crit in df_norm.index:
        w = weights.get(crit, 0) / total_weight if total_weight else 0
        contributions.loc[crit] = df_norm.loc[crit] * w

    final_scores = contributions.sum(axis=0)
    ranks = final_scores.rank(ascending=False, method='min').astype(int)
    results = pd.DataFrame({'Final Score': final_scores, 'Rank': ranks})
    results.index.name = 'Physician'

    explanations = {}
    total_weight = sum(weights.values())
    for physician in df_norm.columns:
        lines = [f"Scoring method: {method} (normalized 0–1, then weighted sum)\n"]
        for crit in df_norm.index:
            w   = weights.get(crit, 0) / total_weight if total_weight else 0
            raw = df_processed.loc[crit, physician]
            nv  = df_norm.loc[crit, physician]
            c   = contributions.loc[crit, physician]
            lines.append(f"  {crit}: raw={raw:.3f}, norm={nv:.3f}, w={w:.4f}, contrib={c:.6f}")
        lines.append(f"\n  Final Score = {final_scores[physician]:.6f}  |  Rank = {ranks[physician]}")
        explanations[physician] = "\n".join(lines)

    return results, contributions, pd.Series(explanations), df_norm


def _hex_for_score(value, lo, hi):
    if hi <= lo: t = 0.5
    else: t = float(np.clip((value - lo) / (hi - lo), 0, 1))
    c0 = np.array([0.11, 0.24, 0.31])   # DARK
    c1 = np.array([0.18, 0.72, 0.60])   # ACCENT
    c2 = np.array([0.91, 0.59, 0.35])   # AMBER
    rgb = (c0 + (c1 - c0) * (t / 0.5)) if t < 0.5 else (c1 + (c2 - c1) * ((t - 0.5) / 0.5))
    r, g, b_ = (np.clip(rgb, 0, 1) * 255).astype(int)
    bg  = f"#{r:02x}{g:02x}{b_:02x}"
    lum = (0.299 * r + 0.587 * g + 0.114 * b_) / 255.0
    fg  = "#f8f8f8" if lum < 0.45 else "#1a1a1a"
    return bg, fg


def _surgeon_criteria_radar(surgeon_name, df_norm, score_method):
    crits = df_norm.index.tolist()
    vals  = df_norm[surgeon_name].reindex(crits).astype(float).tolist()
    crits_closed = crits + [crits[0]]; vals_closed = vals + [vals[0]]
    lo, hi = float(np.nanmin(df_norm.values)), float(np.nanmax(df_norm.values))
    span = hi - lo; pad = 0.08 * span if span > 1e-9 else 0.05
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals_closed, theta=crits_closed, fill="toself",
        fillcolor=hex_to_rgba(ACCENT, 0.30),
        line=dict(color=ACCENT, width=2),
        name="Criterion score",
        hovertemplate="%{theta}<br>Normalized score: %{r:.3f}<extra></extra>",
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[lo - pad, hi + pad],
                            tickformat=".2f", gridcolor=TEAL_MIST, linecolor=TEAL_MIST),
            angularaxis=dict(tickfont=dict(size=11), gridcolor=TEAL_MIST),
            bgcolor=LIGHT_MINT,
        ),
        showlegend=False,
        title=dict(text=f"Criteria profile — {surgeon_name}", font=dict(size=16)),
        paper_bgcolor="white",
        margin=dict(t=56, b=40, l=48, r=48), height=420,
        font=dict(family="Inter", color=DARK),
    )
    return fig


def _surgeon_contribution_bars(surgeon_name, contributions):
    crits = contributions.index.tolist()
    w = contributions[surgeon_name].reindex(crits).astype(float)
    fig = go.Figure(go.Bar(
        x=w.values, y=crits, orientation="h",
        marker_color=ACCENT,
        hovertemplate="%{y}<br>Weighted contribution: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title="Weighted score contribution by criterion",
        xaxis_title="Contribution to final score",
        yaxis=dict(autorange="reversed", tickfont=dict(size=11)),
        plot_bgcolor=LIGHT_MINT, paper_bgcolor="white",
        height=max(320, 40 * len(crits)),
        margin=dict(l=8, r=8, t=48, b=40),
        font=dict(family="Inter", color=DARK),
    )
    fig.update_xaxes(gridcolor=TEAL_MIST)
    return fig


def render_ranked_results_table(results, contributions, df_norm, explanations, calc_method):
    ordered = results.sort_values("Rank", ascending=True)
    scores = ordered["Final Score"].astype(float)
    lo, hi = float(scores.min()), float(scores.max())

    hdr1, hdr2, hdr3, hdr4 = st.columns([2.6, 2.1, 0.75, 5.0])
    hdr1.markdown("**Physician**"); hdr2.markdown("**Final Score**")
    hdr3.markdown("**Rank**");      hdr4.markdown("**Calculation breakdown**")

    for surgeon_name, row in ordered.iterrows():
        fs = float(row["Final Score"]); rk = int(row["Rank"])
        bg, fg = _hex_for_score(fs, lo, hi)
        breakdown = explanations.get(surgeon_name, "No details available.")
        preview   = breakdown[:217] + "…" if len(breakdown) > 220 else breakdown
        preview_safe = html_module.escape(preview)

        try:
            outer = st.container(border=True)
        except TypeError:
            outer = st.container()

        with outer:
            c1, c2, c3, c4 = st.columns([2.6, 2.1, 0.75, 5.0])
            c1.markdown(f"**{surgeon_name}**")
            c2.markdown(
                f'<div style="background-color:{bg};color:{fg};padding:0.45rem 0.65rem;'
                f'border-radius:6px;font-weight:600;text-align:center;">{fs:.6f}</div>',
                unsafe_allow_html=True)
            c3.markdown(f"**{rk}**")
            c4.markdown(f'<div style="font-size:0.82rem;line-height:1.35;color:#3d3d3d;">{preview_safe}</div>',
                        unsafe_allow_html=True)

            with st.expander(f"Criterion profile & calculation — {surgeon_name}", expanded=False):
                ec1, ec2 = st.columns([1.05, 1.0], gap="large")
                with ec1:
                    if surgeon_name in df_norm.columns:
                        st.plotly_chart(_surgeon_criteria_radar(surgeon_name, df_norm, calc_method),
                                        use_container_width=True)
                    else:
                        st.info("No normalized criterion data for this physician.")
                with ec2:
                    if surgeon_name in contributions.columns:
                        st.plotly_chart(_surgeon_contribution_bars(surgeon_name, contributions),
                                        use_container_width=True)
                    else:
                        st.info("No contribution data available.")
                st.markdown("**Full calculation breakdown:**")
                st.code(breakdown, language=None)


# ══════════════════════════════════════════════════════════════════════════════
#  RENDER PAGES
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
if page == "📋 Criteria Extraction":
# ─────────────────────────────────────────────────────────────────────────────
    render_page_header("Criteria Extraction — KEI Pipeline")
    st.markdown("Select the quarter start date, upload the three required files, then click **Run Pipeline**.")
    st.markdown("---")

    col_date, col_enc, col_booking, col_manual = st.columns(4)

    with col_date:
        st.markdown('<div class="step-label"><span class="step-num">1</span> Quarter Start Date</div>',
                    unsafe_allow_html=True)
        today = date.today()
        quarter_month = ((today.month - 1) // 3) * 3 + 1
        default_start = date(today.year, quarter_month, 1) - timedelta(days=90)
        col_start, col_end = st.columns(2)
        with col_start:
            window_start = st.date_input("Start Date", value=default_start)
        with col_end:
            window_end = window_start + relativedelta(months=3) - timedelta(days=1)
    if window_end <= window_start:
        st.error("End date must be after start date.")
        st.stop()
        st.markdown(
            f'<div class="window-pill">📅 {window_start.strftime("%b %d, %Y")} → {window_end.strftime("%b %d, %Y")}</div>',
            unsafe_allow_html=True)

    with col_enc:
        st.markdown('<div class="step-label"><span class="step-num">2</span> Encounter Data</div>',
                    unsafe_allow_html=True)
        enc_file = st.file_uploader("Encounter Excel", type=["xlsx"], label_visibility="collapsed", key="kei_enc")

    with col_booking:
        st.markdown('<div class="step-label"><span class="step-num">3</span> Booking Data</div>',
                    unsafe_allow_html=True)
        booking_file = st.file_uploader("Booking Excel", type=["xlsx"], label_visibility="collapsed", key="kei_booking")

    with col_manual:
        st.markdown('<div class="step-label"><span class="step-num">4</span> Manual Lookup</div>',
                    unsafe_allow_html=True)
        manual_file = st.file_uploader("Manual Lookup Excel", type=["xlsx"], label_visibility="collapsed", key="kei_manual")

    st.markdown("---")
    all_uploaded = enc_file and booking_file and manual_file
    run_btn = st.button("▶  Run Pipeline", disabled=not all_uploaded, key="kei_run")
    if not all_uploaded:
        st.caption("Upload all 3 files to enable the pipeline.")
    st.markdown("---")

    if run_btn or "kei_result" in st.session_state:
        if run_btn:
            log_ = []
            with st.spinner("Running pipeline…"):
                try:
                    result = run_kei_pipeline(enc_file, booking_file, manual_file,
                                              window_start, window_end, log_)
                    st.session_state["kei_result"] = result
                    st.session_state["kei_log"]    = log_
                    st.session_state["kei_ok"]     = True
                except Exception as e:
                    st.session_state["kei_ok"]  = False
                    st.session_state["kei_err"] = str(e)
                    st.session_state["kei_log"] = log_

        warnings_ = [msg for kind, msg in st.session_state.get("kei_log", []) if kind == "warn"]
        if not st.session_state.get("kei_ok"):
            st.error(f"⚠ Upload error: {st.session_state.get('kei_err', 'Unknown error')}")
            st.stop()
        else:
            st.success("✓ Criteria extraction complete.")
            for w in warnings_:
                st.warning(w)

        result    = st.session_state["kei_result"]
        export_df = result.drop(columns=["block_utilization", "on_time_pct",
                                          "booking_adherence", "_efficiency_note"], errors="ignore")

        st.markdown("---")
        section("1. Summary")
        m1, m2, m3, m4 = st.columns(4)
        eff_col = "Efficiency & Performance"
        with m1: st.metric("Physicians", len(result))
        with m2:
            avg_eff = result[eff_col].mean() if eff_col in result.columns else np.nan
            st.metric("Avg Efficiency", f"{avg_eff:.1%}" if not np.isnan(avg_eff) else "N/A")
        with m3:
            missing_eff = result[eff_col].isna().sum() if eff_col in result.columns else "—"
            st.metric("Missing Efficiency", missing_eff)
        with m4:
            window_label = f"{window_start.strftime('%b %d')} – {window_end.strftime('%b %d, %Y')}"
            st.metric("Window", window_label)

        st.markdown("---")
        section("2. Visualizations")

        tab_eff, tab_sub = st.tabs(["Efficiency Overview", "Sub-Metric Breakdown"])

        with tab_eff:
            fig_eff = kei_efficiency_bar(result)
            if fig_eff: st.plotly_chart(fig_eff, use_container_width=True)
            else: st.info("Efficiency & Performance column not found.")

        with tab_sub:
            fig_sub = kei_sub_breakdown(result)
            if fig_sub: st.plotly_chart(fig_sub, use_container_width=True)
            else: st.info("Sub-metric columns not available.")

        st.markdown("---")
        section("3. Output Table")

        show_breakdown = st.checkbox("Show efficiency sub-metrics (Block Util., On-Time Start, Booking Adherence)")
        with st.expander("ℹ How are these sub-metrics calculated?"):
            st.markdown("""
                **Block Utilization**
                Hours used ÷ hours allocated per OR day (assumes 7.5-hour OR day). Higher is better.

                **On-Time Start**
                Percentage of first cases where Patient In Room Time is within 5 minutes of the scheduled start time. Benchmark: 85%. Higher is better.

                **Booking Adherence**
                Percentage of active OR days meeting the subspecialty-specific minimum booking threshold
                (e.g. Cataract surgeons: 16 cases/day). Higher is better.
                """)

        def style_efficiency(val):
            if pd.isna(val): return f"background-color:{PALE_TEAL}; color:{DANGER}"
            elif val >= 0.7: return "background-color:#d4f5ec; color:#0f6b52"
            elif val >= 0.5: return f"background-color:#fef3c7; color:#92400e"
            else:            return f"background-color:#fdd8d4; color:#7a1b12"

        if show_breakdown and all(c in result.columns for c in ["block_utilization", "on_time_pct", "booking_adherence"]):
            base_cols = list(export_df.columns)
            sub_cols  = ["block_utilization", "on_time_pct", "booking_adherence"]
            eff_idx   = base_cols.index(eff_col) + 1 if eff_col in base_cols else len(base_cols)
            display_cols = base_cols[:eff_idx] + sub_cols + base_cols[eff_idx:]
            breakdown_df = export_df.copy()
            for c in sub_cols: breakdown_df[c] = result[c]
            breakdown_df = breakdown_df[display_cols]
            styled = breakdown_df.style.applymap(style_efficiency, subset=[eff_col] if eff_col in breakdown_df.columns else []).format(
                {eff_col: lambda x: f"{x:.1%}" if not pd.isna(x) else "—",
                 "block_utilization": lambda x: f"{x:.1%}" if not pd.isna(x) else "—",
                 "on_time_pct":       lambda x: f"{x:.1%}" if not pd.isna(x) else "—",
                 "booking_adherence": lambda x: f"{x:.1%}" if not pd.isna(x) else "—"}, na_rep="—")
        else:
            styled = export_df.style.applymap(style_efficiency, subset=[eff_col] if eff_col in export_df.columns else []).format(
                {eff_col: lambda x: f"{x:.1%}" if not pd.isna(x) else "—"}, na_rep="—")

        st.dataframe(styled, use_container_width=True, height=380, hide_index=True)

        if "_efficiency_note" in result.columns:
            missing_rows = result[result["_efficiency_note"] != ""].copy()
            if not missing_rows.empty:
                id_col = "physician_id" if "physician_id" in missing_rows.columns else missing_rows.columns[0]
                with st.expander(f"⚠ {len(missing_rows)} physician(s) have missing Efficiency data"):
                    st.dataframe(missing_rows[[id_col, "_efficiency_note"]].rename(
                        columns={id_col: "Physician ID", "_efficiency_note": "Reason"}),
                        use_container_width=True, hide_index=True)

        st.markdown("---")
        section("4. Export")
        csv_bytes = export_df.to_csv(index=False).encode("utf-8")
        filename  = f"or_prioritization_{window_start.strftime('%Y%m%d')}_{window_end.strftime('%Y%m%d')}.csv"
        col_dl, col_info = st.columns([1, 2])
        with col_dl:
            st.download_button("⬇  Download CSV", data=csv_bytes, file_name=filename,
                               mime="text/csv", use_container_width=True)
        with col_info:
            st.markdown(f"""
            <div style="padding:0.6rem 1rem; background:{PALE_TEAL}; border-radius:8px;
                        border:1px solid {TEAL_MIST}; font-size:0.88rem; color:{DARK};">
                <strong>File:</strong> <code>{filename}</code><br>
                <strong>Rows:</strong> {len(result)} physicians &nbsp;|&nbsp;
                <strong>Columns:</strong> {len(export_df.columns)}
            </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚖️ BWM Weight Calculator":
# ─────────────────────────────────────────────────────────────────────────────
    render_page_header("BWM Criterion Weight Calculator")

    # Sidebar inputs for this page
    with st.sidebar:
        st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)
        st.markdown("**Step 1 — Upload Survey Responses**")
        bwm_uploaded = st.file_uploader(
            "Upload the Google Form responses (.xlsx)", type=["xlsx"],
            help="Export your Google Form responses as Excel (.xlsx) and upload here.", key="bwm_upload")
        st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)
        st.markdown("**Step 2 — Define Criteria**")
        default_criteria_txt = "\n".join([
            "DOVS Leadership", "Geographically Full-Time", "Subspecialty Coverage",
            "Committee Membership", "Efficiency & Performance", "OR Time Elsewhere", "Scheduling Flexibility"])
        criteria_text = st.text_area("One criterion per line:", value=default_criteria_txt,
                                     height=180, key="bwm_criteria")
        bwm_criteria = [c.strip() for c in criteria_text.strip().split("\n") if c.strip()]
        st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)
        bwm_run_btn = st.button("▶  Calculate Weights", use_container_width=True,
                                disabled=(bwm_uploaded is None), key="bwm_run")
        if bwm_uploaded is None:
            st.info("Upload a survey file above to get started.", icon="ℹ️")
        st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)
        st.markdown(f"""
        <div style="font-size:11px; opacity:0.65; line-height:1.6;">
          <b>About BWM-L</b><br>
          Best-Worst Method — Linear formulation (Rezaei, 2016). Solves a linear program per
          respondent. Aggregated via geometric mean. CR = ξ* / CI(a_BW) — lower is better;
          CR &lt; 0.1 is acceptable.
        </div>""", unsafe_allow_html=True)

    if "bwm_results" not in st.session_state:
        st.session_state.bwm_results = None

    if bwm_run_btn and bwm_uploaded is not None:
        with st.spinner("Running BWM-L pipeline…"):
            try:
                file_bytes = bwm_uploaded.read()
                df_results, errors, raw_df = bwm_run_pipeline(file_bytes, tuple(bwm_criteria))
                _, _, BEST_COL, WORST_COL, best_cols, worst_cols = bwm_load_and_parse(
                    file_bytes, tuple(bwm_criteria))
                agg_df_, geo_mean, weight_matrix = aggregate_weights(df_results, bwm_criteria)
                st.session_state.bwm_results = {
                    "df": df_results, "errors": errors, "raw": raw_df,
                    "agg": agg_df_, "geo": geo_mean, "wm": weight_matrix,
                    "BEST_COL": BEST_COL, "WORST_COL": WORST_COL,
                    "best_cols": best_cols, "worst_cols": worst_cols,
                    "criteria": bwm_criteria, "file_bytes": file_bytes,
                }
            except Exception as e:
                st.error(f"Pipeline failed: {e}")

    if st.session_state.bwm_results is None:
        c1_, c2_, c3_ = st.columns(3)
        for col_, num_, title_, body_ in [
            (c1_, "1", "Upload Responses",
             "Export your Google Form survey as Excel (.xlsx) and upload it in the sidebar."),
            (c2_, "2", "Confirm Criteria",
             "Verify the criteria in the sidebar match the order used in your survey form."),
            (c3_, "3", "Calculate & Explore",
             "Click 'Calculate Weights' to run BWM optimisation and explore interactive charts."),
        ]:
            col_.markdown(f"""
            <div class="kcard kcard-accent">
              <div style="font-size:32px;font-weight:800;color:{ACCENT};">{num_}</div>
              <div style="font-size:16px;font-weight:700;color:{DARK};margin:4px 0 8px 0;">{title_}</div>
              <div style="font-size:13px;color:#555;line-height:1.5;">{body_}</div>
            </div>""", unsafe_allow_html=True)
        st.stop()

    R_ = st.session_state.bwm_results
    df_bwm    = R_["df"]
    agg_df_   = R_["agg"]
    criteria_ = R_["criteria"]
    n_crit    = len(criteria_)

    if R_["errors"]:
        with st.expander(f"⚠️ {len(R_['errors'])} respondent(s) had errors", expanded=False):
            for e_ in R_["errors"]: st.warning(e_)

    avg_cr_  = df_bwm['CR'].mean()
    top_crit = agg_df_.iloc[0]['Criterion']
    top_wt_  = agg_df_.iloc[0]['Geo Mean Weight']
    if len(df_bwm) > 1:
        ranks_mat = np.argsort(np.argsort(-R_["wm"], axis=1), axis=1) + 1
        S_ = np.sum((ranks_mat.sum(axis=0) - ranks_mat.sum(axis=0).mean()) ** 2)
        W_k_ = 12 * S_ / (len(df_bwm) ** 2 * (n_crit ** 3 - n_crit))
    else:
        W_k_ = None

    m1_, m2_, m3_, m4_, m5_ = st.columns(5)
    for col_, val_, lbl_ in zip(
        [m1_, m2_, m3_, m4_, m5_],
        [str(len(df_bwm)), str(n_crit), f"{avg_cr_:.3f}", f"{top_wt_:.3f}",
         f"{W_k_:.3f}" if W_k_ else "N/A"],
        ["Respondents", "Criteria", "Avg CR", "Top Weight", "Kendall's W"],
    ):
        col_.metric(label=lbl_, value=val_)

    if W_k_ is not None:
        concord = "Strong" if W_k_ > 0.7 else "Moderate" if W_k_ > 0.4 else "Weak"
        st.info(f"**Kendall's W = {W_k_:.3f}** ({concord} agreement among respondents).", icon="ℹ️")

    st.markdown("---")
    tab_b1, tab_b2, tab_b3, tab_b4 = st.tabs(["📊 Aggregated Weights", "👥 Individual Respondents",
                                                "🔍 Consistency Diagnostics", "⬇️ Export"])

    with tab_b1:
        section("Aggregated Criterion Weights")
        st.caption("Weights aggregated via **geometric mean**. Error bars = geometric standard deviation.")
        c_bar_, c_pie_ = st.columns([3, 2])
        with c_bar_: st.plotly_chart(plot_weight_bar(agg_df_), use_container_width=True)
        with c_pie_: st.plotly_chart(plot_pie(agg_df_), use_container_width=True)
        st.markdown("#### Detailed Results Table")
        disp = agg_df_[['Criterion', 'Geo Mean Weight', 'GSD', 'Rank']].copy()
        disp = disp.rename(columns={'Geo Mean Weight': 'Weight (Geo Mean)', 'GSD': 'Geo Std Dev (×)'})
        disp['Weight (Geo Mean)'] = disp['Weight (Geo Mean)'].apply(lambda x: f"{x:.4f}")
        disp['Geo Std Dev (×)']   = disp['Geo Std Dev (×)'].apply(lambda x: f"{x:.3f}×")
        st.dataframe(disp, use_container_width=True, hide_index=True)

    with tab_b2:
        section("Individual Respondent Results")
        col_h_, col_r_ = st.columns([3, 2])
        with col_h_:
            st.plotly_chart(plot_individual_heatmap(df_bwm, criteria_), use_container_width=True)
        with col_r_:
            resp_options_ = ["All respondents overlaid"] + [
                f"R{i+1} — {df_bwm.iloc[i]['Best']} best" for i in range(len(df_bwm))]
            chosen_ = st.selectbox("Radar profile to display:", resp_options_, key="bwm_radar_sel")
            if chosen_ == "All respondents overlaid":
                st.plotly_chart(plot_radar(df_bwm, criteria_), use_container_width=True)
            else:
                r_idx_ = resp_options_.index(chosen_) - 1
                st.plotly_chart(plot_radar(df_bwm, criteria_, r_idx_), use_container_width=True)

        st.markdown("#### Per-Respondent Detail")
        for i_, row_ in df_bwm.iterrows():
            with st.expander(
                f"Respondent {row_['Respondent']}  |  Best: **{row_['Best']}**  ·  "
                f"Worst: **{row_['Worst']}**  " + cr_badge(row_['CR']), expanded=False
            ):
                cw_ = pd.DataFrame({'Criterion': criteria_,
                                    'Weight': [f"{row_[c]:.4f}" for c in criteria_]})
                st.dataframe(cw_[['Criterion', 'Weight']], hide_index=True, use_container_width=True)
                ind_colors_ = px.colors.sample_colorscale(
                    [[0, DARK], [0.4, MID], [0.7, ACCENT], [1.0, DEEP_GREEN]],
                    [j / (len(criteria_) - 1 or 1) for j in range(len(criteria_))])
                fig_ind_ = go.Figure(go.Bar(
                    x=criteria_, y=[row_[c] for c in criteria_],
                    marker_color=ind_colors_,
                    text=[f"{row_[c]:.3f}" for c in criteria_],
                    textposition='outside',
                    marker_line_color=TEAL_MIST, marker_line_width=0.5,
                ))
                fig_ind_.update_layout(
                    yaxis_title='Weight', plot_bgcolor=LIGHT_MINT, paper_bgcolor='white',
                    height=260, margin=dict(l=10, r=10, t=20, b=80),
                    font=dict(family='Inter', color=DARK), xaxis=dict(tickangle=-30))
                fig_ind_.update_yaxes(gridcolor=TEAL_MIST)
                st.plotly_chart(fig_ind_, use_container_width=True)

    with tab_b3:
        section("Consistency Diagnostics")
        st.caption("**CR** (Consistency Ratio) < 0.1 is acceptable per Rezaei (2015).")
        c3a_, c3b_ = st.columns(2)
        with c3a_:
            st.plotly_chart(plot_consistency_bars(df_bwm), use_container_width=True)
        with c3b_:
            st.markdown("#### Respondent Consistency Summary")
            for _, row_ in df_bwm.iterrows():
                icon_, colour_ = ("✅", PALE_TEAL) if row_['CR'] < 0.1 else \
                                 ("⚠️", "#fde8c8") if row_['CR'] < 0.5 else ("❌", "#fdd8d4")
                st.markdown(f"""
                <div style="background:{colour_}; border-radius:8px; padding:10px 14px;
                            margin-bottom:8px; font-size:13px;">
                  <b>{icon_} R{int(row_['Respondent'])}</b> &nbsp;
                  Best: <i>{row_['Best']}</i> &nbsp;|&nbsp;
                  Worst: <i>{row_['Worst']}</i> &nbsp;|&nbsp;
                  CR = <b>{row_['CR']:.4f}</b>
                </div>""", unsafe_allow_html=True)

        st.markdown("#### Within-Response Deviation Heatmap")
        try:
            st.plotly_chart(
                plot_deviation_heatmap(df_bwm, criteria_, R_["raw"], n_crit,
                                       R_["BEST_COL"], R_["WORST_COL"],
                                       R_["best_cols"], R_["worst_cols"]),
                use_container_width=True)
        except Exception as e_:
            st.warning(f"Could not compute deviation heatmap: {e_}")

        if W_k_ is not None:
            st.markdown("#### Inter-Rater Concordance")
            st.plotly_chart(plot_bwm_kendall_gauge(W_k_), use_container_width=True)

    with tab_b4:
        section("Export Results")
        sorted_order_ = np.argsort(R_["geo"])[::-1]
        df_csv_ = pd.DataFrame({'Criterion': [criteria_[i] for i in sorted_order_],
                                 'Weight': [round(R_["geo"][i], 4) for i in sorted_order_]})
        st.download_button("⬇️  Download Aggregated Weights (CSV)",
                           data=df_csv_.to_csv(index=False).encode(),
                           file_name="BWM_Weights_Summary.csv", mime="text/csv",
                           use_container_width=True)
        st.markdown("---")
        full_cols_ = ['Respondent', 'Timestamp', 'Best', 'Worst', 'a_BW', 'xi_star', 'CR'] + criteria_
        full_cols_ = [c for c in full_cols_ if c in df_bwm.columns]
        st.download_button("⬇️  Download Full Respondent Results (CSV)",
                           data=df_bwm[full_cols_].to_csv(index=False).encode(),
                           file_name="BWM_Full_Results.csv", mime="text/csv",
                           use_container_width=True)
        st.markdown("---")
        st.markdown("#### Preview — Aggregated Weights")
        st.dataframe(agg_df_, use_container_width=True, hide_index=True)
        st.markdown("#### Preview — Individual Results")
        st.dataframe(df_bwm[full_cols_], use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
elif page == "🏆 Physician Conflict Resolution":
# ─────────────────────────────────────────────────────────────────────────────
    render_page_header("Physician Conflict Resolution — Priority Ranking")

    with st.sidebar:
        st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)
        st.markdown("**Upload Physician Data (CSV)**")
        st.caption("CSV: first column = physician ID; remaining columns = criterion scores.")
        pcr_uploaded = st.file_uploader("Upload CSV", type=["csv"], key="pcr_upload")
        st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)
        st.markdown("**Upload Weights (CSV)**")
        st.caption("CSV with columns 'Criterion' and 'Weight'.")
        weights_file = st.file_uploader("Upload Weights CSV", type=["csv"], key="pcr_weights")
        st.markdown("<hr style='margin:16px 0;'>", unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:11px; opacity:0.65; line-height:1.6;">
          <b>Tip:</b> Export the CSV from the Criteria Extraction page, then supply the
          weights CSV from the BWM calculator for a fully integrated end-to-end pipeline.
        </div>""", unsafe_allow_html=True)

    if pcr_uploaded is not None:
        try:
            df_pcr = pd.read_csv(pcr_uploaded)
            is_valid, err_msg = validate_data(df_pcr)
            if not is_valid:
                st.error(f"Validation Error: {err_msg}")
            else:
                is_valid_req, req_msg = check_data_requirements(df_pcr)
                if not is_valid_req:
                    st.error(f"Data Requirement Error: {req_msg}")
                else:
                    df_processed = preprocess_data(df_pcr)

                    section("1. Data Preview (Physicians as Rows)")
                    df_preview = df_processed.T
                    if df_preview.isnull().values.any():
                        st.warning("Non-numeric values found and converted to NaN. Missing values treated as 0.")
                        st.dataframe(df_preview.style.highlight_null(color=PALE_TEAL))
                        df_processed = df_processed.fillna(0)
                    else:
                        st.dataframe(df_preview, height=220)

                    # Weights
                    weights = {}
                    orientations = {}

                    if weights_file is not None:
                        try:
                            w_df_ = pd.read_csv(weights_file)
                            crit_col_ = next((c for c in w_df_.columns if 'crit' in c.lower()), None)
                            val_col_  = next((c for c in w_df_.columns if 'weight' in c.lower() or 'val' in c.lower()), None)
                            if not crit_col_ or not val_col_:
                                st.error("Weights CSV must have 'Criterion' and 'Weight' columns.")
                            else:
                                weights_loaded = dict(zip(w_df_[crit_col_], w_df_[val_col_]))
                                data_criteria  = set(df_processed.index)
                                missing_w      = data_criteria - set(weights_loaded.keys())
                                if missing_w:
                                    st.error(f"Missing weights for: {', '.join(missing_w)}")
                                else:
                                    st.success("✓ Weights loaded.")
                                    weights = weights_loaded
                                    st.markdown("#### Orientation Settings")
                                    st.caption("Check = Higher is Better for that criterion.")
                                    orient_cols = st.columns(min(4, len(df_processed.index)))
                                    for i_, crit_ in enumerate(df_processed.index):
                                        with orient_cols[i_ % len(orient_cols)]:
                                            orientations[crit_] = st.checkbox(
                                                crit_, value=True, key=f"orient_{crit_}")
                        except Exception as e_:
                            st.error(f"Error reading weights file: {e_}")
                    else:
                        st.info("Upload a Weights CSV (sidebar) to proceed with ranking.")

                    if st.button("▶  Calculate Rankings", key="pcr_calc"):
                        if not weights:
                            st.error("Upload a Weights CSV first.")
                        else:
                            results_pcr, contributions_pcr, explanations_pcr, df_norm_pcr = calculate_score(
                                df_processed, "weighted_sum", weights, orientations)
                            
                            st.markdown("---")
                            section("2. Export")
                            csv_pcr = results_pcr.to_csv().encode("utf-8")
                            st.download_button("⬇  Download Ranked Results as CSV",
                                               data=csv_pcr, file_name="ranked_physicians.csv",
                                               mime="text/csv")

                            st.markdown("---")
                            section("3. Ranked Results")
                            st.caption("Expand each row for the radar, weighted contributions, and full calculation formula.")

                            top_3 = results_pcr.sort_values("Rank", ascending=True).head(3)
                            cols3 = st.columns(3)
                            badges3 = ["🥇", "🥈", "🥉"]
                            for i_, (name_, trow_) in enumerate(top_3.iterrows()):
                                if i_ < 3:
                                    with cols3[i_]:
                                        st.metric(label=f"{badges3[i_]} {name_}",
                                                  value=f"{trow_['Final Score']:.2f}",
                                                  delta=f"Rank {int(trow_['Rank'])}")

                            render_ranked_results_table(results_pcr, contributions_pcr,
                                                        df_norm_pcr, explanations_pcr, "weighted_sum")

                            st.markdown("---")
                            section("4. Visual Insights")
                            tab_p1, tab_p2, tab_p3, tab_p4 = st.tabs([
                                "Score Comparison", "Criteria Heatmap",
                                "Contribution Breakdown", "Conflict Insights"])

                            with tab_p1:
                                scores_vals = results_pcr["Final Score"].values
                                bar_colors  = px.colors.sample_colorscale(
                                    [[0, DARK], [0.5, MID], [1.0, ACCENT]],
                                    [(v - scores_vals.min()) / (scores_vals.max() - scores_vals.min() or 1)
                                     for v in scores_vals])
                                fig_bar_ = go.Figure(go.Bar(
                                    x=results_pcr.index.tolist(),
                                    y=results_pcr["Final Score"].tolist(),
                                    marker_color=bar_colors,
                                    text=[f"{v:.3f}" for v in results_pcr["Final Score"]],
                                    textposition="outside",
                                    hovertemplate="<b>%{x}</b><br>Score: %{y:.4f}<extra></extra>",
                                ))
                                fig_bar_.update_layout(
                                    title="Physician Priority Scores",
                                    xaxis_title="Physician", yaxis_title="Final Score",
                                    plot_bgcolor=LIGHT_MINT, paper_bgcolor="white",
                                    margin=dict(l=10, r=10, t=50, b=80), height=400,
                                    font=dict(family="Inter", color=DARK),
                                    xaxis=dict(tickangle=-35, type="category"))
                                fig_bar_.update_yaxes(gridcolor=TEAL_MIST)
                                st.plotly_chart(fig_bar_, use_container_width=True)

                            with tab_p2:
                                fig_heat_ = px.imshow(
                                    df_processed.T, text_auto=True, aspect="auto",
                                    title="Raw Criteria Heatmap (Physician × Criteria)",
                                    color_continuous_scale=KEN_COLORSCALE)
                                fig_heat_.update_layout(
                                    plot_bgcolor=LIGHT_MINT, paper_bgcolor="white",
                                    font=dict(family="Inter", color=DARK), height=400)
                                st.plotly_chart(fig_heat_, use_container_width=True)

                            with tab_p3:
                                contrib_t_ = contributions_pcr.T
                                crit_colors_ = px.colors.sample_colorscale(
                                    [[0, DARK], [0.4, MID], [0.7, ACCENT], [1.0, DEEP_GREEN]],
                                    [i / (len(contributions_pcr.index) - 1 or 1)
                                     for i in range(len(contributions_pcr.index))])
                                fig_stack_ = go.Figure()
                                for j_, crit_ in enumerate(contributions_pcr.index):
                                    fig_stack_.add_trace(go.Bar(
                                        name=crit_,
                                        x=contrib_t_.index.tolist(),
                                        y=contrib_t_[crit_].tolist(),
                                        marker_color=crit_colors_[j_],
                                        hovertemplate=f"<b>%{{x}}</b><br>{crit_}: %{{y:.4f}}<extra></extra>",
                                    ))
                                fig_stack_.update_layout(
                                    barmode="stack",
                                    title="Score Contribution by Criterion",
                                    xaxis_title="Physician", yaxis_title="Weighted Score Contribution",
                                    plot_bgcolor=LIGHT_MINT, paper_bgcolor="white",
                                    margin=dict(l=10, r=10, t=50, b=80), height=420,
                                    font=dict(family="Inter", color=DARK),
                                    xaxis=dict(tickangle=-35, type="category"),
                                    legend=dict(orientation="h", yanchor="bottom", y=1.02))
                                fig_stack_.update_yaxes(gridcolor=TEAL_MIST)
                                st.plotly_chart(fig_stack_, use_container_width=True)

                            with tab_p4:
                                st.markdown("#### Closest Competitors — Score Gaps")
                                ranked_df_ = results_pcr.sort_values("Final Score", ascending=False)
                                names_  = ranked_df_.index.tolist()
                                scores_ = ranked_df_["Final Score"].tolist()
                                gaps_   = []
                                for i_ in range(len(names_) - 1):
                                    gaps_.append({"Higher Rank": names_[i_],
                                                  "Lower Rank": names_[i_+1],
                                                  "Score Gap": scores_[i_] - scores_[i_+1]})
                                gaps_df_ = pd.DataFrame(gaps_).sort_values("Score Gap")

                                # Horizontal bar of gaps
                                gap_labels_ = [f"{int(r['Higher Rank'])} vs {int(r['Lower Rank'])}"
                                               for _, r in gaps_df_.iterrows()]
                                gap_vals_   = gaps_df_["Score Gap"].tolist()
                                gap_colors_ = [DEEP_GREEN if v > 0.05 else AMBER if v > 0.01 else DANGER
                                               for v in gap_vals_]
                                fig_gap_ = go.Figure(go.Bar(
                                    x=gap_vals_, y=gap_labels_, orientation="h",
                                    marker_color=gap_colors_,
                                    text=[f"{v:.4f}" for v in gap_vals_], textposition="outside",
                                    hovertemplate="<b>%{y}</b><br>Gap: %{x:.4f}<extra></extra>",
                                ))
                                fig_gap_.update_layout(
                                    title="Score Gap Between Adjacent Ranks (smaller = higher conflict)",
                                    xaxis_title="Score Gap", yaxis=dict(autorange="reversed"),
                                    plot_bgcolor=LIGHT_MINT, paper_bgcolor="white",
                                    margin=dict(l=10, r=80, t=50, b=30),
                                    height=max(280, 40 * len(gap_labels_)),
                                    font=dict(family="Inter", color=DARK),
                                )
                                fig_gap_.update_xaxes(gridcolor=TEAL_MIST, zeroline=False)
                                st.plotly_chart(fig_gap_, use_container_width=True)

                                st.caption("🔴 Red gaps indicate potential **High Conflict** zones where physicians are nearly tied.")
                                st.dataframe(gaps_df_.style.background_gradient(
                                    subset=["Score Gap"], cmap="RdYlGn", axis=0),
                                    use_container_width=True, hide_index=True)

                            
        except Exception as e:
            st.error(f"Error reading file: {e}")
    else:
        # Welcome state
        st.markdown(f"""
        <div class="kcard" style="background:{PALE_TEAL}; border:1px solid {TEAL_MIST};">
          <b style="color:{DARK};">Getting started</b><br>
          <p style="margin:8px 0 0 0; font-size:13px; color:{DARK}; line-height:1.6;">
            Upload a physician CSV (sidebar) with one physician per row and one criterion per column.
            Then supply a Weights CSV with the criterion importance scores — or export these directly
            from the <b>BWM Weight Calculator</b> tab.<br><br>
            The dashboard will rank all physicians, produce per-physician radar charts, weighted
            contribution bars, and identify high-conflict zones where scores are closely matched.
          </p>
        </div>""", unsafe_allow_html=True)

        c1_, c2_, c3_ = st.columns(3)
        for col_, title_, body_ in [
            (c1_, "Upload CSV", "One row per physician. First column = ID; remaining = criterion scores."),
            (c2_, "Upload Weights", "Two-column CSV: 'Criterion', 'Weight'. Use BWM output directly."),
            (c3_, "Rank & Explore", "Hit Calculate Rankings to see scores, radar profiles, and conflict zones."),
        ]:
            col_.markdown(f"""
            <div class="kcard kcard-accent">
              <div style="font-size:16px; font-weight:700; color:{DARK}; margin-bottom:8px;">{title_}</div>
              <div style="font-size:13px; color:#555; line-height:1.5;">{body_}</div>
            </div>""", unsafe_allow_html=True)
