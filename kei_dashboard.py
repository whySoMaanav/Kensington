"""
KEI — OR Prioritization Dashboard
Run with: streamlit run kei_dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, date
from dateutil.relativedelta import relativedelta
import os

# ─────────────────────────────────────────────────────────────
#  Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KEI Criteria Extraction",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Sidebar — light grey to match existing dashboard */
    [data-testid="stSidebar"] {
        background-color: #f8f9fa;
        border-right: 1px solid #e2e8f0;
    }

    /* Main background — white */
    .main .block-container {
        background-color: #ffffff;
        padding-top: 2rem;
    }

    /* Window pill — orange accent */
    .window-pill {
        display:inline-block;
        background:#fff7ed;
        color:#c2410c;
        font-size:0.85rem;
        padding:6px 14px;
        border-radius:8px;
        border: 1px solid #fed7aa;
        margin-top:4px;
    }

    /* Step numbers — orange accent */
    .step-num {
        display:inline-flex;
        align-items:center;
        justify-content:center;
        width:28px; height:28px;
        background:#f97316;
        color:white;
        border-radius:50%;
        font-size:0.8rem;
        font-weight:700;
        margin-right:8px;
        flex-shrink:0;
    }
    .step-label {
        font-size:0.95rem;
        font-weight:600;
        color:#111827;
        display:flex;
        align-items:center;
        margin-bottom:0.6rem;
    }

    /* Run button — orange */
    .stButton > button {
        background: #f97316 !important;
        color: white !important;
        border: none !important;
        border-radius: 6px !important;
        padding: 0.6rem 2rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        width: 100%;
    }
    .stButton > button:hover {
        background: #ea6c0a !important;
    }

    /* Download button — plain border style */
    [data-testid="stDownloadButton"] > button {
        background: white !important;
        color: #111827 !important;
        border: 1px solid #d1d5db !important;
        border-radius: 6px !important;
        font-weight: 500 !important;
        width: 100%;
    }
    [data-testid="stDownloadButton"] > button:hover {
        background: #f9fafb !important;
    }

    /* Metric tiles */
    [data-testid="stMetric"] {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 1rem;
    }

    .stAlert { border-radius: 6px; }
    hr { border-color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  Constants (mirrors notebook)
# ─────────────────────────────────────────────────────────────
WINDOW_DAYS  = 90
BLOCK_HOURS  = 7.5
GRACE_MIN    = 5

BOOKING_THRESHOLDS = {
    "Cataract": 16,
    "Retina":    8,
    "Cornea":    4,
    "Glaucoma":  8,
}
SUBSPECIALTIES = list(BOOKING_THRESHOLDS.keys())

# ─────────────────────────────────────────────────────────────
#  Criteria registry
#  To add a new criterion:
#    1. Add a row here: (internal_name, display_name, source)
#    2. If source="manual"     → add the column to the Manual Lookup Excel
#    3. If source="calculated" → add the calculation logic in run_pipeline()
#  Everything else (col_order, rename, validation) updates automatically.
# ─────────────────────────────────────────────────────────────
CRITERIA = [
    # (internal_name,       display_name,                source)
    ("leadership",          "DOVS Leadership",           "manual"),
    ("gft",                 "Geographically Full-Time",  "manual"),
    ("subspecialty",        "Subspecialty Coverage",     "manual"),
    ("committee",           "Committee Membership",      "manual"),
    ("scheduling_flex",     "Scheduling Flexibility",    "manual"),
    ("or_days_elsewhere",   "OR Time Elsewhere",         "manual"),
    ("avg_efficiency",      "Efficiency & Performance",  "calculated"),
]

# Derived lookups — do not edit these
CRITERIA_COL_ORDER  = ["physician_id"] + [c[0] for c in CRITERIA]
CRITERIA_RENAME_MAP = {c[0]: c[1] for c in CRITERIA}
CRITERIA_MANUAL_COLS = {"physician_id"} | {c[0] for c in CRITERIA if c[2] == "manual"}


# ─────────────────────────────────────────────────────────────
#  Shared utility
# ─────────────────────────────────────────────────────────────
def clean_id(series):
    """Normalise physician ID to clean string: strip whitespace and trailing .0"""
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


ENCOUNTER_COLS = [
    "encounter_id",
    "Surgeon Provider",
    "Operation Room",
    "Surgerydate",
    "Procedure",
    "Patient In Room Time",
    "Surgery StartTime",
    "Surgery StopTime",
]
BOOKING_COLS = [
    "Physician ID",
    "Appointment Date",
    "Room",
    "First appointment time",
    "Last appointment time",
] + SUBSPECIALTIES


# ─────────────────────────────────────────────────────────────
#  Pipeline functions (extracted from notebook)
# ─────────────────────────────────────────────────────────────

def extract_encounter(file, window_start, window_end):
    raw = pd.read_excel(file)

    # Validate first — before any column access
    required = {"Surgerydate", "Surgeon Provider", "Surgery StartTime", "Surgery StopTime", "Patient In Room Time"}
    if not required.issubset(set(raw.columns)):
        missing_req = sorted(required - set(raw.columns))
        raise ValueError(
            f"Wrong file for Encounter Data — missing columns: {missing_req}. "
            f"Found: {sorted(raw.columns.tolist())}"
        )

    raw["Surgerydate"] = pd.to_datetime(raw["Surgerydate"], errors="coerce")

    in_window = (
        (raw["Surgerydate"] >= pd.Timestamp(window_start)) &
        (raw["Surgerydate"] <= pd.Timestamp(window_end))
    )
    windowed = raw.loc[in_window].copy()

    if len(windowed) == 0:
        raise ValueError(
            f"No Encounter data found for {window_start} to {window_end}. "
            f"Please check the date range and try again."
        )

    missing = [c for c in ENCOUNTER_COLS if c not in windowed.columns]
    if missing:
        raise ValueError(f"Encounter file is missing required columns: {missing}")

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
    clean["duration_hr"] = (
        clean["Surgery StopTime"] - clean["Surgery StartTime"]
    ).dt.total_seconds() / 3600
    return clean


def extract_booking(file, window_start, window_end):
    try:
        raw = pd.read_excel(file, header=1)
        if "Physician ID" not in raw.columns:
            raw = pd.read_excel(file, header=0)
    except Exception:
        raw = pd.read_excel(file, header=0)

    # Validate first — before any column access
    required_booking = {"Physician ID", "Appointment Date", "First appointment time"}
    if not required_booking.issubset(set(raw.columns)):
        missing_req = sorted(required_booking - set(raw.columns))
        raise ValueError(
            f"Wrong file for Booking Data — missing columns: {missing_req}. "
            f"Found: {sorted(raw.columns.tolist())}"
        )

    raw["Appointment Date"] = pd.to_datetime(raw["Appointment Date"], errors="coerce")

    in_window = (
        (raw["Appointment Date"] >= pd.Timestamp(window_start)) &
        (raw["Appointment Date"] <= pd.Timestamp(window_end))
    )
    windowed = raw.loc[in_window].copy()

    if len(windowed) == 0:
        raise ValueError(
            f"No Booking data found for {window_start} to {window_end}. "
            f"Please check the date range and try again."
        )

    missing = [c for c in BOOKING_COLS if c not in windowed.columns]
    if missing:
        raise ValueError(f"Booking file is missing required columns: {missing}")
    clean = windowed[BOOKING_COLS].copy()
    clean["First appointment time"] = pd.to_datetime(clean["First appointment time"], errors="coerce")
    clean["Last appointment time"]  = pd.to_datetime(clean["Last appointment time"],  errors="coerce")
    clean[SUBSPECIALTIES] = clean[SUBSPECIALTIES].fillna(0).astype(int)
    clean = clean.dropna(subset=["Physician ID", "Appointment Date"])
    clean["Physician ID"]     = clean_id(clean["Physician ID"])
    clean["Appointment Date"] = pd.to_datetime(clean["Appointment Date"]).dt.date
    return clean


def calc_block_utilization(enc):
    daily = (
        enc.groupby(["Surgeon Provider", "Surgerydate"])["duration_hr"]
        .sum().reset_index().rename(columns={"duration_hr": "daily_used_hr"})
    )
    daily["daily_util"] = (daily["daily_used_hr"] / BLOCK_HOURS).clip(upper=1.0)
    result = (
        daily.groupby("Surgeon Provider")
        .agg(block_utilization=("daily_util", "mean"), n_or_days=("daily_util", "count"))
        .reset_index().rename(columns={"Surgeon Provider": "physician_id"})
    )
    result["block_utilization"] = result["block_utilization"].round(4)
    return result


def calc_ontime_start(enc, booking):
    enc = enc.copy()
    booking = booking.copy()
    enc["Surgeon Provider"] = enc["Surgeon Provider"].astype(str).str.strip()
    booking["Physician ID"] = booking["Physician ID"].astype(str).str.strip()

    # Normalise join keys to plain date (matches notebook logic)
    enc["_date"] = pd.to_datetime(enc["Surgerydate"]).dt.date
    booking["_date"] = pd.to_datetime(booking["Appointment Date"]).dt.date

    first_cases = (
        enc.sort_values("Patient In Room Time")
        .groupby(["Surgeon Provider", "_date"]).first().reset_index()
        [["Surgeon Provider", "_date", "Patient In Room Time"]]
    )
    merged = first_cases.merge(
        booking[["Physician ID", "_date", "First appointment time"]],
        left_on=["Surgeon Provider", "_date"],
        right_on=["Physician ID", "_date"],
        how="left",
    )
    merged["delay_min"] = (
        merged["Patient In Room Time"] - merged["First appointment time"]
    ).dt.total_seconds() / 60
    merged["on_time"] = merged["delay_min"] <= GRACE_MIN

    result = (
        merged.groupby("Surgeon Provider")
        .agg(
            on_time_pct=("on_time", "mean"),
            total_first_cases=("on_time", "count"),
            avg_delay_min=("delay_min", "mean"),
        )
        .reset_index().rename(columns={"Surgeon Provider": "physician_id"})
    )
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
            rows.append({"physician_id": physician_id, "booking_adherence": np.nan})
            continue

        threshold   = BOOKING_THRESHOLDS[primary_sub]
        active_days = grp[grp[SUBSPECIALTIES].sum(axis=1) > 0]
        adherence   = (active_days[primary_sub] >= threshold).mean() if len(active_days) else np.nan
        rows.append({
            "physician_id":      physician_id,
            "booking_adherence": round(adherence, 4) if not np.isnan(adherence) else np.nan,
        })
    return pd.DataFrame(rows)


def run_pipeline(enc_file, booking_file, manual_file, window_start, window_end, log):
    """Full pipeline. log = list to collect status messages."""

    log.append(("info", f"Calculation window: **{window_start}** → **{window_end}**"))

    # Extract
    log.append(("step", "Reading Encounter data…"))
    enc = extract_encounter(enc_file, window_start, window_end)
    log.append(("ok", f"Encounter: {len(enc):,} rows after date filter"))

    log.append(("step", "Reading Booking data…"))
    booking = extract_booking(booking_file, window_start, window_end)
    log.append(("ok", f"Booking: {len(booking):,} rows after date filter"))

    # Validation
    enc_ids  = set(enc["Surgeon Provider"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True).unique())
    book_ids = set(booking["Physician ID"].astype(str).unique())
    only_enc = enc_ids - book_ids
    if only_enc:
        log.append(("warn", f"{len(only_enc)} physician(s) in Encounter but not in Booking: {sorted(only_enc)}"))

    dup_enc = enc.duplicated(subset=["encounter_id"]).sum()
    if dup_enc:
        log.append(("warn", f"{dup_enc} duplicate encounter_id rows found"))
    else:
        log.append(("ok", "No duplicate encounter rows ✓"))

    # Calculate
    log.append(("step", "Calculating Block Utilization…"))
    block = calc_block_utilization(enc)

    log.append(("step", "Calculating On-Time Start…"))
    ontime = calc_ontime_start(enc, booking)
    match_pct = ontime["on_time_pct"].notna().sum() / len(ontime) * 100 if len(ontime) else 0
    log.append(("ok", f"On-Time Start calculated for {len(ontime)} physicians"))

    log.append(("step", "Calculating Booking Adherence…"))
    adherence = calc_booking_adherence(booking)

    # Normalise physician_id to clean string in all three frames
    block["physician_id"]     = clean_id(block["physician_id"])
    ontime["physician_id"]    = clean_id(ontime["physician_id"])
    adherence["physician_id"] = clean_id(adherence["physician_id"])

    # Merge calculated
    calculated = (
        block
        .merge(ontime[["physician_id", "on_time_pct", "ontime_meets_85pct"]], on="physician_id", how="outer")
        .merge(adherence, on="physician_id", how="outer")
    )
    calculated["physician_id"] = clean_id(calculated["physician_id"])
    calculated["avg_efficiency"] = calculated[
        ["block_utilization", "on_time_pct", "booking_adherence"]
    ].mean(axis=1, skipna=False).round(4)

    def _efficiency_note(row):
        if pd.notna(row["avg_efficiency"]):
            return ""
        missing = []
        if pd.isna(row["block_utilization"]): missing.append("no OR cases")
        if pd.isna(row["on_time_pct"]):       missing.append("no booking match")
        if pd.isna(row["booking_adherence"]): missing.append("no booking data")
        return "Missing: " + ", ".join(missing) if missing else "insufficient data"

    calculated["_efficiency_note"] = calculated.apply(_efficiency_note, axis=1)

    # Manual lookup
    log.append(("step", "Reading Manual Lookup table…"))
    manual = pd.read_excel(manual_file, dtype={"physician_id": str})
    manual["physician_id"] = clean_id(manual["physician_id"])

    # Validate columns — catch typos before they silently drop data
    expected = CRITERIA_MANUAL_COLS
    actual = set(manual.columns.str.strip().str.lower())
    missing_cols = expected - actual
    if missing_cols:
        raise ValueError(
            f"Manual Lookup file is missing or has misspelled columns: {sorted(missing_cols)}\n"
            f"Found columns: {sorted(manual.columns.tolist())}"
        )

    log.append(("ok", f"Manual lookup: {len(manual)} physicians"))

    # Final merge
    final = calculated.merge(manual, on="physician_id", how="left")

    col_order = [c for c in CRITERIA_COL_ORDER if c in final.columns]
    # Keep _efficiency_note alongside criteria cols for display (dropped before CSV export)
    extra_cols = [c for c in ["_efficiency_note"] if c in final.columns]
    final = final[col_order + extra_cols]
    final = final.rename(columns=CRITERIA_RENAME_MAP)

    log.append(("ok", f"Output: {len(final)} physicians × {len(final.columns)} columns ✓"))
    return final


# ─────────────────────────────────────────────────────────────
#  Top controls — date + file uploads
# ─────────────────────────────────────────────────────────────
st.markdown("# Criteria Extraction")
st.markdown("Select the quarter start date, upload the three required files, then click **Run Pipeline**.")
st.markdown("---")

col_date, col_enc, col_booking, col_manual = st.columns(4)

with col_date:
    st.markdown('<div class="step-label"><span class="step-num">1</span> Quarter Start Date</div>', unsafe_allow_html=True)
    today = date.today()
    quarter_month = ((today.month - 1) // 3) * 3 + 1
    default_start = date(today.year, quarter_month, 1) - timedelta(days=90)
    run_date = st.date_input("Quarter start date", value=default_start, label_visibility="collapsed")
    window_start = run_date
    window_end   = run_date + relativedelta(months=3) - timedelta(days=1)
    st.markdown(
        f'<div class="window-pill">📅 {window_start.strftime("%b %d, %Y")} → {window_end.strftime("%b %d, %Y")}</div>',
        unsafe_allow_html=True,
    )

with col_enc:
    st.markdown('<div class="step-label"><span class="step-num">2</span> Encounter Data</div>', unsafe_allow_html=True)
    enc_file = st.file_uploader("Encounter Excel", type=["xlsx"], label_visibility="collapsed", key="enc")

with col_booking:
    st.markdown('<div class="step-label"><span class="step-num">3</span> Booking Data</div>', unsafe_allow_html=True)
    booking_file = st.file_uploader("Booking Excel", type=["xlsx"], label_visibility="collapsed", key="booking")

with col_manual:
    st.markdown('<div class="step-label"><span class="step-num">4</span> Manual Lookup</div>', unsafe_allow_html=True)
    manual_file = st.file_uploader("Manual Lookup Excel", type=["xlsx"], label_visibility="collapsed", key="manual")

st.markdown("---")

all_uploaded = enc_file and booking_file and manual_file
run_btn = st.button("▶  Run Pipeline", disabled=not all_uploaded, use_container_width=False)
if not all_uploaded:
    st.caption("Upload all 3 files to enable.")

st.markdown("---")

# ─────────────────────────────────────────────────────────────
#  Main area
# ─────────────────────────────────────────────────────────────
# ── Run ───────────────────────────────────────────────────────
# ── Run ───────────────────────────────────────────────────────
if run_btn or "pipeline_result" in st.session_state:

    if run_btn:
        log = []
        with st.spinner("Running pipeline…"):
            try:
                result = run_pipeline(
                    enc_file, booking_file, manual_file,
                    window_start, window_end, log,
                )
                st.session_state["pipeline_result"] = result
                st.session_state["pipeline_log"]    = log
                st.session_state["pipeline_ok"]     = True
            except Exception as e:
                st.session_state["pipeline_ok"]  = False
                st.session_state["pipeline_err"] = str(e)
                st.session_state["pipeline_log"] = log

    # ── Status — simple success/error only ───────────────────
    warnings = [msg for kind, msg in st.session_state.get("pipeline_log", []) if kind == "warn"]

    if not st.session_state.get("pipeline_ok"):
        err_msg = st.session_state.get("pipeline_err", "Unknown error")
        st.error(f"⚠ Upload error: {err_msg}")
        st.stop()
    else:
        st.success("✓ Criteria extraction complete.")
        for w in warnings:
            st.warning(w)

    result    = st.session_state["pipeline_result"]
    export_df = result.drop(columns=["_efficiency_note"], errors="ignore")

    st.markdown("---")

    # ── Summary metrics ───────────────────────────────────────
    st.markdown("### 1. Summary")
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Physicians", len(result))
    with m2:
        eff_col = "Efficiency & Performance"
        if eff_col in result.columns:
            avg_eff = result[eff_col].mean()
            st.metric("Avg Efficiency", f"{avg_eff:.1%}" if not np.isnan(avg_eff) else "N/A")
    with m3:
        if eff_col in result.columns:
            missing_eff = result[eff_col].isna().sum()
            st.metric("Missing Efficiency", missing_eff, delta=None)
    with m4:
        window_label = f"{window_start.strftime('%b %d')} – {window_end.strftime('%b %d, %Y')}"
        st.metric("Window", window_label)

    st.markdown("---")

    # ── Preview table ─────────────────────────────────────────
    st.markdown("### 2. Output Preview")

    # Colour efficiency column
    def style_efficiency(val):
        if pd.isna(val):
            return "background-color:#fee2e2; color:#991b1b"
        elif val >= 0.7:
            return "background-color:#d1fae5; color:#065f46"
        elif val >= 0.5:
            return "background-color:#fef3c7; color:#92400e"
        else:
            return "background-color:#fee2e2; color:#991b1b"

    styled = export_df.style.applymap(
        style_efficiency,
        subset=["Efficiency & Performance"] if "Efficiency & Performance" in export_df.columns else [],
    ).format(
        {"Efficiency & Performance": lambda x: f"{x:.1%}" if not pd.isna(x) else "—"},
        na_rep="—",
    )
    st.dataframe(styled, use_container_width=True, height=380)

    # ── Efficiency missing notes ─────────────────────────────
    if "_efficiency_note" in result.columns:
        missing_eff = result[result["_efficiency_note"] != ""].copy()
        if not missing_eff.empty:
            id_col = "physician_id" if "physician_id" in missing_eff.columns else missing_eff.columns[0]
            display_cols = [id_col, "_efficiency_note"]
            missing_eff = missing_eff[display_cols].rename(columns={
                id_col: "Physician ID",
                "_efficiency_note": "Reason"
            })
            with st.expander(f"⚠ {len(missing_eff)} physician(s) have missing Efficiency & Performance data"):
                st.dataframe(missing_eff, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Download ──────────────────────────────────────────────
    st.markdown("### 3. Export")
    csv_bytes = export_df.to_csv(index=False).encode("utf-8")
    filename  = f"or_prioritization_{window_start.strftime('%Y%m%d')}_{window_end.strftime('%Y%m%d')}.csv"

    col_dl, col_info = st.columns([1, 2])
    with col_dl:
        st.download_button(
            label="⬇  Download CSV",
            data=csv_bytes,
            file_name=filename,
            mime="text/csv",
            use_container_width=True,
        )
    with col_info:
        st.markdown(f"""
        <div style="padding:0.6rem 1rem; background:#f0f9ff; border-radius:8px; border:1px solid #bae6fd; font-size:0.88rem; color:#0369a1;">
            <strong>File:</strong> <code>{filename}</code><br>
            <strong>Rows:</strong> {len(result)} physicians &nbsp;|&nbsp;
            <strong>Columns:</strong> {len(result.columns)}
        </div>
        """, unsafe_allow_html=True)
