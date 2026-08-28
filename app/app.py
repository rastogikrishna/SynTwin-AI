import streamlit as st
import pandas as pd
import numpy as np
import pathlib
import sys
import os
import plotly.express as px
import plotly.graph_objects as go

# Add project root to path to load src modules
project_root = pathlib.Path(__file__).parent.parent
sys.path.append(str(project_root))

# Load LLM_API_KEY / LLM_PROVIDER / LLM_MODEL from a local .env file (see
# .env.example) so the AI Assistant page can actually pick up credentials.
# Without this, os.environ.get("LLM_API_KEY") below would always be None
# unless the variable was exported in the shell by hand.
try:
    from dotenv import load_dotenv
    load_dotenv(project_root / ".env")
except ImportError:
    pass

import styles
import components as ui
from styles import COLORS, CHART_SEQUENCE, plotly_layout

from src.data.loader import load_data
from src.data.profiler import profile_dataset
from src.analysis.kpi_engine import discover_kpis
from src.analysis.pattern_engine import analyze_patterns
from src.analysis.anomaly_engine import detect_anomalies
from src.models.target_detector import detect_targets
from src.models.preprocessor import DataPreprocessor
from src.models.trainer import train_best_model
from src.models.evaluator import evaluate_model

# ============================================================
# PAGE CONFIG + GLOBAL STYLE
# ============================================================
st.set_page_config(
    page_title="SynTwin AI — Decision Intelligence Platform",
    page_icon="◆",
    layout="wide",
    initial_sidebar_state="collapsed",
)
styles.inject_global_css()

# Initialize session state for navigation selection if not present
if "active_nav" not in st.session_state:
    st.session_state["active_nav"] = "Overview"

# PAGE TAGLINES
PAGE_TAGLINES = {
    "Overview": "Understand your data at a glance — KPIs, trends, anomalies and quality in one view.",
    "Data Profile": "Column-level schema, types, and variables breakdown.",
    "Data Quality": "Automated diagnostics and quality flags scoring overall dataset health.",
    "Diagnosis": "Automatically discovered KPIs, patterns, relationships and anomalies.",
    "Prediction": "Build and evaluate a predictive model for a business outcome you choose.",
    "Explainability": "See which factors drive the model's predictions, globally and per record.",
    "Forecast": "Project business metrics forward using historical time-series patterns.",
    "Digital Twin": "Test what-if scenarios against the trained model before acting.",
    "Decision": "Search for the action that optimizes your chosen business objective.",
    "AI Assistant": "Ask questions about your data, models, forecasts and recommendations.",
}


def format_kpi_value(value) -> str:
    """Safely format a KPI value that may be numeric or textual."""
    if isinstance(value, float):
        return f"{value:,.2f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def clean_name(col: str) -> str:
    return str(col).replace("_", " ").title()


@st.cache_data(show_spinner="Loading dataset...")
def load_data_cached(file_bytes: bytes, file_name: str) -> pd.DataFrame:
    import io
    file_io = io.BytesIO(file_bytes)
    file_io.name = file_name
    return load_data(file_io, file_name=file_name)


@st.cache_data(show_spinner="Profiling dataset...")
def profile_dataset_cached(df: pd.DataFrame) -> dict:
    return profile_dataset(df)


@st.cache_data(show_spinner=False)
def run_diagnosis_engines(df: pd.DataFrame, profile: dict):
    """KPI / pattern / anomaly discovery, cached per dataset so switching
    pages or widgets doesn't force recomputation (see performance requirements)."""
    kpis = discover_kpis(df, profile)
    patterns = analyze_patterns(df, profile)
    anomalies = detect_anomalies(df, profile)
    return kpis, patterns, anomalies


def get_representative_sample(df: pd.DataFrame, target_col: str, target_type: str, sample_size: int = 10000) -> pd.DataFrame:
    if len(df) <= sample_size:
        return df
    
    # Stratified split for classification target
    if target_type in ["binary_classification", "multiclass_classification"] and target_col in df.columns:
        clean_target_df = df[df[target_col].notna()]
        if not clean_target_df.empty:
            class_counts = clean_target_df[target_col].value_counts()
            if len(class_counts) >= 2 and class_counts.min() >= 2:
                try:
                    from sklearn.model_selection import train_test_split
                    _, sample_df = train_test_split(
                        clean_target_df,
                        test_size=sample_size,
                        stratify=clean_target_df[target_col],
                        random_state=42
                    )
                    return sample_df
                except Exception:
                    pass
                    
    # Random sampling for regression or fallback
    return df.sample(n=sample_size, random_state=42)


def build_auto_insights(profile, kpis, patterns, anomalies):
    """Compose the 'AI Insights' panel strictly from computed values —
    never fabricated. Returns list of (title, html_desc, kind)."""
    insights = []

    if patterns.get("temporal_patterns"):
        t = patterns["temporal_patterns"][0]
        insights.append((
            "Trend detected",
            f"<b>{clean_name(t['value_column'])}</b> shows an <b>{t['trend_type']}</b> trend across "
            f"{t['periods_count']} monthly periods of <b>{clean_name(t['date_column'])}</b> "
            f"(trend strength {t['correlation']:.2f}).",
            "pos" if t["trend_type"] == "increasing" else "neg" if t["trend_type"] == "decreasing" else "info",
        ))

    if patterns.get("correlations"):
        c = patterns["correlations"][0]
        direction = "positive" if c["coefficient"] > 0 else "inverse"
        insights.append((
            "Strongest relationship",
            f"<b>{clean_name(c['col1'])}</b> and <b>{clean_name(c['col2'])}</b> show a <b>{direction}</b> "
            f"relationship (r = {c['coefficient']:.2f}).",
            "info",
        ))

    if patterns.get("categorical_patterns"):
        cp = patterns["categorical_patterns"][0]
        insights.append((
            "Dominant category",
            f"<b>{clean_name(cp['column'])}</b> is dominated by value <b>'{cp['dominant_value']}'</b>, "
            f"representing {cp['percentage']:.1f}% of all records.",
            "warn",
        ))

    if anomalies.get("total_anomalies", 0) > 0:
        insights.append((
            "Statistical outliers present",
            f"<b>{anomalies['columns_with_anomalies']}</b> column(s) contain outliers — "
            f"{anomalies['total_anomalies']:,} flagged records ({anomalies['anomaly_percentage']:.2f}% of the dataset).",
            "warn",
        ))

    warnings = profile.get("warnings", [])
    high_sev = [w for w in warnings if w.get("severity") == "high"]
    if high_sev:
        insights.append((
            "Data quality requires attention",
            f"{len(high_sev)} variable(s) have high-severity quality issues, including "
            f"<b>{clean_name(high_sev[0]['column'])}</b> ({high_sev[0]['message']}).",
            "neg",
        ))
    elif warnings:
        insights.append((
            "Minor data quality notes",
            f"{len(warnings)} lower-severity quality note(s) were recorded across the dataset.",
            "info",
        ))
    else:
        insights.append((
            "Clean dataset",
            "No data-quality warnings were recorded across any column.",
            "pos",
        ))

    if kpis:
        top_kpi = kpis[0]
        insights.append((
            f"Key metric — {top_kpi['name']}",
            f"{top_kpi['interpretation']} Current value: <b>{format_kpi_value(top_kpi['value'])}</b>.",
            "info",
        ))

    return insights


def problem_type_for(profile, target_col: str) -> str:
    """Shared helper reproducing the exact target-type inference used
    throughout the original app (kept identical for backward compatibility)."""
    target_col_info = profile["column_profiles"].get(target_col, {})
    target_uniq = target_col_info.get("unique_count", 2)
    target_group = target_col_info.get("type_group", "numeric")
    is_float = "float" in target_col_info.get("dtype", "")
    if target_group in ["categorical", "boolean"] or (target_group == "numeric" and target_uniq <= 15 and not is_float):
        return "binary_classification" if target_uniq == 2 else "multiclass_classification"
    return "regression"


def is_conversational(query: str) -> bool:
    q = query.lower().strip().rstrip("?.!")
    # Basic greetings
    greetings = {"hello", "hi", "hey", "good morning", "good afternoon", "good evening", "greetings", "yo", "sup"}
    if q in greetings:
        return True
    
    # Conversational words
    conv_words = {"thanks", "thank you", "bye", "goodbye", "who are you", "your name", "whats your name", "what's your name"}
    if q in conv_words:
        return True
        
    # Checks for name introductions or name queries
    if "name" in q:
        return True
        
    # Other conversational markers
    identity_phrases = ["who are you", "what can you do", "what is your purpose", "how are you", "how's it going", "how are things", "nice to meet you", "thank you"]
    if any(p in q for p in identity_phrases):
        return True
        
    return False


def local_conversational_response(query: str) -> str:
    q = query.lower().strip().rstrip("?.!")
    if "krishna" in q:
        return "Nice to meet you, Krishna! I'm SynTwin AI. What would you like to explore?"
    if "what is my name" in q or "what's my name" in q:
        # Try to extract name from chat history
        history = st.session_state.get("chat_history", [])
        for msg in reversed(history):
            if msg["role"] == "user":
                content = msg["content"].lower()
                if "my name is " in content:
                    name_idx = content.find("my name is ") + 11
                    name = msg["content"][name_idx:].strip().rstrip("?.!")
                    return f"Your name is {name}."
        return "I don't know your name yet. What should I call you?"
    if any(x in q for x in ["who are you", "your name", "what is your name", "what's your name"]):
        return "I'm SynTwin AI, a decision-intelligence assistant that helps analyze business data, explain model results, forecast trends, and evaluate decisions."
    if any(x in q for x in ["hello", "hi", "hey", "good morning", "good afternoon", "good evening"]):
        return "Hello! I'm SynTwin AI. I can help you understand your business data, identify important patterns, explain predictions, review forecasts, and explore what-if decisions."
    return "Hello! How can I help you explore your data today?"


def local_assistant_answer(query: str, df: pd.DataFrame, profile: dict) -> str:
    """Fast, dependency-free grounded fallback.
    The app remains usable without an external LLM key; Gemini is used when configured.
    """
    q = query.lower().strip()
    rows = profile.get("overview", {}).get("num_rows", len(df))
    cols = profile.get("overview", {}).get("num_columns", len(df.columns))
    health = profile.get("overall_status", "Unknown")
    warnings = profile.get("warnings", [])
    numeric = profile.get("data_types", {}).get("numerical", [])
    categorical = profile.get("data_types", {}).get("categorical", [])

    if "syntwin" in q and ("mean" in q or "what is" in q or "name" in q):
        return "**SynTwin** means **Synthetic + Twin**: a system that creates a digital representation of a real business/data environment so you can analyze, predict and test decisions before acting."
    if "digital twin" in q and ("what" in q or "mean" in q or "explain" in q):
        return "A **Digital Twin** is a virtual model of a real system. In SynTwin, it uses your trained ML model and selected variables to run **what-if scenarios** and estimate how the outcome may change."
    if any(x in q for x in ["summarize", "summary", "current situation", "overview"]):
        return (f"**Dataset summary:** {rows:,} rows and {cols} variables. "
                f"Data health is **{health}** with {len(warnings)} quality warning(s). "
                f"The dataset contains {len(numeric)} numeric and {len(categorical)} categorical variables.")
    if any(x in q for x in ["risk", "problem", "issue", "concern"]):
        if warnings:
            top = warnings[0]
            return f"The main data-quality risk is **{clean_name(top.get('column','unknown'))}**: {top.get('message','quality warning')}. There are {len(warnings)} warning(s) in total."
        return "No profile-level data-quality warnings were detected. Check the Diagnosis and Anomaly sections for statistical risks."
    if "forecast" in q:
        if any(k.startswith("forecast_res_") for k in st.session_state.keys()):
            return "A forecast has been generated in this session. Open **Forecast** to view the projected values and uncertainty details."
        return "No forecast has been generated yet. Open **Forecast**; SynTwin will automatically look for a usable date/time column and numeric business signal."
    if any(x in q for x in ["prediction", "model", "shap", "driver", "factor", "important"]):
        trained = [k for k in st.session_state.keys() if k.startswith("model_")]
        if trained:
            target = trained[0].replace("model_", "")
            meta = st.session_state.get(f"meta_{target}", {})
            return f"The current trained model targets **{clean_name(target)}** and uses **{meta.get('best_name','the selected best model')}**. Open **Explainability** for the strongest model drivers."
        return "No predictive model is trained yet. Open **Prediction** and use the automatically selected target; advanced target settings are optional."
    if any(x in q for x in ["what-if", "scenario", "simulation", "change"]):
        if "last_sim_res" in st.session_state:
            return "A what-if simulation is available. Open **Digital Twin** to review the baseline, scenario and estimated impact."
        return "Open **Digital Twin** to test a small number of automatically selected important variables. You do not need to configure every column."
    if any(x in q for x in ["decision", "recommend", "optimization", "action"]):
        if "last_ga_res" in st.session_state or "last_rl_res" in st.session_state:
            return "A model-based decision recommendation is available in **Decision**. Treat it as an estimate within the observed data range, not a guaranteed business outcome."
        return "No decision recommendation has been generated yet. Open **Decision** after training a model; SynTwin will use sensible defaults rather than requiring many manual settings."
    return (f"Local Analysis: The dataset contains {rows:,} rows and {cols} variables. "
            "You can navigate to Overview, Diagnosis, Prediction, Forecast, Digital Twin, or Decision tabs "
            "to perform analytical actions, or ask specific questions about the data, models, or optimization.")

# ============================================================
# REFERENCE-STYLE HEADER + NAVIGATION
# ============================================================
status_placeholder = st.empty()

header_left, header_right = st.columns([1.7, 1.0], vertical_alignment="center")
with header_left:
    st.markdown(
        """
        <div class="ref-brand">
            <div class="ref-brand-mark">◆</div>
            <div>
                <div class="ref-brand-name">SynTwin AI</div>
                <div class="ref-brand-sub">DOMAIN-ADAPTIVE DECISION INTELLIGENCE</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with header_right:
    uploaded_file = st.file_uploader(
        "Upload dataset", type=["csv", "xlsx"], key="dataset_uploader",
        help="Upload a CSV or Excel business dataset. The platform adapts its analysis to the available columns.",
        label_visibility="collapsed",
    )

st.markdown('<div class="ref-header-rule"></div>', unsafe_allow_html=True)

# Main navigation uses the same backend page/state names as before.
nav_items = [
    ("Overview", "▦"), ("Diagnosis", "⌁"), ("Prediction", "◉"),
    ("Explainability", "◇"), ("Forecast", "⌁"), ("Digital Twin", "◌"),
    ("Decision", "⌘"), ("AI Assistant", "✦"),
]
nav_cols = st.columns(len(nav_items), gap="small")
for col, (label, icon) in zip(nav_cols, nav_items):
    with col:
        if st.button(
            f"{icon}  {label}",
            key=f"nav_btn_{label.lower().replace(' ', '_')}",
            width="stretch",
            type="primary" if st.session_state["active_nav"] == label else "secondary",
        ):
            st.session_state["active_nav"] = label
            st.rerun()

navigation_selection = st.session_state["active_nav"]


# -----------------------------------------------------------------
# DATASET LIFECYCLE & STATE PERSISTENCE MANAGER
# -----------------------------------------------------------------
# IMPORTANT: Do not clear the persisted dataset merely because the
# Streamlit uploader temporarily returns None during a navigation rerun.
# The uploader is a UI control; the persisted dataset in session_state is
# the source of truth for page navigation. Clearing it here caused every
# sidebar navigation click to send the app back to the "Load Data" state.
#
# If the user actually replaces a dataset, the file-name check below handles
# invalidation and reloads the new dataset. If the user wants to clear the
# current dataset, use the explicit reset control provided in the sidebar.
# Handle active dataset state Ingestion
if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    current_file_name = uploaded_file.name
    
    # Trigger full state invalidation on dataset file swap
    if st.session_state.get("dataset_name") != current_file_name:
        keys_to_clear = [
            k for k in st.session_state.keys()
            if k.startswith("model_") or k.startswith("preprocessor_") or k.startswith("meta_")
            or k.startswith("shap_vals_") or k.startswith("shap_base_")
            or k.startswith("global_importance_") or k.startswith("local_expl_")
            or k.startswith("forecast_res_") or k == "last_sim_res"
            or k == "saved_scenarios" or k == "last_ga_res" or k == "last_rl_res"
            or k == "chat_history" or k == "uploaded_doc_names" or k == "forecast_eligibility"
        ]
        for k in keys_to_clear:
            st.session_state.pop(k, None)
            
        if "vector_store" in st.session_state:
            st.session_state["vector_store"].clear()
            
        df = load_data_cached(file_bytes, file_name=current_file_name)
        profile = profile_dataset_cached(df)
        
        st.session_state["dataset"] = df
        st.session_state["profile"] = profile
        st.session_state["dataset_name"] = current_file_name
        st.session_state["active_nav"] = "Overview"
        st.rerun()
    else:
        # Load from session state if cached, otherwise reload
        if "dataset" not in st.session_state:
            df = load_data_cached(file_bytes, file_name=current_file_name)
            st.session_state["dataset"] = df
        else:
            df = st.session_state["dataset"]
            
        if "profile" not in st.session_state:
            profile = profile_dataset_cached(df)
            st.session_state["profile"] = profile
        else:
            profile = st.session_state["profile"]
else:
    # Check if a dataset was already loaded and persist it
    if "dataset" in st.session_state:
        df = st.session_state["dataset"]
        profile = st.session_state["profile"]
        current_file_name = st.session_state["dataset_name"]
    else:
        df = None
        profile = None
        current_file_name = None

# Compact dataset status strip in the reference header position.
if df is not None:
    _status = profile["overall_status"]
    _status_color = COLORS["good"] if _status == "Good" else COLORS["warn"] if _status == "Needs Attention" else COLORS["bad"]
    status_placeholder.markdown(
        f'<div class="ref-status"><span>Dataset</span><b>{current_file_name}</b>'
        f'<span class="ref-status-dot" style="background:{_status_color};"></span>'
        f'<b>{_status}</b><span class="ref-status-meta">{profile["overview"]["num_rows"]:,} rows · {profile["overview"]["num_columns"]} variables</span></div>',
        unsafe_allow_html=True,
    )
else:
    status_placeholder.markdown(
        '<div class="ref-status"><span>Dataset</span><b>Not loaded</b><span class="ref-status-dot"></span>'
        '<span class="ref-status-meta">Upload CSV or Excel to begin</span></div>',
        unsafe_allow_html=True,
    )

# ============================================================
# EMPTY STATE — no dataset uploaded yet
# ============================================================
if df is None:
    with status_placeholder:
        st.markdown('<hr style="margin: 10px 0;">', unsafe_allow_html=True)
        st.markdown('<div class="stw-nav-section-label" style="margin-top: 5px;">DATASET</div>', unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="font-size: 0.78rem; padding: 2px 4px; line-height: 1.4;">
                <div style="display: flex; align-items: center; margin-bottom: 4px;">
                    <span class="stw-dot stw-dot-bad" style="margin-right: 6px; width: 8px; height: 8px;"></span>
                    <span style="color: {COLORS['bad']}; font-weight: 700;">Offline</span>
                </div>
                <div style="color: {COLORS['text_muted']}; font-size: 0.74rem;">
                    No dataset loaded
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    ui.top_bar(
        "Business Intelligence Overview",
        "Understand your data. Predict what comes next. Simulate decisions. Act with confidence.",
        [("Dataset", "Not loaded"), ("Status", "Awaiting Ingestion")],
    )

    ui.empty_state(
        "Connect your business data",
        "Upload a CSV or Excel dataset using the Upload button above.",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="display: flex; justify-content: space-between; align-items: center; padding: 10px 0; margin-bottom: 15px; font-weight: 700; font-size: 0.8rem; letter-spacing: 0.1em; color: #17B996;">
            <div style="flex: 1; text-align: center;">UNDERSTAND</div>
            <div style="color: #5E6D6B; font-size: 1.2rem;">→</div>
            <div style="flex: 1; text-align: center;">DIAGNOSE</div>
            <div style="color: #5E6D6B; font-size: 1.2rem;">→</div>
            <div style="flex: 1; text-align: center;">PREDICT</div>
            <div style="color: #5E6D6B; font-size: 1.2rem;">→</div>
            <div style="flex: 1; text-align: center;">DECIDE</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    stages = [
        ("Understand", "Automatically profile your data"),
        ("Diagnose", "Find patterns, KPIs and anomalies"),
        ("Predict", "Forecast future outcomes"),
        ("Decide", "Simulate scenarios and optimize actions"),
    ]
    cols = st.columns(4)
    for i, (name, desc) in enumerate(stages):
        with cols[i]:
            with ui.card(name, desc):
                pass
            st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

else:
    try:
        with status_placeholder:
            st.markdown('<hr style="margin: 10px 0;">', unsafe_allow_html=True)
            st.markdown('<div class="stw-nav-section-label" style="margin-top: 5px;">DATASET</div>', unsafe_allow_html=True)
            status = profile["overall_status"]
            dot = "stw-dot-good" if status == "Good" else ("stw-dot-warn" if status == "Needs Attention" else "stw-dot-bad")
            st.markdown(
                f"""
                <div style="font-size: 0.78rem; padding: 2px 4px; line-height: 1.4;">
                    <div style="display: flex; align-items: center; margin-bottom: 4px;">
                        <span class="stw-dot {dot}" style="margin-right: 6px; width: 8px; height: 8px;"></span>
                        <span style="color: {COLORS['good'] if status == 'Good' else (COLORS['warn'] if status == 'Needs Attention' else COLORS['bad'])}; font-weight: 700;">Connected ({status})</span>
                    </div>
                    <div style="color: {COLORS['text']}; font-weight: 600; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; margin-bottom: 2px;" title="{current_file_name}">
                        {current_file_name[:22]}...
                    </div>
                    <div style="color: {COLORS['text_muted']}; font-size: 0.74rem;">
                        {profile['overview']['num_rows']:,} records<br>
                        {profile['overview']['num_columns']} variables
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

        meta_items = [
            ("Dataset", current_file_name),
            ("Rows", f"{profile['overview']['num_rows']:,}"),
            ("Columns", str(profile["overview"]["num_columns"])),
            ("Data Health", profile["overall_status"]),
        ]
        ui.top_bar(navigation_selection, PAGE_TAGLINES.get(navigation_selection, ""), meta_items)

        # ================================================
        # OVERVIEW — Power BI style landing dashboard
        # ================================================
        if navigation_selection == "Overview":
            # Single cached diagnosis pass powers the whole landing dashboard.
            kpis, patterns, anomalies = run_diagnosis_engines(df, profile)
            status = profile["overall_status"]
            status_class = "good" if status == "Good" else "warn" if status == "Needs Attention" else "bad"
            total_rows = profile["overview"]["num_rows"]
            total_cols = profile["overview"]["num_columns"]
            anomaly_pct = anomalies["anomaly_percentage"]
            health_segments = 6 if status == "Good" else 4 if status == "Needs Attention" else 2

            st.markdown(
                f"""
                <div class="ref-hero">
                    <div>
                        <div class="ref-eyebrow">BUSINESS INTELLIGENCE</div>
                        <div class="ref-title">SynTwin AI</div>
                        <div class="ref-subtitle">Decision intelligence built around your uploaded business data</div>
                    </div>
                    <div class="ref-health">
                        <div class="ref-health-label">Dataset Health</div>
                        <div class="ref-health-row"><span class="ref-dot {status_class}"></span><b>{status}</b></div>
                        <div class="ref-segments">
                            {''.join('<span class="filled"></span>' if i < health_segments else '<span></span>' for i in range(6))}
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            ui.kpi_row([
                {"label": "RECORDS", "value": f"{total_rows:,}", "sub": "Rows loaded"},
                {"label": "VARIABLES", "value": str(total_cols),
                 "sub": f"{len(profile['data_types'].get('numerical', []))} numeric · {len(profile['data_types'].get('categorical', []))} categorical"},
                {"label": "DATA HEALTH", "value": status, "sub": f"{len(profile.get('warnings', []))} quality alerts"},
                {"label": "ANOMALY RATE", "value": f"{anomaly_pct:.2f}%", "sub": f"{anomalies['total_anomalies']:,} flagged records"},
            ], columns=4)

            st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
            main_left, main_right = st.columns([2.05, 1.0])

            with main_left:
                with ui.card("Business Trend", "Automatically selected temporal signal from your dataset."):
                    if patterns.get("temporal_patterns"):
                        t = patterns["temporal_patterns"][0]
                        try:
                            tdf = pd.DataFrame({
                                "date": pd.to_datetime(df[t["date_column"]], errors="coerce", format="mixed"),
                                "value": pd.to_numeric(df[t["value_column"]], errors="coerce"),
                            }).dropna()
                            tdf["period"] = tdf["date"].dt.to_period("M")
                            monthly = tdf.groupby("period")["value"].mean().reset_index()
                            monthly["period_str"] = monthly["period"].astype(str)
                            fig = px.line(monthly, x="period_str", y="value", markers=True,
                                          labels={"period_str": "Period", "value": clean_name(t["value_column"])})
                            fig.update_traces(line_color=COLORS["accent"], fill="tozeroy",
                                              fillcolor="rgba(23,185,150,0.06)", line_width=2.3)
                            plotly_layout(fig, height=300, show_legend=False)
                            st.plotly_chart(fig, width="stretch")
                        except Exception:
                            st.info("Trend could not be rendered for the detected signal.")
                    else:
                        st.info("No suitable date + numeric metric combination was detected.")

                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                split1, split2 = st.columns(2)

                with split1:
                    with ui.card("Anomaly Intelligence", "IQR outlier scan across numeric variables."):
                        if anomalies.get("column_details"):
                            for item in sorted(anomalies["column_details"],
                                                key=lambda x: x["outlier_percentage"], reverse=True)[:4]:
                                st.markdown(
                                    f'<div class="ref-list-row"><div><b>{clean_name(item["column"])}</b>'
                                    f'<small>{item["method"]} · {item["outlier_count"]:,} flagged rows</small></div>'
                                    f'<span class="ref-badge">{item["outlier_percentage"]:.1f}%</span></div>',
                                    unsafe_allow_html=True)
                        else:
                            st.success("No statistical outliers detected.")

                with split2:
                    with ui.card("Category Distribution", "Most informative categorical field detected automatically."):
                        cat_cols = profile["data_types"].get("categorical", [])
                        focus_cat = (patterns["categorical_patterns"][0]["column"]
                                     if patterns.get("categorical_patterns") else
                                     (cat_cols[0] if cat_cols else None))
                        if focus_cat:
                            freq_dict = profile["column_profiles"][focus_cat].get(
                                "categorical_summary", {}).get("top_frequencies", {})
                            for value, count in list(freq_dict.items())[:4]:
                                pct = count / total_rows * 100 if total_rows else 0
                                st.markdown(
                                    f'<div class="ref-segment"><div><span>{str(value)[:34]}</span>'
                                    f'<b>{pct:.1f}%</b></div><div class="ref-track">'
                                    f'<div class="ref-fill" style="width:{min(100,pct):.1f}%"></div></div></div>',
                                    unsafe_allow_html=True)
                        else:
                            st.info("No categorical variable detected.")

                st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
                with ui.card("Business Snapshot", "Dynamic operational observations from the loaded data."):
                    # 1. Key Trend Card
                    if patterns.get("temporal_patterns"):
                        t = patterns["temporal_patterns"][0]
                        trend_desc = (
                            f"<b>{clean_name(t['value_column'])}</b> shows an <b>{t['trend_type']}</b> trend "
                            f"across the temporal index (r = {t['correlation']:.2f})."
                        )
                        trend_kind = "pos" if t["trend_type"] == "increasing" else "neg" if t["trend_type"] == "decreasing" else "info"
                    else:
                        trend_desc = "No distinct linear trend was automatically detected across temporal dimensions."
                        trend_kind = "info"
                    ui.insight_card("Key Trend", trend_desc, trend_kind)
                    
                    # 2. Data Health Card
                    health_status = profile["overall_status"]
                    warn_count = len(profile.get("warnings", []))
                    health_desc = (
                        f"Overall data health is evaluated as <b>{health_status}</b>. "
                        f"A total of <b>{warn_count}</b> data quality alert(s) or warnings were flagged."
                    )
                    health_kind = "pos" if health_status == "Good" else "warn" if health_status == "Needs Attention" else "neg"
                    ui.insight_card("Data Health", health_desc, health_kind)
                    
                    # 3. Anomaly Signal Card
                    anom_pct = anomalies["anomaly_percentage"]
                    anom_count = anomalies["total_anomalies"]
                    if anom_count > 0:
                        anom_desc = (
                            f"Statistical scan flagged <b>{anom_count:,}</b> outlier record(s) "
                            f"(<b>{anom_pct:.2f}%</b> anomaly rate) requiring investigation."
                        )
                        anom_kind = "warn" if anom_pct > 1.0 else "info"
                    else:
                        anom_desc = "No statistical anomalies or outliers were detected across numeric dimensions."
                        anom_kind = "pos"
                    ui.insight_card("Anomaly Signal", anom_desc, anom_kind)
                    
                    # 4. Dominant Segment / Important Metric Card
                    if patterns.get("categorical_patterns"):
                        cp = patterns["categorical_patterns"][0]
                        seg_desc = (
                            f"<b>{clean_name(cp['column'])}</b> is dominated by value <b>'{cp['dominant_value']}'</b>, "
                            f"accounting for <b>{cp['percentage']:.1f}%</b> of all records."
                        )
                        seg_kind = "info"
                    elif kpis:
                        k = kpis[0]
                        seg_desc = f"Key operational KPI <b>{k['name']}</b> has a current value of <b>{format_kpi_value(k['value'])}</b>."
                        seg_kind = "info"
                    else:
                        seg_desc = "No dominant categorical segment or unique KPI metrics were discovered."
                        seg_kind = "info"
                    ui.insight_card("Dominant Segment / Important Metric", seg_desc, seg_kind)

            with main_right:
                with ui.card("SynTwin AI Assistant", "Grounded in your live analysis."):
                    context_items = [
                        ("Dataset", True),
                        ("Diagnosis", True),
                        ("SHAP Drivers", any(k.startswith("shap_vals_") for k in st.session_state.keys())),
                        ("Forecast", any(k.startswith("forecast_res_") for k in st.session_state.keys())),
                        ("Digital Twin", "last_sim_res" in st.session_state),
                        ("Decision Engine", "last_ga_res" in st.session_state or "last_rl_res" in st.session_state),
                        ("RAG Knowledge", "vector_store" in st.session_state and st.session_state["vector_store"].get_chunks_count() > 0),
                    ]
                    for label, active in context_items:
                        state_text = "● Active" if active else "○ Ready"
                        state_class = "active" if active else "inactive"
                        st.markdown(
                            f'<div class="ref-context"><span>{label}</span>'
                            f'<b class="{state_class}">{state_text}</b></div>',
                            unsafe_allow_html=True)
                    st.markdown(
                        '<div class="ref-chat-preview"><div class="ref-chat-icon">◆</div><div>'
                        '<b>Ask SynTwin about your data</b><p>Explore risks, model drivers, forecasts and decisions without configuring the engines.</p>'
                        '</div></div>', unsafe_allow_html=True)
                    if st.button("Open AI Assistant", key="overview_open_ai", type="primary", width="stretch"):
                        st.session_state["active_nav"] = "AI Assistant"
                        st.rerun()

            with st.expander("Preview ingested data", expanded=False):
                st.dataframe(df.head(10), width="stretch")

        # ================================================
        # DATA PROFILE
        # ================================================
        elif navigation_selection == "Data Profile":
            tab_schema, tab_num, tab_cat = st.tabs(
                ["Column Schema Properties", "Numerical Distributions", "Categorical Distributions"]
            )

            with tab_schema:
                ui.section_title("Column Properties Schema", "Inferred type, cardinality and role for every column.")
                rows_html = []
                for col, info in profile["column_profiles"].items():
                    if col in profile["inferred_columns"]["ids"]:
                        role_class, role_name = "id", "Identifier"
                    elif col in profile["inferred_columns"]["targets"]:
                        role_class, role_name = "target", "Target Candidate"
                    elif col in profile["inferred_columns"]["dates"]:
                        role_class, role_name = "date", "Datetime"
                    elif info["type_group"] == "boolean":
                        role_class, role_name = "boolean", "Boolean"
                    elif info["type_group"] == "categorical":
                        role_class, role_name = "categorical", "Categorical"
                    else:
                        role_class, role_name = "numeric", "Numerical"

                    rows_html.append(
                        f"<tr style='border-bottom:1.5px solid {COLORS['border']};'>"
                        f"<td style='padding:9px 12px; font-weight:600; color:{COLORS['text']};'>{col}</td>"
                        f"<td style='padding:9px 12px; font-family:monospace; font-size:0.8rem; color:{COLORS['text_muted']};'>{info['dtype']}</td>"
                        f"<td style='padding:9px 12px;'>{info['unique_count']:,}</td>"
                        f"<td style='padding:9px 12px;'>{info['missing_count']:,}</td>"
                        f"<td style='padding:9px 12px;'>{info['missing_percentage']:.2f}%</td>"
                        f"<td style='padding:9px 12px;'>{ui.badge(role_name, role_class)}</td>"
                        f"</tr>"
                    )
                table_html = (
                    f"<div class='stw-card' style='padding:0; overflow:hidden;'>"
                    f"<table style='width:100%; border-collapse:collapse;'>"
                    f"<thead><tr style='background:{COLORS['surface_alt']}; text-align:left; border-bottom:2px solid {COLORS['border_strong']};'>"
                    f"<th style='padding:10px 12px; font-size:0.75rem; text-transform:uppercase; color:{COLORS['text_muted']};'>Column</th>"
                    f"<th style='padding:10px 12px; font-size:0.75rem; text-transform:uppercase; color:{COLORS['text_muted']};'>Type</th>"
                    f"<th style='padding:10px 12px; font-size:0.75rem; text-transform:uppercase; color:{COLORS['text_muted']};'>Unique</th>"
                    f"<th style='padding:10px 12px; font-size:0.75rem; text-transform:uppercase; color:{COLORS['text_muted']};'>Missing</th>"
                    f"<th style='padding:10px 12px; font-size:0.75rem; text-transform:uppercase; color:{COLORS['text_muted']};'>Missing %</th>"
                    f"<th style='padding:10px 12px; font-size:0.75rem; text-transform:uppercase; color:{COLORS['text_muted']};'>Inferred Role</th>"
                    f"</tr></thead><tbody>{''.join(rows_html)}</tbody></table></div>"
                )
                st.markdown(table_html, unsafe_allow_html=True)

            with tab_num:
                numeric_cols = profile["data_types"]["numerical"]
                if numeric_cols:
                    selected_num = st.selectbox("Select numerical feature", numeric_cols)
                    stats = profile["column_profiles"][selected_num].get("numerical_summary", {})
                    if stats:
                        sc1, sc2 = st.columns([1, 2])
                        with sc1:
                            with ui.card(f"Summary — {clean_name(selected_num)}"):
                                st.write(pd.Series(stats, name="Value"))
                        with sc2:
                            with ui.card(f"Distribution — {clean_name(selected_num)}"):
                                fig = px.histogram(df, x=selected_num)
                                fig.update_traces(marker_color=COLORS["accent"])
                                plotly_layout(fig, height=340)
                                st.plotly_chart(fig, width="stretch")
                    else:
                        st.info("No statistics computed for this column.")
                else:
                    st.info("No numeric columns present in this dataset.")

            with tab_cat:
                categorical_cols = profile["data_types"]["categorical"]
                if categorical_cols:
                    selected_cat = st.selectbox("Select categorical feature", categorical_cols)
                    cat_summary = profile["column_profiles"][selected_cat].get("categorical_summary", {})
                    if cat_summary:
                        sc1, sc2 = st.columns([1, 2])
                        with sc1:
                            st.metric("Unique Classes", cat_summary["num_categories"])
                            st.metric("Top Class (Mode)", cat_summary["top_value"])
                            st.metric("Top Class Frequency", f"{cat_summary['top_frequency']:,} ({cat_summary['top_percentage']:.1f}%)")
                        with sc2:
                            with ui.card(f"Class Frequency — {clean_name(selected_cat)}"):
                                freq_dict = cat_summary["top_frequencies"]
                                if freq_dict:
                                    freq_df = pd.DataFrame(list(freq_dict.items()), columns=["Value", "Count"]).sort_values("Count", ascending=False)
                                    fig = px.bar(freq_df, x="Value", y="Count")
                                    fig.update_traces(marker_color=COLORS["good"])
                                    plotly_layout(fig, height=300)
                                    st.plotly_chart(fig, width="stretch")
                    else:
                        st.info("No statistics computed for this column.")
                else:
                    st.info("No categorical columns present in this dataset.")

        # ================================================
        # DATA QUALITY (Redesigned as separate view)
        # ================================================
        elif navigation_selection == "Data Quality":
            status = profile["overall_status"]
            kind = "good" if status == "Good" else ("warn" if status == "Needs Attention" else "bad")
            label = {
                "good": "Dataset Health is GOOD — Ingestion checks passed without issue.",
                "warn": "Dataset Health requires ATTENTION — Minor anomalies/outliers detected.",
                "bad": "Dataset Health is CRITICAL — Serious data anomalies/missing parameters detected.",
            }[kind]
            ui.status_banner(label, kind)

            ui.section_title("Automated Data Quality Diagnostics")
            
            # Show summary stats
            dq_c1, dq_c2, dq_c3, dq_c4 = st.columns(4)
            with dq_c1:
                ui.kpi_card("Duplicate Rows", f"{profile['overview']['duplicate_rows']:,}", "Identical rows")
            with dq_c2:
                total_missing = sum(info["missing_count"] for info in profile["column_profiles"].values())
                ui.kpi_card("Missing Values", f"{total_missing:,}", "Across all cells")
            with dq_c3:
                constant_cols = len([col for col, info in profile["column_profiles"].items() if info.get("unique_count") == 1])
                ui.kpi_card("Constant Columns", str(constant_cols), "Single value columns")
            with dq_c4:
                high_card_cols = len([col for col, info in profile["column_profiles"].items() if info.get("type_group") == "categorical" and info.get("unique_count", 0) > 100])
                ui.kpi_card("High Cardinality", str(high_card_cols), "Cols with > 100 values")

            ui.section_title("Detailed Quality Observations")
            warnings = profile.get("warnings", [])
            if not warnings:
                st.success("No data quality warnings recorded for this dataset.")
            else:
                for w in warnings:
                    sev = "neg" if w["severity"] == "high" else "warn" if w["severity"] == "medium" else "info"
                    ui.insight_card(f"{clean_name(w['column'])} — {w['type'].upper()}", w["message"], sev)

            missing_cols = {col: info["missing_percentage"] for col, info in profile["column_profiles"].items() if info["missing_count"] > 0}
            if missing_cols:
                ui.section_title("Missing Values by Column")
                missing_df = pd.DataFrame(list(missing_cols.items()), columns=["Column", "Missing %"]).sort_values("Missing %", ascending=False)
                fig = px.bar(missing_df, x="Column", y="Missing %")
                fig.update_traces(marker_color=COLORS["bad"])
                plotly_layout(fig, height=300)
                st.plotly_chart(fig, width="stretch")

        # ================================================
        # DIAGNOSIS
        # ================================================
        elif navigation_selection == "Diagnosis":
            with st.spinner("Analyzing KPIs, patterns and anomalies..."):
                kpis, patterns, anomalies = run_diagnosis_engines(df, profile)

            ui.section_title("Discovered Key Performance Indicators")
            if kpis:
                display_kpis = kpis[:6]
                cards = [{"label": k["name"], "value": format_kpi_value(k["value"]), "sub": f"{k['type']} · {k['interpretation']}"} for k in display_kpis]
                ui.kpi_row(cards, columns=3)
            else:
                st.info("No business KPIs could be automatically extracted from this dataset structure.")

            st.markdown("---")
            ui.section_title("Automatically Detected Patterns")
            p_col1, p_col2 = st.columns(2)

            with p_col1:
                with ui.card("Numerical & Categorical Insights"):
                    has_insights = False
                    for corr in patterns["correlations"][:3]:
                        has_insights = True
                        direction = "positive" if corr["coefficient"] > 0 else "inverse"
                        ui.insight_card(
                            "Strong relationship",
                            f"<b>{clean_name(corr['col1'])}</b> and <b>{clean_name(corr['col2'])}</b> show a "
                            f"<b>{direction}</b> relationship. Correlation coefficient: <b>{corr['coefficient']:.3f}</b>.",
                            "info",
                        )
                    for cat in patterns["categorical_patterns"]:
                        has_insights = True
                        ui.insight_card(
                            "Dominant category class",
                            f"Feature <b>{clean_name(cat['column'])}</b> is heavily dominated by value "
                            f"<b>'{cat['dominant_value']}'</b>, representing <b>{cat['percentage']:.1f}%</b> of all rows.",
                            "warn",
                        )
                    if not has_insights:
                        st.info("No significant numerical correlations or categorical skews detected.")

            with p_col2:
                with ui.card("Temporal Insights"):
                    if patterns["temporal_patterns"]:
                        for temp in patterns["temporal_patterns"]:
                            ui.insight_card(
                                "Temporal trend detected",
                                f"The average of <b>{clean_name(temp['value_column'])}</b> shows an "
                                f"<b>{temp['trend_type']}</b> trend across {temp['periods_count']} monthly periods "
                                f"of <b>{clean_name(temp['date_column'])}</b>. Trend strength: <b>{temp['correlation']:.2f}</b>.",
                                "pos" if temp["trend_type"] == "increasing" else "neg" if temp["trend_type"] == "decreasing" else "info",
                            )
                            try:
                                temp_df = pd.DataFrame({
                                    "date": pd.to_datetime(df[temp["date_column"]], errors="coerce", format="mixed"),
                                    "value": pd.to_numeric(df[temp["value_column"]], errors="coerce"),
                                }).dropna()
                                temp_df["period"] = temp_df["date"].dt.to_period("M")
                                monthly_agg = temp_df.groupby("period")["value"].mean().reset_index()
                                monthly_agg["period_str"] = monthly_agg["period"].astype(str)
                                fig = px.line(monthly_agg, x="period_str", y="value", labels={"period_str": "Month", "value": f"Avg {clean_name(temp['value_column'])}"})
                                fig.update_traces(line_color=COLORS["good"])
                                plotly_layout(fig, height=260)
                                st.plotly_chart(fig, width="stretch")
                            except Exception:
                                pass
                    else:
                        st.info("No temporal indicators detected in the dataset to perform time trend checks.")

            st.markdown("---")
            ui.section_title("Statistical Outlier Analysis")
            col_an1, col_an2, col_an3 = st.columns(3)
            with col_an1:
                st.metric("Columns with Outliers", anomalies["columns_with_anomalies"])
            with col_an2:
                st.metric("Total Outlier Records", f"{anomalies['total_anomalies']:,}")
            with col_an3:
                st.metric("Dataset Outlier Ratio", f"{anomalies['anomaly_percentage']:.2f}%")

            if anomalies["column_details"]:
                anom_table_df = pd.DataFrame([
                    {
                        "Column": clean_name(item["column"]),
                        "Outlier Count": f"{item['outlier_count']:,}",
                        "Outlier %": f"{item['outlier_percentage']:.2f}%",
                        "Min Outlier Value": f"{item['min_outlier']:.4f}" if item["min_outlier"] is not None else "N/A",
                        "Max Outlier Value": f"{item['max_outlier']:.4f}" if item["max_outlier"] is not None else "N/A",
                        "Method": item["method"],
                    }
                    for item in anomalies["column_details"]
                ])
                st.dataframe(anom_table_df, width="stretch")

                fig = px.bar(
                    anom_table_df.head(10), x="Column",
                    y=[float(v.replace("%", "")) for v in anom_table_df.head(10)["Outlier %"]],
                    labels={"y": "Outliers (%)"},
                )
                fig.update_traces(marker_color=COLORS["bad"])
                plotly_layout(fig, height=300)
                st.plotly_chart(fig, width="stretch")
            else:
                st.success("No statistical outliers detected in this dataset.")

            st.markdown("---")
            ui.section_title("Correlation Heatmap")
            numeric_cols = profile.get("data_types", {}).get("numerical", [])
            if len(numeric_cols) > 1:
                selected_num_cols = numeric_cols[:15]
                corr_matrix = df[selected_num_cols].corr()
                names = [clean_name(c) for c in corr_matrix.columns]
                
                # Dynamically hide text labels if there are more than 10 columns to prevent visual clutter
                show_text = len(selected_num_cols) <= 10
                text_val = np.round(corr_matrix.values, 2) if show_text else None
                text_tmpl = "%{text}" if show_text else None
                
                fig = go.Figure(data=go.Heatmap(
                    z=corr_matrix.values, x=names, y=names, colorscale="RdBu", zmin=-1, zmax=1,
                    text=text_val, texttemplate=text_tmpl, showscale=True,
                    hovertemplate="Variable X: %{x}<br>Variable Y: %{y}<br>Correlation: %{z:.3f}<extra></extra>"
                ))
                
                # Apply compact margins and responsive font sizing
                fig.update_layout(
                    margin=dict(l=40, r=40, t=20, b=40),
                    xaxis=dict(tickangle=45, tickfont=dict(size=10)),
                    yaxis=dict(tickfont=dict(size=10))
                )
                plotly_layout(fig, height=360)
                st.plotly_chart(fig, width="stretch")
            else:
                st.info("At least two numerical columns are required to construct a correlation heatmap.")

        # ================================================
        # PREDICTION
        # ================================================
        elif navigation_selection == "Prediction":
            targets_info = detect_targets(df, profile)

            if targets_info["best_target"] is None:
                st.warning("No clear prediction target was detected for this dataset.")
            else:
                selected_target = targets_info["best_target"]
                with st.expander("Change target (optional)", expanded=False):
                    all_targets = [targets_info["best_target"]] + [alt["column"] for alt in targets_info["alternatives"]]
                    selected_target = st.selectbox("Prediction target", all_targets, index=0)
                st.info(f"**Recommended target:** {targets_info['reason']}")

                selected_type = problem_type_for(profile, selected_target)
                pill_col1, pill_col2 = st.columns([1, 3])
                with pill_col1:
                    ui.pill(f"{clean_name(selected_type)}")

                if len(df) > 10000:
                    st.info("Large dataset detected — training on a representative sample of 10,000 rows for faster analysis.")

                train_button = st.button("Train Predictive Models", type="primary")

                model_key = f"model_{selected_target}"
                prepr_key = f"preprocessor_{selected_target}"
                metrics_key = f"metrics_{selected_target}"
                meta_key = f"meta_{selected_target}"

                if train_button:
                    # Clear previous models to avoid stale state
                    st.session_state.pop(model_key, None)
                    st.session_state.pop(prepr_key, None)
                    st.session_state.pop(metrics_key, None)
                    st.session_state.pop(meta_key, None)
                    
                    status_placeholder = st.empty()
                    try:
                        with status_placeholder.container():
                            st.info("Preparing representative data...")
                        
                        train_df = get_representative_sample(df, selected_target, selected_type, sample_size=10000)
                        
                        with status_placeholder.container():
                            st.info("Preprocessing features...")
                        
                        date_cols = profile.get("inferred_columns", {}).get("dates", []) + profile.get("data_types", {}).get("datetime", [])
                        primary_date = date_cols[0] if date_cols else None
                        
                        preprocessor = DataPreprocessor(target_col=selected_target, target_type=selected_type, date_col=primary_date)
                        X_train, X_test, y_train, y_test, feat_names = preprocessor.split_and_preprocess(train_df)
                        
                        with status_placeholder.container():
                            st.info("Training model...")
                        
                        best_model, best_name, scores = train_best_model(X_train, y_train, X_test, y_test, selected_type)
                        
                        with status_placeholder.container():
                            st.info("Evaluating model...")
                        
                        metrics = evaluate_model(best_model, X_test, y_test, selected_type)
                        
                        st.session_state[model_key] = best_model
                        st.session_state[prepr_key] = preprocessor
                        st.session_state[metrics_key] = metrics
                        st.session_state[meta_key] = {
                            "best_name": best_name,
                            "test_samples": len(y_test),
                            "train_samples": len(y_train),
                            "comparison_scores": scores,
                            "feature_names": feat_names,
                            "X_test_proc": X_test,
                            "y_test": y_test,
                            "predictions": best_model.predict(X_test),
                        }
                        if hasattr(best_model, "predict_proba"):
                            st.session_state[meta_key]["probs"] = best_model.predict_proba(X_test)
                        
                        status_placeholder.empty()
                        st.success("Complete!")
                        st.rerun()
                        
                    except Exception as e:
                        status_placeholder.empty()
                        st.error(f"Prediction model training failed: {str(e)}")

                if model_key in st.session_state:
                    metrics = st.session_state[metrics_key]
                    meta = st.session_state[meta_key]

                    ui.status_banner(f"Trained best model ({meta['best_name']}) successfully.", "good")

                    ui.section_title("Model Summary")
                    primary_metric = "F1 Score" if selected_type in ["binary_classification", "multiclass_classification"] else "R2 Score"
                    primary_val = metrics.get(primary_metric, 0.0)
                    ui.kpi_row([
                        {"label": "Best Model", "value": meta["best_name"]},
                        {"label": "Test Sample Size", "value": f"{meta['test_samples']:,}"},
                        {"label": f"Primary Metric ({primary_metric})", "value": f"{primary_val:.4f}"},
                        {"label": "Features Used", "value": len(meta["feature_names"])},
                    ], columns=4)

                    ui.section_title("Performance Metrics")
                    pm_cols = st.columns(2)
                    with pm_cols[0]:
                        with ui.card("Validation Scores"):
                            for m_name, m_val in metrics.items():
                                st.write(f"**{m_name}**: `{m_val:.4f}`")
                            st.markdown("**Model Comparison**")
                            for comp_name, comp_score in meta["comparison_scores"].items():
                                st.write(f"{comp_name}: `{comp_score:.4f}`")

                    with pm_cols[1]:
                        with ui.card("Diagnostic Visualization"):
                            y_test = meta["y_test"]
                            preds = meta["predictions"]
                            if selected_type == "regression":
                                viz_df = pd.DataFrame({"Actual": y_test, "Predicted": preds})
                                fig = px.scatter(viz_df, x="Actual", y="Predicted", trendline="ols")
                                fig.update_traces(marker_color=COLORS["accent"])
                                plotly_layout(fig, height=320)
                                st.plotly_chart(fig, width="stretch")
                            else:
                                from sklearn.metrics import confusion_matrix
                                cm = confusion_matrix(y_test, preds)
                                classes = sorted(list(set(y_test)))
                                fig = go.Figure(data=go.Heatmap(
                                    z=cm, x=[str(c) for c in classes], y=[str(c) for c in classes],
                                    colorscale="Blues", text=cm, texttemplate="%{text}", showscale=True,
                                ))
                                fig.update_layout(xaxis_title="Predicted", yaxis_title="Actual")
                                plotly_layout(fig, height=320)
                                st.plotly_chart(fig, width="stretch")

                    ui.section_title("Predictions & Sample Inferences")
                    y_test = meta["y_test"]
                    preds = meta["predictions"]
                    if selected_type == "regression":
                        pred_df = pd.DataFrame({"Actual": y_test, "Predicted": preds, "Prediction Error": preds - y_test})
                        st.dataframe(pred_df.head(15), width="stretch")
                    else:
                        conf = [float(np.max(row)) for row in meta["probs"]] if "probs" in meta else []
                        pred_data = {"Actual": y_test, "Predicted": preds}
                        if conf:
                            pred_data["Confidence (%)"] = [f"{c*100:.1f}%" for c in conf]
                        st.dataframe(pd.DataFrame(pred_data).head(15), width="stretch")

        # ================================================
        # EXPLAINABILITY
        # ================================================
        elif navigation_selection == "Explainability":
            trained_targets = [k.replace("model_", "") for k in st.session_state.keys() if k.startswith("model_")]

            if not trained_targets:
                st.warning("Train a predictive model first to enable explainability.")
                st.info("Go to **Prediction**, select a target variable, and click **Train Predictive Models**.")
            else:
                target_to_explain = trained_targets[0]

                model_key = f"model_{target_to_explain}"
                prepr_key = f"preprocessor_{target_to_explain}"
                meta_key = f"meta_{target_to_explain}"
                model = st.session_state[model_key]
                preprocessor = st.session_state[prepr_key]
                meta = st.session_state[meta_key]

                target_type = problem_type_for(profile, target_to_explain)
                ui.pill(f"{meta['best_name']} · {clean_name(target_to_explain)} · {clean_name(target_type)}")
                st.caption("SHAP values show what a feature *contributed to* the prediction — this does not establish that it *caused* the outcome.")

                class_idx = 0
                if target_type == "multiclass_classification":
                    classes = sorted(list(set(meta["y_test"])))
                    class_idx = 0
                elif target_type == "binary_classification":
                    class_idx = 1

                shap_cache_key = f"shap_vals_{target_to_explain}_{class_idx}"
                base_val_cache_key = f"shap_base_{target_to_explain}_{class_idx}"

                if shap_cache_key not in st.session_state:
                    with st.spinner("Calculating SHAP explainability values..."):
                        from src.explainability.shap_engine import get_shap_values
                        X_test_proc = meta["X_test_proc"]
                        shap_vals, base_value = get_shap_values(model, X_test_proc, X_test_proc, target_type, class_idx)
                        st.session_state[shap_cache_key] = shap_vals
                        st.session_state[base_val_cache_key] = base_value

                shap_vals = st.session_state[shap_cache_key]
                base_value = st.session_state[base_val_cache_key]

                tab_g, tab_l = st.tabs(["Global Drivers", "Explain a Prediction"])

                with tab_g:
                    ui.section_title("Global Drivers", "Features ranked by average absolute impact (mean |SHAP value|) on the model output.")
                    global_importance_key = f"global_importance_{target_to_explain}_{class_idx}"
                    if global_importance_key not in st.session_state:
                        from src.explainability.shap_engine import explain_globally
                        st.session_state[global_importance_key] = explain_globally(model, meta["X_test_proc"], meta["X_test_proc"], meta["feature_names"], target_type, class_idx)
                    global_importance = st.session_state[global_importance_key]

                    top_n = min(10, len(global_importance))
                    top_features = global_importance[:top_n]
                    importance_df = pd.DataFrame(top_features)
                    importance_df["clean_feature"] = importance_df["feature"].apply(clean_name)

                    fig = px.bar(
                        importance_df, x="importance", y="clean_feature", orientation="h",
                        labels={"importance": "Mean |SHAP value|", "clean_feature": "Feature"},
                        color="importance", color_continuous_scale=styles.CHART_ACCENT_SCALE,
                    )
                    fig.update_layout(yaxis={"categoryorder": "total ascending"})
                    plotly_layout(fig, height=max(320, top_n * 26))
                    st.plotly_chart(fig, width="stretch")

                    if top_features:
                        strongest_feat = clean_name(top_features[0]["feature"])
                        ui.insight_card(
                            "Automated explainability insight",
                            f"<b>{strongest_feat}</b> is the strongest global driver of the model's predictions, "
                            f"contributing approximately <b>{top_features[0]['relative_importance']*100:.1f}%</b> of total feature impact.",
                            "info",
                        )

                with tab_l:
                    ui.section_title("Explain a Prediction", "Select an individual test record to see how each feature contributed to its prediction.")
                    test_len = len(meta["y_test"])
                    row_select = 0
                    with st.expander("Choose another observation (optional)", expanded=False):
                        row_select = st.number_input("Observation index", min_value=0, max_value=test_len - 1, value=0, step=1)

                    local_expl_key = f"local_expl_{target_to_explain}_{class_idx}_{row_select}"
                    if local_expl_key not in st.session_state:
                        from src.explainability.shap_engine import explain_locally
                        raw_row = preprocessor.X_test_raw.iloc[row_select] if hasattr(preprocessor, "X_test_raw") else None
                        raw_feature_map = preprocessor.get_raw_feature_map() if hasattr(preprocessor, "get_raw_feature_map") else None
                        st.session_state[local_expl_key] = explain_locally(
                            model, meta["X_test_proc"], meta["X_test_proc"], row_select, meta["feature_names"], target_type, class_idx,
                            raw_row=raw_row, raw_feature_map=raw_feature_map,
                        )
                    local_expl = st.session_state[local_expl_key]

                    y_true = meta["y_test"][row_select]
                    y_pred = meta["predictions"][row_select]
                    col_pred1, col_pred2 = st.columns(2)
                    with col_pred1:
                        st.metric("Actual Value", f"{y_true}" if target_type != "regression" else f"{y_true:,.2f}")
                    with col_pred2:
                        st.metric("Predicted Value", f"{y_pred}" if target_type != "regression" else f"{y_pred:,.2f}")

                    ui.section_title("Feature Influence Breakdown")
                    local_contribs = local_expl["contributions"]
                    top_contribs = local_contribs[:8]
                    other_contribs = local_contribs[8:]

                    waterfall_data = [{"name": "Base Value (Mean Prediction)", "val": base_value, "type": "absolute"}]
                    for c in top_contribs:
                        fv = c["feature_value"]
                        fv_label = f"{fv:,.2f}" if isinstance(fv, (int, float)) else str(fv)
                        waterfall_data.append({"name": f"{clean_name(c['feature'])} ({fv_label})", "val": c["shap_value"], "type": "relative"})
                    if other_contribs:
                        other_sum = sum(c["shap_value"] for c in other_contribs)
                        waterfall_data.append({"name": f"Other {len(other_contribs)} Features", "val": other_sum, "type": "relative"})
                    waterfall_data.append({"name": "Final Prediction", "val": local_expl["prediction_value"], "type": "total"})

                    w_df = pd.DataFrame(waterfall_data)
                    fig = go.Figure(go.Waterfall(
                        name="SHAP Waterfall", orientation="v", measure=w_df["type"].tolist(), x=w_df["name"].tolist(),
                        textposition="outside",
                        text=[f"{v:+.3f}" if t not in ("total", "absolute") else f"{v:.3f}" for v, t in zip(w_df["val"], w_df["type"])],
                        y=w_df["val"].tolist(),
                        connector={"line": {"color": COLORS["border_strong"]}},
                        decreasing={"marker": {"color": COLORS["bad"]}},
                        increasing={"marker": {"color": COLORS["good"]}},
                        totals={"marker": {"color": COLORS["accent"]}},
                    ))
                    plotly_layout(fig, height=380, show_legend=False)
                    st.plotly_chart(fig, width="stretch")

                    ui.section_title("Contribution Explanations")
                    for c in top_contribs[:5]:
                        impact_direction = "pushed the prediction higher" if c["shap_value"] > 0 else "pushed the prediction lower"
                        fv = c["feature_value"]
                        fv_str = f"{fv:,.2f}" if isinstance(fv, (int, float)) else str(fv)
                        st.markdown(
                            f"- The variable **{clean_name(c['feature'])}** (value *{fv_str}*) **{impact_direction}**, "
                            f"contributing a SHAP value of **{c['shap_value']:+.4f}**."
                        )

        # ================================================
        # FORECAST
        # ================================================
        elif navigation_selection == "Forecast":
            import time
            start_render = time.time()
            
            # Cache the eligibility check in session state to avoid parsing dates on every rerun
            if "forecast_eligibility" not in st.session_state:
                from src.forecasting.detector import detect_forecasting_eligibility
                st.session_state["forecast_eligibility"] = detect_forecasting_eligibility(df, profile)
            eligibility = st.session_state["forecast_eligibility"]

            if not eligibility["eligible"]:
                st.warning("Forecasting is unavailable because no suitable date and business metric were detected.")
                st.info(f"Reason: {eligibility['reason']}")
            else:
                ui.section_title("Forecast", "Automatic configuration keeps the workflow simple.")
                date_candidates = eligibility["date_columns"]
                rec_date = eligibility["recommended_date"]
                selected_date_col = rec_date if rec_date in date_candidates else date_candidates[0]
                metric_candidates = eligibility["metric_columns"]
                rec_metric = eligibility["recommended_metric"]
                selected_metric_col = rec_metric if rec_metric in metric_candidates else metric_candidates[0]
                base_freq = eligibility["frequency"] or "D"
                freq_val = "W" if "W" in base_freq else "M" if "M" in base_freq else "D"
                freq_name = {"D": "daily", "W": "weekly", "M": "monthly"}[freq_val]
                selected_horizon = {"D": 30, "W": 12, "M": 12}[freq_val]
                agg_op = "Sum"
                st.caption(f"Using **{clean_name(selected_metric_col)}** over **{clean_name(selected_date_col)}** · {freq_name} · {selected_horizon} periods.")
                fit_forecast = st.button("Generate Future Forecasts", type="primary")
                forecast_cache_key = f"forecast_res_{selected_date_col}_{selected_metric_col}_{selected_horizon}_{freq_val}_{agg_op}"

                if fit_forecast:
                    with st.spinner("Generating forecast..."):
                        from src.forecasting.forecaster import prepare_time_series, train_and_forecast
                        try:
                            start_calc = time.time()
                            series = prepare_time_series(df, selected_date_col, selected_metric_col, freq=freq_val, agg_func=agg_op.lower())
                            res = train_and_forecast(series, selected_horizon, freq_val)
                            calc_duration = time.time() - start_calc
                            print(f"[TIMING] Forecast calculation took: {calc_duration:.4f} seconds")
                            st.session_state[forecast_cache_key] = {"results": res, "series": series}
                        except Exception as e:
                            st.error(f"Failed to fit forecasting models: {str(e)}")

                if forecast_cache_key in st.session_state:
                    cache_item = st.session_state[forecast_cache_key]
                    res = cache_item["results"]
                    series = cache_item["series"]

                    ui.status_banner(f"Time-series model generated successfully. Best model: {res['best_model']}", "good")

                    ui.section_title("Model Selection Performance")
                    comp_data = []
                    for model_name, m in res["comparison"].items():
                        is_best = "Selected" if model_name == res["best_model"] else ""
                        comp_data.append({
                            "Model": model_name, "MAE": f"{m['MAE']:,.4f}", "RMSE": f"{m['RMSE']:,.4f}",
                            "MAPE": f"{m['MAPE']*100:.2f}%" if m["MAPE"] != float("inf") else "N/A", "Status": is_best,
                        })
                    st.dataframe(pd.DataFrame(comp_data), width="stretch")
                    st.caption("Models are evaluated on a chronological 20% validation split of the historical time series.")

                    ui.section_title("Interactive Forecast Chart")
                    start_chart = time.time()
                    forecast_df = res["forecast_df"]
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=series.index, y=series.values, mode="lines", name="Historical", line=dict(color=COLORS["accent"], width=2.5)))
                    fig.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df["Forecast"], mode="lines+markers", name=f"Forecast ({res['best_model']})", line=dict(color=COLORS["good"], width=2.5, dash="dash")))
                    fig.add_trace(go.Scatter(
                        x=list(forecast_df.index) + list(forecast_df.index)[::-1],
                        y=list(forecast_df["Upper Bound"]) + list(forecast_df["Lower Bound"])[::-1],
                        fill="toself", fillcolor="rgba(18,183,106,0.12)", line=dict(color="rgba(255,255,255,0)"),
                        hoverinfo="skip", name="95% Confidence Interval",
                    ))
                    clean_metric_title = clean_name(selected_metric_col)
                    fig.update_layout(xaxis_title="Timeline", yaxis_title=clean_metric_title)
                    plotly_layout(fig, height=380)
                    st.plotly_chart(fig, width="stretch")
                    chart_duration = time.time() - start_chart
                    print(f"[TIMING] Forecast chart rendering took: {chart_duration:.4f} seconds")

                    ui.section_title("Forecasting Insights")
                    recent_avg = np.mean(series.values[-5:])
                    forecast_avg = np.mean(forecast_df["Forecast"].values)
                    diff_pct = ((forecast_avg - recent_avg) / recent_avg) * 100 if recent_avg > 0 else 0
                    direction = "increase" if diff_pct > 1.5 else "decrease" if diff_pct < -1.5 else "remain stable"
                    ui.insight_card(
                        "Automated trend prediction",
                        f"Over the next {selected_horizon} periods, <b>{clean_metric_title}</b> is forecast to <b>{direction}</b>. "
                        f"Average projected shift: <b>{diff_pct:+.2f}%</b> vs. the recent historical period. "
                        f"Expected range: <b>{forecast_df['Lower Bound'].min():,.2f}</b> to <b>{forecast_df['Upper Bound'].max():,.2f}</b>.",
                        "pos" if direction == "increase" else "neg" if direction == "decrease" else "info",
                    )

                    ui.section_title("Projected Data Values")
                    st.dataframe(forecast_df.reset_index().rename(columns={"Date": "Date / Period"}), width="stretch")
            
            render_duration = time.time() - start_render
            print(f"[TIMING] Forecast page render took: {render_duration:.4f} seconds")

        # ================================================
        # DIGITAL TWIN
        # ================================================
        elif navigation_selection == "Digital Twin":
            trained_targets = [k.replace("model_", "") for k in st.session_state.keys() if k.startswith("model_")]

            if not trained_targets:
                st.warning("Train a predictive model before running simulations.")
                st.info("Go to **Prediction**, select a target variable, and click **Train Predictive Models**.")
            else:
                target_to_explain = trained_targets[0]

                model_key = f"model_{target_to_explain}"
                prepr_key = f"preprocessor_{target_to_explain}"
                meta_key = f"meta_{target_to_explain}"
                model = st.session_state[model_key]
                preprocessor = st.session_state[prepr_key]
                meta = st.session_state[meta_key]
                target_type = problem_type_for(profile, target_to_explain)

                ui.pill(f"{meta['best_name']} · {clean_name(target_to_explain)} · {clean_name(target_type)}")

                if not hasattr(preprocessor, "X_test_raw") or preprocessor.X_test_raw is None:
                    st.error("Reference test set not found. Re-train the model in Prediction to enable simulations.")
                else:
                    test_df = preprocessor.X_test_raw
                    test_len = len(test_df)
                    st.markdown("<div class='ref-flow'>CURRENT STATE <span>→</span> SCENARIO <span>→</span> SIMULATE <span>→</span> RESULT</div>", unsafe_allow_html=True)
                    row_select = 0
                    with st.expander("Choose another reference record (optional)", expanded=False):
                        row_select = st.number_input("Observation index", min_value=0, max_value=test_len - 1, value=0, step=1)
                    baseline_row = test_df.iloc[row_select]

                    base_df = pd.DataFrame([baseline_row])
                    X_base_proc = preprocessor.transform_row(base_df)
                    y_base_pred = model.predict(X_base_proc)[0]

                    ui.kpi_row([
                        {"label": "Observation Index", "value": str(row_select)},
                        {"label": f"Baseline Prediction ({clean_name(target_to_explain)})", "value": (f"{y_base_pred}" if target_type != "regression" else f"{y_base_pred:,.2f}")},
                    ], columns=2)

                    st.markdown("---")

                    from src.simulation.scenario_engine import rank_simulation_variables
                    shap_cache_key = f"shap_vals_{target_to_explain}_1"
                    if shap_cache_key not in st.session_state:
                        shap_cache_key = f"shap_vals_{target_to_explain}_0"

                    global_importance = None
                    twin_class_idx = 1 if target_type == "binary_classification" else 0
                    global_importance_key = f"global_importance_{target_to_explain}_{twin_class_idx}"
                    if global_importance_key in st.session_state:
                        global_importance = st.session_state[global_importance_key]
                    elif shap_cache_key in st.session_state:
                        from src.explainability.shap_engine import explain_globally
                        try:
                            global_importance = explain_globally(model, meta["X_test_proc"], meta["X_test_proc"], meta["feature_names"], target_type, class_idx=twin_class_idx)
                            st.session_state[global_importance_key] = global_importance
                        except Exception:
                            pass

                    candidates = rank_simulation_variables(df, profile, global_importance=global_importance, target_col=target_to_explain)

                    ui.section_title("Scenario Builder")
                    selected_vars = [c["feature"] for c in candidates[:3]]

                    changes = []
                    if selected_vars:
                        for col in selected_vars:
                            cand = next(c for c in candidates if c["feature"] == col)
                            orig_val = baseline_row[col]
                            st.write(f"**{clean_name(col)}** — current value: `{orig_val}`")
                            c_col1, c_col2 = st.columns([1, 2])

                            if cand["type"] == "numeric":
                                with c_col1:
                                    st.write("Scenario")
                                with c_col2:
                                    pct_val = st.slider(f"Change {clean_name(col)} (%)", min_value=-30, max_value=30, value=0, step=5, key=f"pct_{col}")
                                    resulting_val = float(orig_val) * (1.0 + pct_val / 100.0)
                                    st.write(f"Simulated value: `{resulting_val:,.2f}`")
                                    changes.append({"feature": col, "type": "percentage", "value": pct_val})
                            elif cand["type"] == "categorical":
                                with c_col2:
                                    cat_val = st.selectbox("Scenario value", cand["categories"], index=cand["categories"].index(orig_val) if orig_val in cand["categories"] else 0, key=f"cat_{col}")
                                    changes.append({"feature": col, "type": "category", "value": cat_val})
                            elif cand["type"] == "boolean":
                                with c_col2:
                                    bool_val = st.toggle("Toggle scenario value", value=bool(orig_val), key=f"bool_{col}")
                                    changes.append({"feature": col, "type": "boolean", "value": bool_val})
                            st.markdown("---")

                    scenario_name = "Quick What-If Scenario"
                    run_sim = st.button("Run What-If Simulation", type="primary")

                    if "saved_scenarios" not in st.session_state:
                        st.session_state["saved_scenarios"] = []

                    if run_sim:
                        with st.spinner("Processing modified twin state..."):
                            from src.simulation.twin_engine import run_twin_simulation
                            try:
                                sim_res = run_twin_simulation(model, preprocessor, baseline_row, changes, target_type, profile)
                                st.session_state["last_sim_res"] = {"name": scenario_name, "target": target_to_explain, "target_type": target_type, "changes": changes, "res": sim_res}
                            except Exception as e:
                                st.error(f"Simulation failed: {str(e)}")

                    if "last_sim_res" in st.session_state:
                        sim_data = st.session_state["last_sim_res"]
                        sim_res = sim_data["res"]

                        if sim_data["target"] == target_to_explain:
                            ui.section_title(f"Simulation Results — {sim_data['name']}")
                            st.caption("Model-based scenario estimate — not a guaranteed outcome.")

                            if sim_res["out_of_range"]:
                                ui.status_banner("This scenario lies outside observed historical bounds. Forecast reliability may be reduced.", "warn")
                                for w in sim_res["warnings"]:
                                    st.write(f"- {w}")

                            sc_col1, sc_col2, sc_col3 = st.columns(3)
                            with sc_col1:
                                val_b = f"{sim_res['baseline_prediction']}" if target_type != "regression" else f"{sim_res['baseline_prediction']:,.2f}"
                                ui.kpi_card("Baseline Prediction", val_b)
                            with sc_col2:
                                val_s = f"{sim_res['scenario_prediction']}" if target_type != "regression" else f"{sim_res['scenario_prediction']:,.2f}"
                                ui.kpi_card("Scenario Prediction", val_s)
                            with sc_col3:
                                if target_type == "regression":
                                    val_diff = f"{sim_res['abs_difference']:+,.2f} ({sim_res['pct_difference']:+.2f}%)"
                                    delta_dir = "up" if sim_res["abs_difference"] > 0 else "down"
                                else:
                                    is_changed = sim_res["scenario_prediction"] != sim_res["baseline_prediction"]
                                    val_diff = "Outcome Shifted" if is_changed else "Stable Outcome"
                                    delta_dir = "flat"
                                ui.kpi_card("Predicted Impact", val_diff, delta_dir=delta_dir)

                            ui.section_title("Scenario Impact Breakdown")
                            impact_records = []
                            for change in changes:
                                col = change["feature"]
                                orig_val = baseline_row[col]
                                sim_val = sim_res["modified_row"][col]
                                if isinstance(orig_val, (int, float)):
                                    diff_str = f"{sim_val - orig_val:+,.2f}"
                                else:
                                    diff_str = "Value Override"
                                impact_records.append({"Feature": clean_name(col), "Baseline": orig_val, "Scenario": sim_val, "Difference": diff_str})
                            st.dataframe(pd.DataFrame(impact_records), width="stretch")

                            ui.section_title("Scenario Comparison")
                            if target_type == "regression":
                                fig = go.Figure(data=[
                                    go.Bar(name="Baseline", x=["Outcome"], y=[sim_res["baseline_prediction"]], marker_color=COLORS["accent"]),
                                    go.Bar(name="Scenario", x=["Outcome"], y=[sim_res["scenario_prediction"]], marker_color=COLORS["warn"]),
                                ])
                                plotly_layout(fig, height=300)
                                st.plotly_chart(fig, width="stretch")
                            elif "baseline_probabilities" in sim_res:
                                fig = go.Figure(data=[
                                    go.Bar(name="Baseline", x=sim_res["class_labels"], y=sim_res["baseline_probabilities"], marker_color=COLORS["accent"]),
                                    go.Bar(name="Scenario", x=sim_res["class_labels"], y=sim_res["scenario_probabilities"], marker_color=COLORS["warn"]),
                                ])
                                fig.update_layout(xaxis_title="Classes", yaxis_title="Probability")
                                plotly_layout(fig, height=300)
                                st.plotly_chart(fig, width="stretch")

                            save_btn = st.button("Save Simulation Scenario")
                            if save_btn:
                                saved_sc = {
                                    "name": sim_data["name"], "baseline_prediction": sim_res["baseline_prediction"],
                                    "scenario_prediction": sim_res["scenario_prediction"],
                                    "baseline_prob_val": sim_res.get("baseline_prob_val"), "scenario_prob_val": sim_res.get("scenario_prob_val"),
                                    "abs_difference": sim_res.get("abs_difference", 0.0), "pct_difference": sim_res.get("pct_difference", 0.0),
                                }
                                if not any(s["name"] == saved_sc["name"] for s in st.session_state["saved_scenarios"]):
                                    st.session_state["saved_scenarios"].append(saved_sc)
                                    st.success(f"Scenario '{saved_sc['name']}' saved.")
                                else:
                                    st.info(f"Scenario name '{saved_sc['name']}' already saved.")

                    if st.session_state.get("saved_scenarios"):
                        ui.section_title("Saved Scenario Comparisons")
                        from src.simulation.comparator import compare_scenarios
                        comp_df = compare_scenarios(st.session_state["saved_scenarios"], target_type)
                        st.dataframe(comp_df, width="stretch")
                        if st.button("Clear Scenarios", type="secondary"):
                            st.session_state["saved_scenarios"] = []
                            st.success("Cleared saved scenarios list.")

        # ================================================
        # DECISION
        # ================================================
        elif navigation_selection == "Decision":
            trained_targets = [k.replace("model_", "") for k in st.session_state.keys() if k.startswith("model_")]

            if not trained_targets:
                st.warning("Train a predictive model before running decisions.")
                st.info("Go to **Prediction**, select a target, and click **Train Predictive Models**.")
            else:
                target_to_explain = trained_targets[0]

                model_key = f"model_{target_to_explain}"
                prepr_key = f"preprocessor_{target_to_explain}"
                meta_key = f"meta_{target_to_explain}"
                model = st.session_state[model_key]
                preprocessor = st.session_state[prepr_key]
                meta = st.session_state[meta_key]

                target_type = problem_type_for(profile, target_to_explain)
                class_idx = (1 if target_type == "binary_classification" else 0) if target_type != "regression" else 0

                objective_mode = "Minimize"
                st.info(f"SynTwin will **minimize the model output** for this decision run. Target: **{clean_name(target_to_explain)}**.")

                from src.decision.action_space import identify_controllable_variables
                shap_cache_key = f"shap_vals_{target_to_explain}_1"
                if shap_cache_key not in st.session_state:
                    shap_cache_key = f"shap_vals_{target_to_explain}_0"

                global_importance = None
                global_importance_key = f"global_importance_{target_to_explain}_{class_idx}"
                if global_importance_key in st.session_state:
                    global_importance = st.session_state[global_importance_key]
                elif shap_cache_key in st.session_state:
                    from src.explainability.shap_engine import explain_globally
                    try:
                        global_importance = explain_globally(model, meta["X_test_proc"], meta["X_test_proc"], meta["feature_names"], target_type, class_idx=class_idx)
                        st.session_state[global_importance_key] = global_importance
                    except Exception:
                        pass

                controllables = identify_controllable_variables(df, profile, target_col=target_to_explain, global_importance=global_importance)
                numeric_controllables = [c for c in controllables if c["type"] == "numeric"]

                if not numeric_controllables:
                    st.error("No numerical controllable variables found in this dataset. Decision intelligence requires at least one controllable numerical feature.")
                else:
                    test_df = preprocessor.X_test_raw
                    test_len = len(test_df)

                    row_select = 0
                    with st.expander("Choose another reference record (optional)", expanded=False):
                        row_select = st.number_input("Observation index", min_value=0, max_value=test_len - 1, value=0, step=1, key="dec_row_select")
                    baseline_row = test_df.iloc[row_select]

                    base_df = pd.DataFrame([baseline_row])
                    X_base_proc = preprocessor.transform_row(base_df)
                    y_base_pred = model.predict(X_base_proc)[0]

                    val_str = f"{y_base_pred}" if target_type != "regression" else f"{y_base_pred:,.2f}"
                    ui.kpi_card("Baseline Reference Prediction", val_str)

                    ui.section_title("Decision Search", "The optimizer uses historical ranges and sensible defaults automatically.")
                    bounds_dict = {var["feature"]: (float(var["min"]), float(var["max"])) for var in numeric_controllables[:5]}
                    allow_out_of_bounds = False
                    pop_size, generations, mutation_rate, crossover_rate = 30, 12, 0.15, 0.8
                    st.caption("Searching within observed historical ranges. Advanced optimization settings stay internal.")
                    run_ga = st.button("Find Best Action", type="primary")

                    if run_ga:
                        with st.spinner("Executing genetic algorithm optimization search..."):
                            from src.decision.genetic_optimizer import GeneticOptimizer
                            try:
                                opt = GeneticOptimizer(
                                    model=model, preprocessor=preprocessor, baseline_row=baseline_row,
                                    controllable_vars=numeric_controllables, target_type=target_type,
                                    objective_mode=objective_mode, class_idx=class_idx, pop_size=pop_size,
                                    generations=generations, mutation_rate=mutation_rate, crossover_rate=crossover_rate,
                                    bounds_dict=bounds_dict, allow_out_of_bounds=allow_out_of_bounds,
                                )
                                st.session_state["last_ga_res"] = opt.optimize()
                                st.success("Optimization search completed successfully.")
                            except Exception as e:
                                st.error(f"GA optimization failed: {str(e)}")

                    if "last_ga_res" in st.session_state:
                        ga_res = st.session_state["last_ga_res"]
                        ui.section_title("Genetic Optimization Results")
                        if ga_res["out_of_range"]:
                            ui.status_banner("Some recommended values are outside historical observed bounds. Predictions may be unreliable.", "warn")
                            for w in ga_res["warnings"]:
                                st.write(f"- {w}")

                        col_r1, col_r2, col_r3 = st.columns(3)
                        with col_r1:
                            val_b = f"{ga_res['baseline_prediction']}" if target_type != "regression" else f"{ga_res['baseline_prediction']:,.2f}"
                            ui.kpi_card("Baseline Value", val_b)
                        with col_r2:
                            val_o = f"{ga_res['optimized_prediction']}" if target_type != "regression" else f"{ga_res['optimized_prediction']:,.2f}"
                            ui.kpi_card("Optimized Value", val_o)
                        with col_r3:
                            imp = ga_res["predicted_improvement"]
                            if target_type == "regression":
                                val_i = f"{imp:+,.2f}"
                            else:
                                val_i = f"{imp*100:+.2f}% Probability"
                            ui.kpi_card("Expected Change", val_i, delta_dir="up" if imp > 0 else "down")

                        ui.section_title("Recommended Genetic Actions")
                        action_records = []
                        for name, val in ga_res["recommended_values"].items():
                            curr_val = float(baseline_row[name])
                            diff_val = val - curr_val
                            diff_pct = (diff_val / curr_val) * 100.0 if curr_val != 0 else 0.0
                            shap_note = "Medium influence"
                            if global_importance:
                                matched_shap = next((item for item in global_importance if item["feature"] == name), None)
                                if matched_shap:
                                    if matched_shap == global_importance[0]:
                                        shap_note = "Strongest model driver"
                                    elif matched_shap in global_importance[:3]:
                                        shap_note = "High model driver"
                            action_records.append({
                                "Controllable Variable": clean_name(name), "Baseline Value": f"{curr_val:,.2f}",
                                "Recommended Value": f"{val:,.2f}", "Absolute Change": f"{diff_val:+,.2f}",
                                "Percentage Change": f"{diff_pct:+.2f}%", "Sensitivity Note": shap_note,
                            })
                        st.dataframe(pd.DataFrame(action_records), width="stretch")
                        st.caption("Sensitivity Note reflects whether a variable is among the strongest SHAP-based model drivers.")

                    st.markdown("---")
                    with st.expander("Advanced: Reinforcement Learning (optional)", expanded=False):
                        st.caption("RL remains available for research/experimentation but requires no normal user configuration.")
                        timesteps = 500
                        run_rl = st.button("Train RL Agent", key="run_rl_advanced")

                    if run_rl:
                        with st.spinner("Training lightweight PPO policy agent..."):
                            from src.decision.environment import TwinOptimizationEnv
                            from src.decision.rl_agent import train_rl_agent
                            try:
                                env = TwinOptimizationEnv(
                                    model=model, preprocessor=preprocessor, baseline_row=baseline_row,
                                    controllable_vars=numeric_controllables, target_type=target_type,
                                    objective_mode=objective_mode, class_idx=class_idx, bounds_dict=bounds_dict,
                                )
                                rl_res = train_rl_agent(env, total_timesteps=timesteps)
                                st.session_state["last_rl_res"] = rl_res
                                if rl_res["status"] == "success":
                                    st.success("RL policy agent trained successfully.")
                                elif rl_res["status"] == "unavailable":
                                    st.warning("Stable-Baselines3 is not available in this environment.")
                                else:
                                    st.error(f"RL training failed: {rl_res.get('reason')}")
                            except ValueError as ve:
                                st.warning("RL optimization is unavailable for this dataset because no suitable controllable variables were detected.")
                            except Exception as e:
                                st.error(f"RL training exception: {str(e)}")

                    if "last_rl_res" in st.session_state:
                        rl_res = st.session_state["last_rl_res"]
                        if rl_res["status"] == "success":
                            ui.section_title("RL Optimized Policy Results")
                            col_rl1, col_rl2, col_rl3 = st.columns(3)
                            with col_rl1:
                                val_b = f"{rl_res['baseline_prediction']}" if target_type != "regression" else f"{rl_res['baseline_prediction']:,.2f}"
                                ui.kpi_card("Baseline Value", val_b)
                            with col_rl2:
                                val_o = f"{rl_res['optimized_prediction']}" if target_type != "regression" else f"{rl_res['optimized_prediction']:,.2f}"
                                ui.kpi_card("RL Policy Value", val_o)
                            with col_rl3:
                                imp = rl_res["predicted_improvement"]
                                val_i = f"{imp:+,.2f}" if target_type == "regression" else f"{imp*100:+.2f}% Probability"
                                ui.kpi_card("Expected Change", val_i, delta_dir="up" if imp > 0 else "down")

                            ui.section_title("Recommended RL Policy Actions")
                            rl_action_records = []
                            for name, val in rl_res["recommended_values"].items():
                                curr_val = float(baseline_row[name])
                                diff_val = val - curr_val
                                diff_pct = (diff_val / curr_val) * 100.0 if curr_val != 0 else 0.0
                                rl_action_records.append({
                                    "Controllable Variable": clean_name(name), "Baseline Value": f"{curr_val:,.2f}",
                                    "RL Policy Value": f"{val:,.2f}", "Absolute Change": f"{diff_val:+,.2f}", "Percentage Change": f"{diff_pct:+.2f}%",
                                })
                            st.dataframe(pd.DataFrame(rl_action_records), width="stretch")
                        elif rl_res["status"] == "unavailable":
                            st.warning(f"RL recommendation unavailable for this dataset/environment. Reason: {rl_res.get('reason')}")
                        else:
                            st.error(f"RL recommendation could not be compiled. Reason: {rl_res.get('reason')}")

                    st.markdown("---")
                    ui.section_title("Decision Recommendation Summary", "Compare Genetic Search and Reinforcement Learning recommendations side by side.")
                    comp_records = [{"Method": "Baseline Reference State", "Estimated Outcome": f"{y_base_pred}" if target_type != "regression" else f"{y_base_pred:,.2f}", "Improvement Margin": "0.00"}]

                    if "last_ga_res" in st.session_state:
                        ga_res = st.session_state["last_ga_res"]
                        ga_val = f"{ga_res['optimized_prediction']}" if target_type != "regression" else f"{ga_res['optimized_prediction']:,.2f}"
                        ga_imp = ga_res["predicted_improvement"]
                        comp_records.append({"Method": "Genetic Algorithm (model-based recommendation)", "Estimated Outcome": ga_val, "Improvement Margin": f"{ga_imp:+,.2f}" if target_type == "regression" else f"{ga_imp*100:+.2f}%"})

                    if "last_rl_res" in st.session_state:
                        rl_res = st.session_state["last_rl_res"]
                        if rl_res["status"] == "success":
                            rl_val = f"{rl_res['optimized_prediction']}" if target_type != "regression" else f"{rl_res['optimized_prediction']:,.2f}"
                            rl_imp = rl_res["predicted_improvement"]
                            comp_records.append({"Method": "Reinforcement Learning Policy (model-based recommendation)", "Estimated Outcome": rl_val, "Improvement Margin": f"{rl_imp:+,.2f}" if target_type == "regression" else f"{rl_imp*100:+.2f}%"})

                    st.dataframe(pd.DataFrame(comp_records), width="stretch")
                    st.caption("Safety note: recommendations are model-based estimations under scenario bounds, not guaranteed business outcomes. Review boundaries before implementation. Historical correlations do not prove causality.")

        # ================================================
        # AI ASSISTANT — compact reference-style chat
        # ================================================
        elif navigation_selection == "AI Assistant":
            api_key = os.environ.get("LLM_API_KEY") or os.environ.get("GEMINI_API_KEY")
            if not api_key:
                try:
                    api_key = st.secrets.get("LLM_API_KEY") or st.secrets.get("GEMINI_API_KEY")
                except Exception:
                    pass
            provider = os.environ.get("LLM_PROVIDER", "gemini")
            model_name = os.environ.get("LLM_MODEL", "gemini-3.6-flash")

            from src.genai.rag_engine import LocalVectorStore
            if "vector_store" not in st.session_state:
                st.session_state["vector_store"] = LocalVectorStore()
            vector_store = st.session_state["vector_store"]

            ai_left, ai_right = st.columns([2.1, 1.0], gap="medium")

            with ai_left:
                with ui.card("SynTwin AI Assistant", "Grounded in your current dataset and analysis."):
                    if "chat_history" not in st.session_state:
                        st.session_state["chat_history"] = []

                    if not st.session_state["chat_history"]:
                        st.markdown('<div class="ref-chat-preview"><div class="ref-chat-icon">✦</div><div><b>Ask SynTwin about your data</b><p>Try “summarize the dataset”, “what is a digital twin?”, or “what are the biggest risks?”</p></div></div>', unsafe_allow_html=True)

                    for msg in st.session_state["chat_history"][-8:]:
                        with st.chat_message(msg["role"]):
                            st.write(msg["content"])

                    st.markdown('<div class="ref-quick-label">QUICK QUESTIONS</div>', unsafe_allow_html=True)
                    q1, q2, q3 = st.columns(3)
                    quick_query = None
                    with q1:
                        if st.button("Summarize data", key="q_summary", width="stretch"):
                            quick_query = "Summarize the current situation"
                    with q2:
                        if st.button("Biggest risks", key="q_risks", width="stretch"):
                            quick_query = "What are the biggest risks?"
                    with q3:
                        if st.button("Explain drivers", key="q_drivers", width="stretch"):
                            quick_query = "What are the main prediction drivers?"

                    user_input = st.chat_input("Ask about your data, models, forecasts or decisions...")
                    query_to_run = user_input or quick_query

                    if st.session_state["chat_history"] and st.button("Clear chat", key="clear_chat_compact"):
                        st.session_state["chat_history"] = []
                        st.rerun()

            with ai_right:
                with ui.card("Live Context", "Only the information currently available to SynTwin."):
                    context_items = [
                        ("Dataset", True),
                        ("Diagnosis", True),
                        ("SHAP Drivers", any(k.startswith("shap_vals_") for k in st.session_state.keys())),
                        ("Forecast", any(k.startswith("forecast_res_") for k in st.session_state.keys())),
                        ("Digital Twin", "last_sim_res" in st.session_state),
                        ("Decision", "last_ga_res" in st.session_state or "last_rl_res" in st.session_state),
                    ]
                    for label, active in context_items:
                        cls = "active" if active else "inactive"
                        txt = "✓ Active" if active else "○ Not used"
                        st.markdown(f'<div class="ref-context"><span>{label}</span><b class="{cls}">{txt}</b></div>', unsafe_allow_html=True)

                    llm_active = False
                    if api_key and not st.session_state.get("llm_error_occurred", False):
                        llm_active = True
                    
                    if llm_active:
                        st.markdown(f'<div class="ref-ai-note" style="border-color: #194B37; background: #0F2C20;"><b style="color: {COLORS["good"]};">● Gemini LLM Active</b><br>Generating answers using {model_name}.</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="ref-ai-note" style="border-color: #5C2A20; background: #381F1A;"><b>● LLM unavailable</b><br>LLM unavailable — using local analysis.</div>', unsafe_allow_html=True)

                    if vector_store.get_chunks_count() > 0:
                        st.caption(f"RAG documents: {vector_store.get_chunks_count()} indexed chunks")

            if query_to_run:
                st.session_state["chat_history"].append({"role": "user", "content": query_to_run})
                is_conv = is_conversational(query_to_run)
                
                if is_conv:
                    context_str = ""
                    retrieved_chunks = []
                else:
                    from src.genai.context_builder import build_context, format_context_to_text
                    context_dict = build_context(df, profile, st_state=st.session_state)
                    context_str = format_context_to_text(context_dict)
                    retrieved_chunks = []
                    if vector_store.get_chunks_count() > 0:
                        retrieved_chunks = vector_store.retrieve(query_to_run, top_k=2)

                from src.genai.response_generator import generate_grounded_response
                try:
                    if api_key:
                        from src.genai.llm_client import LLMClient
                        client = LLMClient(api_key=api_key, provider=provider, model_name=model_name)
                        history_payload = [{"role": m["role"], "content": m["content"]} for m in st.session_state["chat_history"][:-1]]
                        with st.spinner("Generating answer..."):
                            res = generate_grounded_response(client, query_to_run, context_str, retrieved_chunks, chat_history=history_payload)
                        if res.get("status") == "success":
                            answer = res.get("response", "No response returned.")
                            st.session_state["llm_error_occurred"] = False
                        else:
                            if is_conv:
                                answer = local_conversational_response(query_to_run)
                            else:
                                st.session_state["llm_error_occurred"] = True
                                answer = local_assistant_answer(query_to_run, df, profile)
                    else:
                        if is_conv:
                            answer = local_conversational_response(query_to_run)
                        else:
                            answer = local_assistant_answer(query_to_run, df, profile)
                except Exception as e:
                    if is_conv:
                        answer = local_conversational_response(query_to_run)
                    else:
                        st.session_state["llm_error_occurred"] = True
                        answer = local_assistant_answer(query_to_run, df, profile)

                st.session_state["chat_history"].append({"role": "assistant", "content": answer, "sources": []})
                st.rerun()

    except Exception as e:
        st.error(f"Ingestion, profiling and inference failed: {str(e)}")
        with st.expander("Technical details"):
            st.exception(e)
