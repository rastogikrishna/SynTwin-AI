import streamlit as st
import plotly.graph_objects as go

# ==========================================================
# COLOR SYSTEM
# ==========================================================
COLORS = {
    # Theme Base
    "bg": "#0A0F0F",
    "surface": "#111817",
    "surface_alt": "#141F1D",
    "border": "#1E2827",
    "border_strong": "#2C3A38",
    "text": "#F5F7F6",
    "text_muted": "#8B9997",
    "text_faint": "#5E6D6B",
    
    # Brand Accents (Teal/Emerald)
    "accent": "#17B996",
    "accent_dark": "#118C72",
    "accent_soft": "#112320",
    "accent_soft_border": "#1E3A35",
    
    # Semantic Colors
    "good": "#3ED598",
    "good_bg": "#0F2C20",
    "good_border": "#194B37",
    
    "warn": "#E8593A",
    "warn_bg": "#381F1A",
    "warn_border": "#5C2A20",
    
    "bad": "#E8593A",
    "bad_bg": "#381F1A",
    "bad_border": "#5C2A20",
    
    "info": "#1FD9A8",
    "info_bg": "#0F2F28",
    "info_border": "#184E42",
    
    "purple": "#8B5CF6",
    "purple_bg": "#251B3B",
    "purple_border": "#4C1D95",
}

# Plotly Color Palette
CHART_SEQUENCE = ["#17B996", "#1FD9A8", "#3ED598", "#0ba5ec", "#f79009", "#E8593A", "#8B5CF6"]
CHART_POSITIVE = COLORS["good"]
CHART_NEGATIVE = COLORS["warn"]
CHART_NEUTRAL = COLORS["accent"]

CHART_ACCENT_SCALE = [
    [0.0, "#112320"],
    [0.2, "#133E35"],
    [0.5, "#157A64"],
    [0.8, "#17B996"],
    [1.0, "#1FD9A8"]
]

PLOTLY_THEME = {
    "paper_bgcolor": "rgba(0, 0, 0, 0)",
    "plot_bgcolor": "rgba(0, 0, 0, 0)",
    "font": {
        "family": "Inter, -apple-system, sans-serif",
        "color": COLORS["text"],
        "size": 12,
    },
    "xaxis": {
        "gridcolor": COLORS["border"],
        "linecolor": COLORS["border_strong"],
        "zerolinecolor": COLORS["border"],
        "showgrid": True,
    },
    "yaxis": {
        "gridcolor": COLORS["border"],
        "linecolor": COLORS["border_strong"],
        "zerolinecolor": COLORS["border"],
        "showgrid": True,
    },
}


def plotly_layout(fig, height=280, show_legend=True, title=None):
    """Style Plotly figures consistently according to the design system."""
    fig.update_layout(
        paper_bgcolor=PLOTLY_THEME["paper_bgcolor"],
        plot_bgcolor=PLOTLY_THEME["plot_bgcolor"],
        font=PLOTLY_THEME["font"],
        height=height,
        showlegend=show_legend,
        margin=dict(l=12, r=12, t=44 if title else 16, b=12),
        colorway=CHART_SEQUENCE,
    )
    if title:
        fig.update_layout(
            title={
                "text": f"<b>{title}</b>",
                "font": {"size": 14, "color": COLORS["text"]},
                "x": 0.01,
                "y": 0.98,
            }
        )
    fig.update_xaxes(
        gridcolor=PLOTLY_THEME["xaxis"]["gridcolor"],
        linecolor=PLOTLY_THEME["xaxis"]["linecolor"],
        zerolinecolor=PLOTLY_THEME["xaxis"]["zerolinecolor"],
        showgrid=True,
    )
    fig.update_yaxes(
        gridcolor=PLOTLY_THEME["yaxis"]["gridcolor"],
        linecolor=PLOTLY_THEME["yaxis"]["linecolor"],
        zerolinecolor=PLOTLY_THEME["yaxis"]["zerolinecolor"],
        showgrid=True,
    )


def inject_global_css():
    """Apply styling overrides to force Streamlit into the custom enterprise dark theme."""
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        /* Typography & Core Styles */
        html, body, [class*="css"] {{
            font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        }}

        .stApp {{
            background-color: {COLORS["bg"]};
            color: {COLORS["text"]};
        }}

        /* Clean Streamlit Layout Chrome */
        #MainMenu, footer {{ visibility: hidden; }}
        header[data-testid="stHeader"] {{ background: transparent; height: 0; }}
        .block-container {{ padding-top: 1rem; padding-bottom: 2rem; max-width: 1400px; }}

        h1, h2, h3, h4, h5, h6 {{
            color: {COLORS["text"]} !important;
            font-weight: 700 !important;
            letter-spacing: -0.01em;
        }}
        p, li, span, label, div {{
            color: {COLORS["text_muted"]};
        }}
        b, strong {{
            color: {COLORS["text"]};
        }}

        hr {{
            border-color: {COLORS["border"]};
            margin: 1.2rem 0;
        }}

        /* ---------------- Sidebar Navigation ---------------- */
        [data-testid="stSidebar"] {{
            background-color: {COLORS["bg"]};
            border-right: 1px solid {COLORS["border"]};
        }}
        [data-testid="stSidebar"] * {{
            color: {COLORS["text_muted"]};
        }}
        [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {{
            color: {COLORS["text"]} !important;
        }}
        [data-testid="stSidebar"] .block-container {{
            padding-top: 1.0rem;
        }}
        [data-testid="stSidebarNav"] {{
            display: none;
        }}

        /* Compact spacing in sidebar block container */
        [data-testid="stSidebar"] [data-testid="stVerticalBlock"] > div {{
            margin-bottom: 2px !important;
            padding-top: 0 !important;
            padding-bottom: 0 !important;
        }}

        /* Style Streamlit Buttons in the Sidebar to look like compact text links */
        [data-testid="stSidebar"] .stButton > button {{
            background-color: transparent !important;
            border: none !important;
            color: {COLORS["text_muted"]} !important;
            padding: 4px 8px !important;
            margin: 0 !important;
            width: 100% !important;
            text-align: left !important;
            justify-content: flex-start !important;
            font-size: 0.81rem !important;
            font-weight: 500 !important;
            border-radius: 4px !important;
            line-height: 1.35 !important;
            height: auto !important;
            min-height: unset !important;
            box-shadow: none !important;
            display: flex !important;
            align-items: center !important;
            transition: background-color 0.1s ease, color 0.1s ease, border-left 0.1s ease !important;
        }}

        [data-testid="stSidebar"] .stButton > button:hover {{
            background-color: {COLORS["surface"]} !important;
            color: {COLORS["text"]} !important;
        }}

        [data-testid="stSidebar"] .stButton > button[kind="primary"] {{
            background-color: {COLORS["accent_soft"]} !important;
            color: {COLORS["accent"]} !important;
            border-left: 3px solid {COLORS["accent"]} !important;
            border-radius: 0 4px 4px 0 !important;
            font-weight: 600 !important;
            padding-left: 8px !important;
        }}

        [data-testid="stSidebar"] .stButton > button:focus {{
            box-shadow: none !important;
            outline: none !important;
        }}

        /* Custom Brand Layout */
        .stw-brand {{
            display: flex;
            align-items: center;
            gap: 10px;
            margin-bottom: 4px;
        }}
        .stw-brand-mark {{
            width: 32px;
            height: 32px;
            border-radius: 8px;
            background: linear-gradient(135deg, {COLORS["accent"]}, {COLORS["info"]});
            display: flex;
            align-items: center;
            justify-content: center;
            font-weight: 800;
            color: {COLORS["bg"]};
            font-size: 15px;
            flex-shrink: 0;
        }}
        .stw-brand-name {{
            font-size: 1.15rem;
            font-weight: 800;
            color: {COLORS["text"]};
            line-height: 1.1;
        }}
        .stw-brand-sub {{
            font-size: 0.68rem;
            color: {COLORS["text_muted"]};
            letter-spacing: 0.05em;
            margin-top: 1px;
            font-weight: 600;
        }}

        .stw-nav-section-label {{
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.09em;
            color: {COLORS["text_faint"]};
            font-weight: 700;
            margin: 18px 0 6px 2px;
        }}

        [data-testid="stSidebar"] .stRadio [role="radiogroup"] {{
            gap: 1px;
        }}
        [data-testid="stSidebar"] .stRadio label {{
            padding: 7px 10px;
            border-radius: 6px;
            width: 100%;
            font-size: 0.85rem;
            font-weight: 500;
            color: {COLORS["text_muted"]} !important;
            transition: all 0.12s ease;
            background-color: transparent !important;
        }}
        [data-testid="stSidebar"] .stRadio label:hover {{
            background-color: {COLORS["surface"]} !important;
            color: {COLORS["text"]} !important;
        }}
        [data-testid="stSidebar"] .stRadio label p {{
            color: inherit !important;
            font-size: 0.85rem;
        }}
        [data-testid="stSidebar"] .stRadio input:checked + div {{
            color: {COLORS["accent"]} !important;
            font-weight: 600;
        }}
        [data-testid="stSidebar"] div[data-baseweb="radio"] > div:first-child {{
            display: none;
        }}

        /* Dynamic Status Connected Box */
        .stw-status-box {{
            background-color: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 8px;
            padding: 10px 12px;
            margin-top: 10px;
        }}
        .stw-status-label {{
            font-size: 0.65rem;
            text-transform: uppercase;
            letter-spacing: 0.06em;
            color: {COLORS["text_muted"]};
            font-weight: 700;
            margin-bottom: 6px;
        }}
        .stw-status-row {{
            display: flex;
            justify-content: space-between;
            font-size: 0.78rem;
            padding: 2.5px 0;
            color: {COLORS["text_muted"]};
        }}
        .stw-status-row b {{
            color: {COLORS["text"]};
            font-weight: 600;
        }}
        .stw-dot {{
            display: inline-block;
            width: 7px;
            height: 7px;
            border-radius: 50%;
            margin-right: 6px;
        }}
        .stw-dot-good {{
            background-color: {COLORS["good"]};
            box-shadow: 0 0 0 2px {COLORS["good_border"]};
        }}
        .stw-dot-warn {{
            background-color: {COLORS["warn"]};
            box-shadow: 0 0 0 2px {COLORS["warn_border"]};
        }}
        .stw-dot-bad {{
            background-color: {COLORS["bad"]};
            box-shadow: 0 0 0 2px {COLORS["bad_border"]};
        }}

        /* Top bar override */
        .stw-topbar {{
            display: flex;
            justify-content: space-between;
            align-items: flex-start;
            padding: 16px 20px;
            background: linear-gradient(135deg, {COLORS["surface"]} 0%, {COLORS["surface_alt"]} 100%);
            border-radius: 10px;
            margin-bottom: 20px;
            color: {COLORS["text"]};
            border: 1px solid {COLORS["border"]};
        }}
        .stw-topbar-eyebrow {{
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            color: {COLORS["accent"]};
            font-weight: 700;
        }}
        .stw-topbar-title {{
            font-size: 1.4rem;
            font-weight: 800;
            color: {COLORS["text"]};
            margin: 2px 0;
        }}
        .stw-topbar-sub {{
            font-size: 0.82rem;
            color: {COLORS["text_muted"]};
        }}
        .stw-topbar-meta {{
            text-align: right;
            font-size: 0.78rem;
            color: {COLORS["text_muted"]};
        }}
        .stw-topbar-meta b {{
            color: {COLORS["text"]};
            font-size: 0.9rem;
        }}

        /* ---------------- Cards ----------------
           ui.card() renders content inside st.container(border=True),
           which Streamlit wraps in a real DOM element carrying this
           test id — style that element directly so the border actually
           wraps the charts/metrics/tables rendered inside it. */
        [data-testid="stVerticalBlockBorderWrapper"] {{
            background-color: {COLORS["surface"]} !important;
            border: 1px solid {COLORS["border"]} !important;
            border-radius: 10px !important;
        }}
        [data-testid="stVerticalBlockBorderWrapper"] > div {{
            padding: 16px 18px;
        }}
        /* .stw-card is still used directly (single st.markdown call, e.g. the
           Data Profile schema table) where a raw HTML div safely opens and
           closes within one call — that case is fine as plain CSS. */
        .stw-card {{
            background-color: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 10px;
        }}
        .stw-card-title {{
            font-size: 0.95rem;
            font-weight: 700;
            color: {COLORS["text"]} !important;
            margin-bottom: 2px;
        }}
        .stw-card-desc {{
            font-size: 0.8rem;
            color: {COLORS["text_muted"]} !important;
            margin-bottom: 12px;
        }}

        /* KPI cards */
        .stw-kpi {{
            background-color: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 8px;
            padding: 10px 14px;
            display: flex;
            flex-direction: column;
            gap: 2px;
            min-height: 76px;
            transition: border-color 0.15s ease;
        }}
        .stw-kpi:hover {{
            border-color: {COLORS["border_strong"]};
        }}
        .stw-kpi-label {{
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 700;
            color: {COLORS["text_muted"]} !important;
        }}
        .stw-kpi-value {{
            font-size: 1.35rem;
            font-weight: 800;
            color: {COLORS["text"]} !important;
            line-height: 1.15;
        }}
        .stw-kpi-sub {{
            font-size: 0.74rem;
            color: {COLORS["text_faint"]} !important;
        }}
        .stw-kpi-delta {{
            font-size: 0.74rem;
            font-weight: 700;
        }}
        .stw-kpi-delta.up {{ color: {COLORS["good"]}; }}
        .stw-kpi-delta.down {{ color: {COLORS["warn"]}; }}
        .stw-kpi-delta.flat {{ color: {COLORS["text_muted"]}; }}

        /* Section headers */
        .stw-section-title {{
            font-size: 1.1rem;
            font-weight: 700;
            color: {COLORS["text"]} !important;
            margin: 0 0 2px 0;
        }}
        .stw-section-desc {{
            font-size: 0.82rem;
            color: {COLORS["text_muted"]};
            margin-bottom: 12px;
        }}
        .stw-eyebrow {{
            font-size: 0.68rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
            font-weight: 700;
            color: {COLORS["accent"]};
            margin-bottom: 3px;
        }}

        /* Custom Badges */
        .stw-badge {{
            display: inline-block;
            padding: 2.5px 8px;
            border-radius: 4px;
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.01em;
        }}
        .stw-badge-id {{ background: #1C2827; color: {COLORS["text_muted"]}; }}
        .stw-badge-target {{ background: {COLORS["good_bg"]}; color: {COLORS["good"]}; }}
        .stw-badge-date {{ background: {COLORS["purple_bg"]}; color: {COLORS["purple"]}; }}
        .stw-badge-numeric {{ background: {COLORS["info_bg"]}; color: {COLORS["info"]}; }}
        .stw-badge-categorical {{ background: #1E2827; color: {COLORS["text_muted"]}; }}
        .stw-badge-boolean {{ background: {COLORS["warn_bg"]}; color: {COLORS["warn"]}; }}
        .stw-badge-neutral {{ background: {COLORS["surface"]}; color: {COLORS["text_muted"]}; border: 1px solid {COLORS["border"]}; }}

        /* Insight cards */
        .stw-insight {{
            background-color: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-left: 3px solid {COLORS["accent"]};
            border-radius: 6px;
            padding: 10px 12px;
            margin-bottom: 8px;
        }}
        .stw-insight.pos {{ border-left-color: {COLORS["good"]}; }}
        .stw-insight.neg {{ border-left-color: {COLORS["warn"]}; }}
        .stw-insight.warn {{ border-left-color: {COLORS["warn"]}; }}
        .stw-insight.info {{ border-left-color: {COLORS["info"]}; }}
        .stw-insight-title {{
            font-weight: 700;
            font-size: 0.85rem;
            color: {COLORS["text"]};
            margin-bottom: 2px;
        }}
        .stw-insight-desc {{
            font-size: 0.8rem;
            color: {COLORS["text_muted"]};
            line-height: 1.4;
        }}

        /* Status alerts */
        .stw-banner {{
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 10px 14px;
            border-radius: 6px;
            font-weight: 600;
            font-size: 0.88rem;
            margin-bottom: 14px;
        }}
        .stw-banner.good {{ background: {COLORS["good_bg"]}; color: {COLORS["good"]}; border: 1px solid {COLORS["good_border"]}; }}
        .stw-banner.warn {{ background: {COLORS["warn_bg"]}; color: {COLORS["warn"]}; border: 1px solid {COLORS["warn_border"]}; }}
        .stw-banner.bad {{ background: {COLORS["bad_bg"]}; color: {COLORS["bad"]}; border: 1px solid {COLORS["bad_border"]}; }}
        .stw-banner.info {{ background: {COLORS["info_bg"]}; color: {COLORS["info"]}; border: 1px solid {COLORS["info_border"]}; }}

        /* Empty state */
        .stw-empty {{
            text-align: center;
            padding: 48px 20px;
            border: 1.5px dashed {COLORS["border"]};
            border-radius: 12px;
            background-color: {COLORS["surface"]};
            margin-top: 16px;
        }}
        .stw-empty h2 {{ margin-bottom: 4px; font-size: 1.2rem; }}
        .stw-empty p {{ color: {COLORS["text_muted"]}; font-size: 0.85rem; }}

        .stw-notice {{
            background: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 8px;
            padding: 10px 14px;
            font-size: 0.82rem;
            color: {COLORS["text_muted"]};
        }}

        .stw-pill {{
            display: inline-flex;
            align-items: center;
            gap: 6px;
            padding: 3.5px 10px;
            border-radius: 999px;
            font-size: 0.75rem;
            font-weight: 600;
            background: {COLORS["accent_soft"]};
            color: {COLORS["accent"]};
            border: 1px solid {COLORS["accent_soft_border"]};
        }}

        .stw-caption-muted {{
            font-size: 0.75rem;
            color: {COLORS["text_faint"]};
        }}

        /* Buttons & Native Form Element Overrides */
        .stButton > button, .stDownloadButton > button {{
            background-color: {COLORS["surface"]} !important;
            color: {COLORS["text"]} !important;
            border: 1px solid {COLORS["border"]} !important;
            border-radius: 6px !important;
            font-weight: 600 !important;
            font-size: 0.85rem !important;
            transition: all 0.12s ease !important;
            padding: 6px 12px !important;
        }}
        .stButton > button:hover, .stDownloadButton > button:hover {{
            border-color: {COLORS["accent"]} !important;
            color: {COLORS["accent"]} !important;
        }}
        .stButton > button[kind="primary"] {{
            background-color: {COLORS["accent"]} !important;
            border-color: {COLORS["accent"]} !important;
            color: {COLORS["bg"]} !important;
        }}
        .stButton > button[kind="primary"]:hover {{
            background-color: {COLORS["info"]} !important;
            border-color: {COLORS["info"]} !important;
            color: {COLORS["bg"]} !important;
        }}

        /* Inputs, Selectboxes, textareas (eliminating raw white fields) */
        div[data-baseweb="select"] > div {{
            background-color: {COLORS["surface"]} !important;
            border-color: {COLORS["border"]} !important;
            color: {COLORS["text"]} !important;
            border-radius: 6px !important;
        }}
        div[data-baseweb="select"] * {{
            color: {COLORS["text"]} !important;
        }}
        input[type="number"], input[type="text"], textarea {{
            background-color: {COLORS["surface"]} !important;
            border: 1px solid {COLORS["border"]} !important;
            color: {COLORS["text"]} !important;
            border-radius: 6px !important;
            padding: 7px 10px !important;
        }}
        div[data-baseweb="input"] {{
            background-color: {COLORS["surface"]} !important;
            border-radius: 6px !important;
        }}
        [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input {{
            color: {COLORS["text"]} !important;
        }}

        /* Sliders */
        .stSlider > div [data-testid="stSliderTickBar"] {{
            color: {COLORS["text_muted"]};
        }}
        div[role="slider"] {{
            background-color: {COLORS["accent"]} !important;
            width: 14px !important;
            height: 14px !important;
        }}
        div[data-testid="stSliderTrack"] > div > div {{
            background: {COLORS["accent"]} !important;
        }}

        /* Tabs */
        .stTabs [data-baseweb="tab-list"] {{
            gap: 2px;
            border-bottom: 1px solid {COLORS["border"]};
        }}
        .stTabs [data-baseweb="tab"] {{
            font-weight: 600;
            font-size: 0.85rem;
            color: {COLORS["text_muted"]} !important;
            padding: 8px 12px;
            background-color: transparent !important;
        }}
        .stTabs [aria-selected="true"] {{
            color: {COLORS["accent"]} !important;
            border-bottom-color: {COLORS["accent"]} !important;
            font-weight: 700;
        }}

        /* Dataframes & Tables */
        [data-testid="stDataFrame"] {{
            border: 1px solid {COLORS["border"]} !important;
            border-radius: 8px !important;
            overflow: hidden !important;
        }}
        div[data-testid="stTable"] {{
            border: 1px solid {COLORS["border"]};
            border-radius: 8px;
            overflow: hidden;
            background-color: {COLORS["surface"]};
        }}
        div[data-testid="stTable"] table {{
            border-collapse: collapse;
            width: 100%;
        }}
        div[data-testid="stTable"] th {{
            background-color: {COLORS["bg"]} !important;
            color: {COLORS["text"]} !important;
            font-weight: 700 !important;
            font-size: 0.8rem !important;
            text-transform: uppercase !important;
            letter-spacing: 0.05em !important;
            border-bottom: 1px solid {COLORS["border"]} !important;
            padding: 8px 12px !important;
            text-align: left !important;
        }}
        div[data-testid="stTable"] td {{
            color: {COLORS["text"]} !important;
            font-size: 0.82rem !important;
            border-bottom: 1px solid {COLORS["border"]} !important;
            padding: 8px 12px !important;
        }}

        /* Metrics */
        [data-testid="stMetric"] {{
            background-color: {COLORS["surface"]};
            border: 1px solid {COLORS["border"]};
            border-radius: 8px;
            padding: 10px 12px;
        }}
        [data-testid="stMetricLabel"] {{
            font-size: 0.72rem;
            color: {COLORS["text_muted"]};
            text-transform: uppercase;
            letter-spacing: 0.04em;
        }}
        [data-testid="stMetricValue"] {{
            font-weight: 700;
            color: {COLORS["text"]};
        }}

        /* File uploader */
        [data-testid="stFileUploaderDropzone"] {{
            border-radius: 8px;
            border: 1.5px dashed {COLORS["border"]};
            background-color: {COLORS["surface"]};
            padding: 24px !important;
        }}
        [data-testid="stFileUploaderDropzone"] * {{
            color: {COLORS["text_muted"]} !important;
        }}
        [data-testid="stFileUploaderDropzone"] button {{
            background-color: {COLORS["bg"]} !important;
            border: 1px solid {COLORS["border"]} !important;
        }}

        /* Chat Workspace overrides */
        [data-testid="stChatMessage"] {{
            background-color: {COLORS["surface"]} !important;
            border: 1px solid {COLORS["border"]} !important;
            border-radius: 8px !important;
            padding: 14px 16px !important;
            margin-bottom: 10px !important;
        }}
        [data-testid="stChatMessage"] * {{
            color: {COLORS["text"]} !important;
        }}
        [data-testid="stChatMessage"] p {{
            color: {COLORS["text"]} !important;
            font-size: 0.88rem !important;
            line-height: 1.45 !important;
        }}
        [data-testid="stChatMessageAvatar"] {{
            background-color: {COLORS["bg"]} !important;
            border: 1px solid {COLORS["border"]} !important;
        }}


        /* =========================================================
           SynTwin AI — Reference Dashboard UI
           Visual layer only. No backend/data behavior is changed.
           ========================================================= */
        .ref-brand {{
            display:flex;
            align-items:center;
            gap:11px;
            padding:2px 0 4px;
        }}
        .ref-brand-mark {{
            width:34px;
            height:34px;
            border-radius:9px;
            display:flex;
            align-items:center;
            justify-content:center;
            background:linear-gradient(135deg,#17B996,#0D3B34);
            color:#06110F;
            font-weight:800;
            box-shadow:0 0 20px rgba(23,185,150,.16);
        }}
        .ref-brand-name {{
            color:#F5F7F6;
            font-size:1.35rem;
            font-weight:750;
            letter-spacing:-.025em;
            line-height:1;
        }}
        .ref-brand-sub {{
            color:#8B9997;
            font-size:.64rem;
            font-weight:600;
            letter-spacing:.09em;
            margin-top:5px;
        }}
        .ref-header-rule {{
            height:1px;
            background:#1E2827;
            margin:10px 0 12px;
        }}
        .ref-hero {{
            position:relative;
            overflow:hidden;
            min-height:178px;
            padding:26px 28px;
            border:1px solid #1E2827;
            border-radius:16px;
            background:
              radial-gradient(circle at 70% 45%,rgba(23,185,150,.16),transparent 28%),
              linear-gradient(135deg,#0D1918,#0A0F0F 62%);
            margin-bottom:14px;
        }}
        .ref-hero:after {{
            content:"";
            position:absolute;
            right:-65px;
            top:-115px;
            width:440px;
            height:440px;
            border:1px solid rgba(31,217,168,.28);
            border-radius:50%;
            box-shadow:
              0 0 0 32px rgba(31,217,168,.05),
              0 0 0 64px rgba(31,217,168,.035),
              0 0 0 96px rgba(31,217,168,.025);
            transform:rotate(-18deg);
            pointer-events:none;
        }}
        .ref-eyebrow {{
            color:#17B996;
            font-size:.67rem;
            font-weight:750;
            letter-spacing:.12em;
            margin-bottom:10px;
        }}
        .ref-title {{
            color:#F5F7F6;
            font-size:2rem;
            font-weight:700;
            letter-spacing:-.03em;
        }}
        .ref-subtitle {{
            color:#8B9997;
            font-size:.82rem;
            margin-top:8px;
        }}
        .ref-health {{
            position:absolute;
            z-index:1;
            right:28px;
            bottom:27px;
            width:205px;
        }}
        .ref-health-label {{
            color:#8B9997;
            font-size:.68rem;
            margin-bottom:6px;
        }}
        .ref-health-row {{
            display:flex;
            gap:7px;
            align-items:center;
            color:#F5F7F6;
            font-size:.82rem;
            margin-bottom:8px;
        }}
        .ref-dot {{
            width:7px;
            height:7px;
            border-radius:50%;
            display:inline-block;
            box-shadow:0 0 8px currentColor;
        }}
        .ref-dot.good {{ color:#3ED598; background:#3ED598; }}
        .ref-dot.warn {{ color:#E0A93E; background:#E0A93E; }}
        .ref-dot.bad {{ color:#E8593A; background:#E8593A; }}
        .ref-segments {{
            display:flex;
            gap:4px;
        }}
        .ref-segments span {{
            flex:1;
            height:5px;
            border-radius:3px;
            background:#22302E;
        }}
        .ref-segments span.filled {{ background:#3ED598; }}
        .ref-list-row {{
            display:flex;
            justify-content:space-between;
            align-items:center;
            gap:10px;
            padding:10px 0;
            border-bottom:1px solid #1E2827;
        }}
        .ref-list-row:last-child {{ border-bottom:none; }}
        .ref-list-row b {{
            display:block;
            color:#F5F7F6;
            font-size:.78rem;
        }}
        .ref-list-row small {{
            display:block;
            color:#5E6D6B;
            font-size:.66rem;
            margin-top:3px;
        }}
        .ref-badge {{
            color:#E8593A;
            background:rgba(232,89,58,.11);
            border:1px solid rgba(232,89,58,.18);
            padding:4px 8px;
            border-radius:14px;
            font-size:.67rem;
            font-weight:700;
            white-space:nowrap;
        }}
        .ref-segment {{ margin:10px 0 13px; }}
        .ref-segment > div:first-child {{
            display:flex;
            justify-content:space-between;
            gap:8px;
            font-size:.73rem;
            color:#F5F7F6;
            margin-bottom:5px;
        }}
        .ref-segment b {{ color:#8B9997; }}
        .ref-track {{
            height:7px;
            background:#182220;
            border-radius:4px;
            overflow:hidden;
        }}
        .ref-fill {{
            height:100%;
            background:linear-gradient(90deg,#0D3B34,#17B996);
            border-radius:4px;
        }}
        .ref-context {{
            display:flex;
            justify-content:space-between;
            align-items:center;
            padding:8px 0;
            border-bottom:1px solid #1E2827;
            font-size:.73rem;
        }}
        .ref-context span {{ color:#F5F7F6; }}
        .ref-context b {{ font-size:.65rem; }}
        .ref-context b.active {{ color:#3ED598; }}
        .ref-context b.inactive {{ color:#5E6D6B; }}
        .ref-chat-preview {{
            display:flex;
            gap:10px;
            padding:14px 0 8px;
            color:#F5F7F6;
        }}
        .ref-chat-icon {{
            width:27px;
            height:27px;
            border-radius:8px;
            display:flex;
            align-items:center;
            justify-content:center;
            flex-shrink:0;
            color:#17B996;
            background:rgba(23,185,150,.1);
            border:1px solid rgba(23,185,150,.25);
            font-size:.7rem;
        }}
        .ref-chat-preview b {{ font-size:.76rem; color:#F5F7F6; }}
        .ref-chat-preview p {{
            font-size:.69rem;
            color:#8B9997;
            line-height:1.45;
            margin:4px 0 0;
        }}
        .ref-flow {{
            color:#5E6D6B;
            font-size:.68rem;
            font-weight:700;
            letter-spacing:.05em;
            margin:8px 0 16px;
        }}
        .ref-flow span {{ color:#17B996; padding:0 5px; }}

        /* Horizontal navigation buttons closely follow the supplied reference. */
        .block-container > div > div > div > div .stButton > button {{
            border-radius:10px !important;
            border:1px solid #1E2827 !important;
            background:#111817 !important;
            color:#8B9997 !important;
            min-height:38px !important;
            padding:7px 10px !important;
            font-size:.73rem !important;
            font-weight:600 !important;
            box-shadow:none !important;
            transition:all .15s ease !important;
        }}
        .block-container > div > div > div > div .stButton > button:hover {{
            border-color:#2C3A38 !important;
            color:#F5F7F6 !important;
            background:#141F1D !important;
        }}
        .block-container > div > div > div > div .stButton > button[kind="primary"] {{
            background:#112320 !important;
            border-color:#17B996 !important;
            color:#F5F7F6 !important;
            box-shadow:0 0 0 1px rgba(23,185,150,.18) inset !important;
        }}
        /* Keep action buttons prominent. */
        button[kind="primary"] {{
            background:#17B996 !important;
            color:#06110F !important;
        }}
        [data-testid="stFileUploaderDropzone"] {{
            min-height:48px !important;
            padding:7px 12px !important;
        }}
        [data-testid="stFileUploaderDropzone"] small {{
            color:#5E6D6B !important;
        }}
        @media (max-width: 980px) {{
            .ref-health {{ position:static; margin-top:24px; width:100%; }}
            .ref-hero {{ min-height:235px; }}
            .ref-hero:after {{ opacity:.35; }}
        }}

        .ref-status {{
            display:flex;
            align-items:center;
            justify-content:flex-end;
            gap:7px;
            color:#8B9997;
            font-size:.68rem;
            padding:4px 0 10px;
        }}
        .ref-status b {{ color:#F5F7F6; font-weight:650; }}
        .ref-status-meta {{ color:#5E6D6B; margin-left:4px; }}
        .ref-status-dot {{
            width:7px;
            height:7px;
            border-radius:50%;
            background:#5E6D6B;
            display:inline-block;
            box-shadow:0 0 7px currentColor;
        }}
        @media (max-width: 760px) {{
            .ref-status {{ justify-content:flex-start; flex-wrap:wrap; }}
            .ref-title {{ font-size:1.6rem; }}
        }}

        /* Final reference-style cleanup: minimal navigation + compact assistant */
        [data-testid="stSidebar"] {{ display:none !important; }}
        .ref-quick-label {{ color:#5E6D6B; font-size:.64rem; font-weight:700; letter-spacing:.08em; margin:14px 0 7px; }}
        .ref-ai-note {{ margin-top:14px; padding:10px 11px; border:1px solid #1E2827; border-radius:10px; background:#0D1413; color:#8B9997; font-size:.68rem; line-height:1.45; }}
        .ref-ai-note b {{ color:#F5F7F6; }}
        .ref-context {{ padding:9px 0 !important; }}
        .ref-context b.active {{ color:#3ED598 !important; }}
        .ref-context b.inactive {{ color:#5E6D6B !important; }}
        div[data-testid="stChatInput"] {{ border-color:#1E2827 !important; }}
        div[data-testid="stChatInput"] textarea {{ background:#0D1413 !important; color:#F5F7F6 !important; }}
        </style>
        """,
        unsafe_allow_html=True,
    )
