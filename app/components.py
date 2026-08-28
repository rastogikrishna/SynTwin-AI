"""
SynTwin AI - Reusable UI Components
------------------------------------
Small rendering helpers used across every page so the product feels like a
single coherent system rather than a stack of ad-hoc Streamlit blocks.
"""

import html
from contextlib import contextmanager

import streamlit as st


def _esc(value) -> str:
    """HTML-escape any value that gets interpolated into raw markdown blocks."""
    return html.escape(str(value))


def page_header(eyebrow: str, title: str, description: str = ""):
    st.markdown(f'<div class="stw-eyebrow">{_esc(eyebrow)}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="stw-section-title" style="font-size:1.55rem;">{_esc(title)}</div>', unsafe_allow_html=True)
    if description:
        st.markdown(f'<div class="stw-section-desc">{_esc(description)}</div>', unsafe_allow_html=True)


def section_title(title: str, description: str = ""):
    st.markdown(f'<div class="stw-section-title">{_esc(title)}</div>', unsafe_allow_html=True)
    if description:
        st.markdown(f'<div class="stw-section-desc">{_esc(description)}</div>', unsafe_allow_html=True)


def kpi_card(label: str, value: str, sub: str = "", delta: str = "", delta_dir: str = "flat"):
    """Render a single KPI card. delta_dir in {'up','down','flat'}."""
    delta_html = f'<div class="stw-kpi-delta {delta_dir}">{_esc(delta)}</div>' if delta else ""
    sub_html = f'<div class="stw-kpi-sub">{_esc(sub)}</div>' if sub else ""
    st.markdown(
        f"""
        <div class="stw-kpi">
            <div class="stw-kpi-label">{_esc(label)}</div>
            <div class="stw-kpi-value">{_esc(value)}</div>
            {sub_html}
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(cards, columns=4):
    """cards: list of dicts with keys label,value,sub,delta,delta_dir"""
    if not cards:
        return
    cols = st.columns(columns)
    for i, card in enumerate(cards):
        with cols[i % columns]:
            kpi_card(
                card.get("label", ""),
                card.get("value", ""),
                card.get("sub", ""),
                card.get("delta", ""),
                card.get("delta_dir", "flat"),
            )


def badge(text: str, kind: str = "neutral") -> str:
    return f'<span class="stw-badge stw-badge-{kind}">{_esc(text)}</span>'


def insight_card(title: str, description: str, kind: str = "info"):
    """kind in {'pos','neg','warn','info'}"""
    st.markdown(
        f"""
        <div class="stw-insight {kind}">
            <div class="stw-insight-title">{_esc(title)}</div>
            <div class="stw-insight-desc">{description}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def status_banner(message: str, kind: str = "info", icon: str = ""):
    """kind in {'good','warn','bad','info'}"""
    default_icons = {"good": "✓", "warn": "⚠", "bad": "✕", "info": "ℹ"}
    icon = icon or default_icons.get(kind, "")
    st.markdown(
        f'<div class="stw-banner {kind}"><span>{icon}</span><span>{message}</span></div>',
        unsafe_allow_html=True,
    )


def empty_state(title: str, description: str):
    st.markdown(
        f"""
        <div class="stw-empty">
            <h2>{_esc(title)}</h2>
            <p>{_esc(description)}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def notice(text: str):
    st.markdown(f'<div class="stw-notice">{text}</div>', unsafe_allow_html=True)


def pill(text: str):
    st.markdown(f'<span class="stw-pill">{_esc(text)}</span>', unsafe_allow_html=True)


@contextmanager
def card(title: str = "", desc: str = ""):
    """Bordered card container.

    Uses Streamlit's native ``st.container(border=True)`` so everything
    rendered inside — charts, metrics, dataframes, nested columns — is
    genuinely nested inside the card in the DOM. (An earlier version of
    this helper opened a `<div>` in one `st.markdown()` call and closed it
    in another; Streamlit renders every `st.*` call as its own sibling
    element, so that never actually wrapped the content in between — the
    border/title box rendered detached from the real content. This version
    fixes that.)

    Usage:
        with ui.card("Title", "Optional description"):
            st.plotly_chart(fig)
    """
    with st.container(border=True):
        if title:
            st.markdown(f'<div class="stw-card-title">{_esc(title)}</div>', unsafe_allow_html=True)
        if desc:
            st.markdown(f'<div class="stw-card-desc">{_esc(desc)}</div>', unsafe_allow_html=True)
        yield


def format_metric_value(value, is_regression: bool = True) -> str:
    if isinstance(value, (int, float)):
        return f"{value:,.2f}" if is_regression else f"{value}"
    return str(value)


def top_bar(title: str, subtitle: str, meta_items):
    """meta_items: list of (label, value) tuples shown on the right side."""
    meta_html = "".join(
        f'<div style="margin-bottom:4px;">{_esc(label)}: <b>{_esc(value)}</b></div>'
        for label, value in meta_items
    )
    st.markdown(
        f"""
        <div class="stw-topbar">
            <div>
                <div class="stw-topbar-eyebrow">SynTwin AI · Decision Intelligence Platform</div>
                <div class="stw-topbar-title">{_esc(title)}</div>
                <div class="stw-topbar-sub">{_esc(subtitle)}</div>
            </div>
            <div class="stw-topbar-meta">{meta_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
