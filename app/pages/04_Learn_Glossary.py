#!/usr/bin/env python3
# app/pages/04_Resources.py
# Comments in English as requested.

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import streamlit as st

# Optional: nicer markdown -> HTML conversion inside cards
try:
    import markdown as mdlib  # type: ignore
except Exception:
    mdlib = None


# -----------------------------
# Page config
# -----------------------------
st.set_page_config(
    page_title="Resources",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Styling
# -----------------------------
st.markdown(
    """
<style>
/* App background */
.stApp { background: #f6f7fb; }

/* Ensure readable default text color even if Streamlit theme is "dark" */
.stApp, .stApp * { color: #111827; }

.block-container { padding-top: 2.6rem !important; }

h1, h2, h3, h4 { color: #111827 !important; letter-spacing: -0.02em; }

.page-title { font-size: 2.2rem; font-weight: 900; color: #0f172a; margin: 0 0 0.35rem 0; }
.page-subtitle { color: #6b7280; font-size: 1.02rem; font-weight: 600; margin: 0 0 1.0rem 0; }

.card {
  background: #ffffff;
  border: 1px solid #e5e7eb;
  border-radius: 18px;
  box-shadow: 0 6px 18px rgba(17, 24, 39, 0.06);
  padding: 18px 18px 16px 18px;
  color: #111827;
}

/* Make sure typical markdown text stays dark inside the card */
.card, .card * { color: #111827; }

/* Links should remain clearly link-like */
.card a, .stMarkdown a { color: #2563eb; text-decoration: none; }
.card a:hover, .stMarkdown a:hover { text-decoration: underline; }

.doc-title { font-size: 1.45rem; font-weight: 950; color: #0f172a; margin: 0 0 0.6rem 0; }
.doc-subtle { color: #6b7280; font-size: 0.95rem; font-weight: 650; margin: 0 0 0.9rem 0; line-height: 1.45; }

.card pre {
  background: #0b1020;
  color: #e5e7eb;
  padding: 12px 14px;
  border-radius: 14px;
  overflow-x: auto;
}
.card pre * { color: #e5e7eb; }

/* Inline code: improve contrast (fix for backtick-highlighted labels) */
.card code {
  background: #eef2ff;  /* light indigo tint */
  color: #0f172a;       /* dark text */
  border: 1px solid #c7d2fe;
  padding: 0.15rem 0.35rem;
  border-radius: 8px;
  font-weight: 700;
}

/* Also cover Streamlit markdown blocks outside .card */
.stMarkdown code {
  background: #eef2ff;
  color: #0f172a;
  border: 1px solid #c7d2fe;
  padding: 0.15rem 0.35rem;
  border-radius: 8px;
  font-weight: 700;
}

.card pre code { background: transparent; padding: 0; border: none; font-weight: 400; }

.card table {
  border-collapse: collapse;
  width: 100%;
  margin: 0.5rem 0 1rem 0;
}
.card table th, .card table td {
  border: 1px solid #e5e7eb;
  padding: 8px 10px;
  vertical-align: top;
}
.card table th { background: #f9fafb; font-weight: 900; }

hr.soft { border: none; border-top: 1px solid #e5e7eb; margin: 14px 0; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Paths / manifest
# -----------------------------
def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


LEARN_DIR = repo_root() / "content" / "learn"

# Explicit ordered manifest: (filename, display_title, subtitle)
DOCS: List[Tuple[str, str, str]] = [
    ("boglehead.md", "Boglehead Intro", ""),
    ("current_situation.md", "Current Situation (context for the indicators)", ""),
    ("actionables.md", "Bogle Portfolio Actionables", ""),
    ("daily_state_tilts_and_indicators.md", "Technical Framework, limitations and Extended Glossary", ""),
]


def _norm_key(name: str) -> str:
    """
    Normalizes a filename for forgiving matching:
    - lowercase
    - remove extension
    - replace non-alphanum with a single underscore
    """
    stem = Path(name).stem.lower()
    stem = re.sub(r"[^a-z0-9]+", "_", stem).strip("_")
    return stem


def resolve_doc_path(learn_dir: Path, expected_filename: str) -> Optional[Path]:
    """
    Resolve an expected filename against the folder, tolerant to:
    - case differences
    - '-' vs '_' differences
    - minor naming drift, as long as the normalized stem matches
    """
    direct = learn_dir / expected_filename
    if direct.exists() and direct.is_file():
        return direct

    # Build a normalized lookup over actual markdown files
    candidates = list(learn_dir.glob("*.md"))
    by_norm: Dict[str, Path] = {}
    for p in candidates:
        by_norm[_norm_key(p.name)] = p

    key = _norm_key(expected_filename)
    return by_norm.get(key)


def read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8", errors="replace")


def remove_first_h1(md: str) -> str:
    """
    Removes the first H1 (a line starting with '# ') found near the top to avoid duplicating titles.
    """
    lines = md.splitlines()
    for i, line in enumerate(lines[:60]):
        if line.strip().startswith("# "):
            return "\n".join(lines[:i] + lines[i + 1 :]).lstrip("\n")
    return md


def render_markdown_in_card(title: str, subtitle: str, md_text: str) -> None:
    """
    Renders content inside a styled card. Uses python-markdown if available for better HTML rendering.
    Falls back to Streamlit markdown if the library is not available.
    """
    md_clean = remove_first_h1(md_text)

    if mdlib is None:
        # Fallback: still no empty "card shells" (single render for the container),
        # but markdown body is rendered by Streamlit right after.
        st.markdown(
            f"""
<div class="card">
  <div class="doc-title">{title}</div>
  {f'<div class="doc-subtle">{subtitle}</div>' if subtitle else ''}
</div>
""",
            unsafe_allow_html=True,
        )
        st.markdown(md_clean)
        return

    body_html = mdlib.markdown(
        md_clean,
        extensions=[
            "fenced_code",
            "tables",
            "toc",
            "sane_lists",
        ],
        output_format="html5",
    )

    st.markdown(
        f"""
<div class="card">
  <div class="doc-title">{title}</div>
  {f'<div class="doc-subtle">{subtitle}</div>' if subtitle else ''}
  <hr class="soft"/>
  {body_html}
</div>
""",
        unsafe_allow_html=True,
    )


# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="page-title">Resources</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="page-subtitle">Glossary and reference notes used by the project.</div>',
    unsafe_allow_html=True,
)

# -----------------------------
# Resolve docs (show all, in order)
# -----------------------------
resolved: List[Tuple[str, str, str, Optional[Path]]] = []
for filename, title, subtitle in DOCS:
    p = resolve_doc_path(LEARN_DIR, filename)
    resolved.append((filename, title, subtitle, p))

missing = [fn for fn, _, _, p in resolved if p is None]
if missing:
    st.warning(
        "Some expected markdown files were not found in content/learn:\n\n- "
        + "\n- ".join(missing)
    )

# Build tabs only for files that exist, but keep ordering
tabs_spec = [(title, subtitle, p) for _, title, subtitle, p in resolved if p is not None]
if not tabs_spec:
    st.error(f"No markdown files found in: {LEARN_DIR}")
    st.stop()

tab_labels = [t for (t, _, _) in tabs_spec]
tabs = st.tabs(tab_labels)

for tab, (title, subtitle, path) in zip(tabs, tabs_spec):
    with tab:
        md_text = read_text(path)
        render_markdown_in_card(title=title, subtitle=subtitle, md_text=md_text)
