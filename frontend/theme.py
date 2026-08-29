"""Shared visual polish, applied identically on every page.

Streamlit's theme.toml handles colours; this covers what it can't (font
family, corner rounding, card shadows) via a small injected <style> block.
Kept deliberately conservative -- targets standard HTML elements
(button/input/[data-testid] attributes Streamlit itself sets, which are
stable across versions) rather than guessing at internal class names that
change between releases.
"""
import streamlit as st

# BUG FOUND: a blank line inside this string, between CSS rules, was being
# read by st.markdown's markdown-to-HTML pass as a paragraph break BEFORE
# the HTML/CSS ever reached the browser -- it split the <style> block into
# two markdown "paragraphs", so only the first rule was actually treated as
# raw HTML; everything after the first blank line rendered as literal
# visible text on the page instead of being applied as CSS. Fixed by
# keeping the whole block as one continuous string with no blank lines.
_CSS = (
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">'
    "<style>"
    "html, body, [class*=\"css\"] { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }"
    # Buttons: soft rounded corners -- matches the reference medical-app
    # screenshots' pill-style call-to-action buttons.
    ".stButton > button, .stDownloadButton > button { border-radius: 8px; font-weight: 500; }"
    # Text/number inputs and selects: match the button rounding instead of
    # Streamlit's default sharper corners, for a more cohesive card feel.
    'div[data-baseweb="input"], div[data-baseweb="select"] > div { border-radius: 8px; }'
    # Expander cards (each test result): rounded + a soft shadow so they
    # read as distinct cards, matching the reference design's card layout.
    'div[data-testid="stExpander"] { border-radius: 10px; '
    "box-shadow: 0 1px 4px rgba(0, 0, 0, 0.06); border: 1px solid #E3E6FA; }"
    # Metric tiles (the summary row at the top of Results): give them a
    # card-like container too, instead of sitting bare on the background.
    'div[data-testid="stMetric"] { background: #F8F9FE; border-radius: 10px; '
    "padding: 12px 16px; border: 1px solid #E3E6FA; }"
    # The "Press Enter to apply" hint on number inputs is absolutely
    # positioned by Streamlit, which lets it overlap the +/- stepper
    # buttons in a narrow column (seen in the manual-entry form's 3-column
    # layout). Flowing it in-place below the input instead avoids the
    # collision everywhere the app uses a number_input.
    'div[data-testid="InputInstructions"] { position: static; margin-top: 2px; }'
    "</style>"
)


def inject() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
