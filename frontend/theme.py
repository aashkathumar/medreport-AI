"""Shared visual polish, applied identically on every page.

Streamlit's theme.toml handles colours; this covers what it can't (font
family, corner rounding, card shadows, the sidebar colour scheme) via a
small injected <style> block, plus the logo via the native st.logo() API.
Kept deliberately conservative, targets standard HTML elements
(button/input/[data-testid] attributes Streamlit itself sets, which are
stable across versions) rather than guessing at internal class names that
change between releases.
"""
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

_ASSETS = Path(__file__).resolve().parent / "assets"

# BUG FOUND: a blank line inside this string, between CSS rules, was being
# read by st.markdown's markdown-to-HTML pass as a paragraph break BEFORE the
# HTML/CSS ever reached the browser -- it split the <style> block into two
_CSS = (
    # CHANGED: Inter -> Plus Jakarta Sans, a warmer, slightly more
    # distinctive geometric sans than Inter's very generic/default-feeling
    # look, while staying just as readable for body text.
    '<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">'
    "<style>"
    "html, body, [class*=\"css\"] { font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif; }"
    # Buttons: soft rounded corners, matches the reference medical-app
    # screenshots' pill-style call-to-action buttons.
    ".stButton > button, .stDownloadButton > button { border-radius: 8px; font-weight: 500; }"
    # BUG FOUND: inputs had no visible border at rest, BaseWeb only draws
    # one on hover/focus, so an empty field looked like plain background.
    'div[data-baseweb="input"], div[data-baseweb="select"] > div { border-radius: 8px; '
    "border: 1px solid #E3D3AA !important; }"
    # BUG FOUND: "border: none" here did nothing, the real border lives on
    # the nested <details> element, not this div.
    'div[data-testid="stExpander"] { border-radius: 8px; '
    "background: #FFF6E8; border: none; box-shadow: none; }"
    'div[data-testid="stExpander"] details { border: none !important; }'
    # Metric tiles (the summary row at the top of Results): give them a
    # card-like container too, instead of sitting bare on the background.
    'div[data-testid="stMetric"] { background: #FFF6E8; border-radius: 10px; '
    "padding: 12px 16px; border: 1px solid #F5E2C0; }"
    # The "Press Enter to apply" hint on number inputs is absolutely
    # positioned by Streamlit, which lets it overlap the +/- stepper buttons
    # in a narrow column (seen in the manual-entry form's 3-column layout).
    'div[data-testid="InputInstructions"] { position: static; margin-top: 2px; }'
    # CHANGED: orange -> the pale yellow from the corner/background circles in
    # the reference image, on request.
    'section[data-testid="stSidebar"] { background: #F9D896; }'
    # CHANGED: white -> dark warm brown, following directly from the
    # background above being pale now instead of dark.
    'section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, '
    'section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] label, '
    'section[data-testid="stSidebar"] p { color: #4A3B1F !important; }'
    # CHANGED: Inter -> Plus Jakarta Sans -- a warmer, slightly more
    # distinctive geometric sans than Inter's very generic/default-feeling
    # look, while staying just as readable for body text.
    'section[data-testid="stSidebar"] label { font-size: 1.05rem; font-weight: 700; }'
    # Sidebar nav links (App/Results/History): same dark text, and a
    # translucent DARK overlay on hover/the active page now (a translucent
    # white overlay, correct on the old dark-orange sidebar, would barely show
    '[data-testid="stSidebarNav"] a { color: #4A3B1F !important; border-radius: 8px; }'
    '[data-testid="stSidebarNav"] a:hover, '
    '[data-testid="stSidebarNav"] a[aria-current="page"] { '
    "background: rgba(0, 0, 0, 0.08) !important; }"
    # CHANGED: Inter -> Plus Jakarta Sans -- a warmer, slightly more
    # distinctive geometric sans than Inter's very generic/default-feeling
    # look, while staying just as readable for body text.
    'section[data-testid="stSidebar"] div[data-baseweb="input"], '
    'section[data-testid="stSidebar"] div[data-baseweb="select"] > div { '
    "background: #FFF6E8 !important; border: none !important; }"
    'section[data-testid="stSidebar"] div[data-baseweb="base-input"] { background: transparent; }'
    'section[data-testid="stSidebar"] div[data-baseweb="input"] input, '
    'section[data-testid="stSidebar"] div[data-baseweb="select"] div, '
    'section[data-testid="stSidebar"] div[data-baseweb="select"] span, '
    'section[data-testid="stSidebar"] div[data-baseweb="select"] svg { '
    "color: #232946 !important; fill: #232946 !important; }"
    # BUG FOUND: a blank line inside this string, between CSS rules, was being
    # read by st.markdown's markdown-to-HTML pass as a paragraph break BEFORE
    # the HTML/CSS ever reached the browser -- it split the <style> block into
    'section[data-testid="stSidebar"] [data-testid="stNumberInputContainer"] { '
    "background: #FFF6E8 !important; }"
    # The "Age" +/- stepper buttons sit on that same cream field rather than
    # picking up any other rule, and their +/- glyphs match the dark navy
    # text colour.
    'section[data-testid="stSidebar"] button[data-testid="stNumberInput-StepDown"], '
    'section[data-testid="stSidebar"] button[data-testid="stNumberInput-StepUp"] { '
    "background: #FFF6E8 !important; }"
    'section[data-testid="stSidebar"] button[data-testid="stNumberInput-StepDown"] svg, '
    'section[data-testid="stSidebar"] button[data-testid="stNumberInput-StepUp"] svg { '
    "fill: #232946 !important; }"
    "</style>"
)

# BUG FOUND (recurring source of user confusion): every st.rerun() -- which
# fires on essentially every button click in this app ("Add test value",
# "Explain my results", the polling loop, etc.) -- resets the browser's scroll
_SCROLL_RESTORE_JS = """
<script>
(function() {
  const doc = window.parent.document;

  // BUG FOUND while verifying this live, round 1: document.scrollingElement
  // / documentElement never actually scrolls in current Streamlit, the
  // outer page is a fixed-height shell, and the real scrollable container
  // is the <section class="main"> inside [data-testid="stAppViewContainer"].
  // Found by directly measuring scrollHeight vs clientHeight on every
  // element, not guessed.
  function getScroller() {
    return doc.querySelector('[data-testid="stAppViewContainer"] section.main');
  }

  // Scoped per page path (Home/Results/History each have their own URL
  // under st.navigation), so switching to a genuinely different page still
  // starts at the top, only a rerun that stays ON the same page should
  // preserve position.
  function storageKey() {
    return "medreport_scroll_pos:" + doc.location.pathname;
  }

  function restoreScroll() {
    const scroller = getScroller();
    if (!scroller) return;
    const saved = sessionStorage.getItem(storageKey());
    if (saved !== null) {
      scroller.scrollTop = parseFloat(saved);
    }
    if (!scroller._medreportScrollBound) {
      scroller.addEventListener("scroll", function() {
        sessionStorage.setItem(storageKey(), scroller.scrollTop);
      }, { passive: true });
      scroller._medreportScrollBound = true;
    }
  }

  restoreScroll();

  // BUG FOUND while verifying this live, round 2: st.components.v1.html()
  // renders into an iframe whose content Streamlit does not reload on every
  // rerun (only on a genuine page load), so a plain one-shot script, even
  // with a short retry timeout, only ever restored scroll ONCE, on first
  // load. A real st.rerun() (clicking "Add test value", etc.) still reset
  // scroll to 0 afterwards, since nothing re-ran to catch that reset.
  // Verified directly: after a real Add-test-value click, scrollTop measured
  // 0 despite this fix being present, before this observer was added.
  // Fixed with a MutationObserver, registered ONCE (guarded so it survives
  // across whatever re-injections do happen) and left running for the life
  // of the tab, watching the app container for the DOM replacement every
  // rerun causes, restoring scroll (one frame later, so Streamlit's own
  // reset-to-0 has already happened and this runs after it, not before) on
  // every single rerun, not just the first page load.
  if (!window.parent._medreportScrollObserverAttached) {
    const target = doc.querySelector('[data-testid="stAppViewContainer"]') || doc.body;
    const observer = new MutationObserver(function() {
      requestAnimationFrame(restoreScroll);
    });
    observer.observe(target, { childList: true, subtree: true });
    window.parent._medreportScrollObserverAttached = true;
  }
})();
</script>
"""


def inject() -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    components.html(_SCROLL_RESTORE_JS, height=0)
    # CHANGED: Inter -> Plus Jakarta Sans -- a warmer, slightly more
    # distinctive geometric sans than Inter's very generic/default-feeling
    # look, while staying just as readable for body text.
    st.logo(
        str(_ASSETS / "logo_sidebar.svg"),
        icon_image=str(_ASSETS / "logo_icon.svg"),
    )
