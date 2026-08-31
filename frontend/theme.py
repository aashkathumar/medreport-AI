"""Shared visual polish, applied identically on every page.

Streamlit's theme.toml handles colours; this covers what it can't (font
family, corner rounding, card shadows, the sidebar colour scheme) via a
small injected <style> block, plus the logo via the native st.logo() API.
Kept deliberately conservative -- targets standard HTML elements
(button/input/[data-testid] attributes Streamlit itself sets, which are
stable across versions) rather than guessing at internal class names that
change between releases.
"""
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components

_ASSETS = Path(__file__).resolve().parent / "assets"

# BUG FOUND: a blank line inside this string, between CSS rules, was being
# read by st.markdown's markdown-to-HTML pass as a paragraph break BEFORE
# the HTML/CSS ever reached the browser -- it split the <style> block into
# two markdown "paragraphs", so only the first rule was actually treated as
# raw HTML; everything after the first blank line rendered as literal
# visible text on the page instead of being applied as CSS. Fixed by
# keeping the whole block as one continuous string with no blank lines.
_CSS = (
    # CHANGED: Inter -> Plus Jakarta Sans -- a warmer, slightly more
    # distinctive geometric sans than Inter's very generic/default-feeling
    # look, while staying just as readable for body text.
    '<link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap" rel="stylesheet">'
    "<style>"
    "html, body, [class*=\"css\"] { font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif; }"
    # Buttons: soft rounded corners -- matches the reference medical-app
    # screenshots' pill-style call-to-action buttons.
    ".stButton > button, .stDownloadButton > button { border-radius: 8px; font-weight: 500; }"
    # Text/number inputs and selects: match the button rounding instead of
    # Streamlit's default sharper corners, for a more cohesive card feel.
    # BUG FOUND: inputs had no visible border in their resting state --
    # Streamlit/BaseWeb only draws a border on hover/focus by default, so an
    # empty field looked like plain background rather than a fillable box
    # until the user happened to hover over it. Added a persistent, always-
    # visible border so the field reads as an input target immediately.
    'div[data-baseweb="input"], div[data-baseweb="select"] > div { border-radius: 8px; '
    "border: 1px solid #E3D3AA !important; }"
    # CHANGED: cream instead of lavender, to match the new gold/white
    # palette instead of the old purple one.
    # BUG FOUND: "border: none" on stExpander itself did nothing visible --
    # inspected the actual DOM and found the real border (Streamlit's own
    # default) lives on the nested <details> element, not the stExpander
    # div this rule was targeting. Added the actual element that needed it.
    'div[data-testid="stExpander"] { border-radius: 8px; '
    "background: #FFF6E8; border: none; box-shadow: none; }"
    'div[data-testid="stExpander"] details { border: none !important; }'
    # Metric tiles (the summary row at the top of Results): give them a
    # card-like container too, instead of sitting bare on the background.
    'div[data-testid="stMetric"] { background: #FFF6E8; border-radius: 10px; '
    "padding: 12px 16px; border: 1px solid #F5E2C0; }"
    # The "Press Enter to apply" hint on number inputs is absolutely
    # positioned by Streamlit, which lets it overlap the +/- stepper
    # buttons in a narrow column (seen in the manual-entry form's 3-column
    # layout). Flowing it in-place below the input instead avoids the
    # collision everywhere the app uses a number_input.
    'div[data-testid="InputInstructions"] { position: static; margin-top: 2px; }'
    # CHANGED: orange -> the pale yellow from the corner/background circles
    # in the reference image, on request. This is a much LIGHTER colour
    # than the sidebar has had so far, which is why the text-colour rule
    # right below also had to flip from white to dark -- white text that
    # worked on the previous dark-orange sidebar would be close to
    # unreadable on a pale yellow one.
    'section[data-testid="stSidebar"] { background: #F9D896; }'
    # CHANGED: white -> dark warm brown, following directly from the
    # background above being pale now instead of dark.
    'section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, '
    'section[data-testid="stSidebar"] h3, section[data-testid="stSidebar"] label, '
    'section[data-testid="stSidebar"] p { color: #4A3B1F !important; }'
    # CHANGED: the Age/Sex/Diet/Name field labels specifically, bigger and
    # bold, on request -- kept separate from the caption-style text below
    # them ("Your profile personalises...", "Noted only for this
    # session..."), which stays at its smaller, lighter weight.
    'section[data-testid="stSidebar"] label { font-size: 1.05rem; font-weight: 700; }'
    # Sidebar nav links (App/Results/History): same dark text, and a
    # translucent DARK overlay on hover/the active page now (a translucent
    # white overlay, correct on the old dark-orange sidebar, would barely
    # show up on this pale background).
    '[data-testid="stSidebarNav"] a { color: #4A3B1F !important; border-radius: 8px; }'
    '[data-testid="stSidebarNav"] a:hover, '
    '[data-testid="stSidebarNav"] a[aria-current="page"] { '
    "background: rgba(0, 0, 0, 0.08) !important; }"
    # CHANGED: Age/Sex/Diet/name fields switched to plain white with dark
    # navy text -- on this pale-yellow sidebar, a white field reads as a
    # clean, distinct "card" the way the reference image's white content
    # card sits on its own gold background, rather than needing a tinted
    # colour of its own to stand out.
    # CHANGED: white -> the same light cream used for the file-uploader
    # dropzone and the "Enter test values manually" expander (both
    # #FFF6E8), on request -- ties these fields into that same established
    # colour rather than standing out as a separate, plain-white style.
    'section[data-testid="stSidebar"] div[data-baseweb="input"], '
    'section[data-testid="stSidebar"] div[data-baseweb="select"] > div { '
    "background: #FFF6E8 !important; border: none !important; }"
    'section[data-testid="stSidebar"] div[data-baseweb="base-input"] { background: transparent; }'
    'section[data-testid="stSidebar"] div[data-baseweb="input"] input, '
    'section[data-testid="stSidebar"] div[data-baseweb="select"] div, '
    'section[data-testid="stSidebar"] div[data-baseweb="select"] span, '
    'section[data-testid="stSidebar"] div[data-baseweb="select"] svg { '
    "color: #232946 !important; fill: #232946 !important; }"
    # BUG FOUND: the visible diagonal "seam" on the Age field specifically
    # (the only sidebar field with +/- steppers) -- inspected the DOM and
    # found the outer number-input container clips its contents to rounded
    # corners, but the wrapper div around the +/- buttons themselves stayed
    # transparent, so the sidebar's own background showed through in the
    # gap between the cream text box and the (also cream, but separately
    # coloured) buttons, reading as a stray diagonal line at the seam.
    # Filling the whole container cream removes the gap entirely.
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
# "Explain my results", the polling loop, etc.) -- resets the browser's
# scroll position to the very top of the page. This is a genuine Streamlit
# platform limitation, not something st.session_state or app-code changes
# can fix directly: Streamlit's own frontend does this on every rerun,
# regardless of what Python code ran. Worked around at the browser level: a
# zero-height component injects JS into the PARENT document (Streamlit
# renders each component in a sandboxed iframe, so `window.parent` is
# needed to reach the actual page, not the iframe's own empty body) that
# saves scroll position to sessionStorage on every scroll event, and
# restores it immediately after each rerun re-renders the page. This is a
# client-side patch over a Streamlit limitation, not a perfect native fix --
# a brief top-then-restore flicker is possible, but it removes the "have to
# manually scroll back down after every click" problem this was written for.
_SCROLL_RESTORE_JS = """
<script>
(function() {
  const doc = window.parent.document;

  // BUG FOUND while verifying this live, round 1: document.scrollingElement
  // / documentElement never actually scrolls in current Streamlit -- the
  // outer page is a fixed-height shell, and the real scrollable container
  // is the <section class="main"> inside [data-testid="stAppViewContainer"].
  // Found by directly measuring scrollHeight vs clientHeight on every
  // element, not guessed.
  function getScroller() {
    return doc.querySelector('[data-testid="stAppViewContainer"] section.main');
  }

  // Scoped per page path (Home/Results/History each have their own URL
  // under st.navigation), so switching to a genuinely different page still
  // starts at the top -- only a rerun that stays ON the same page should
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
  // rerun (only on a genuine page load), so a plain one-shot script -- even
  // with a short retry timeout -- only ever restored scroll ONCE, on first
  // load. A real st.rerun() (clicking "Add test value", etc.) still reset
  // scroll to 0 afterwards, since nothing re-ran to catch that reset.
  // Verified directly: after a real Add-test-value click, scrollTop measured
  // 0 despite this fix being present, before this observer was added.
  // Fixed with a MutationObserver, registered ONCE (guarded so it survives
  // across whatever re-injections do happen) and left running for the life
  // of the tab, watching the app container for the DOM replacement every
  // rerun causes -- restoring scroll (one frame later, so Streamlit's own
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
    # CHANGED: the brand mark previously lived inside the sidebar's own
    # scrollable content, below the native App/Results/History page list --
    # Streamlit renders that list itself, at a fixed position, before any
    # user code runs, so there was no way to place ordinary sidebar content
    # above it. st.logo() (added in Streamlit 1.36, this project was on
    # 1.35 until this) is the actual, supported mechanism for exactly this:
    # it renders above the page list, not as part of the scrollable content
    # below it. Two different images because the same logo sits on two
    # different backgrounds -- white text for the purple sidebar, a
    # coloured mark for the main page's white top-left corner.
    st.logo(
        str(_ASSETS / "logo_sidebar.svg"),
        icon_image=str(_ASSETS / "logo_icon.svg"),
    )
