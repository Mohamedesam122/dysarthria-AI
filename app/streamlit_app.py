"""Dysarthria AI: Streamlit front end (Person 6).

Run from the repository root:
    streamlit run app/streamlit_app.py

Rules from the project documentation that this UI enforces:
  * Mock output is always labelled and never looks like a real prediction.
  * XAI is shown only when explanation.available is true; otherwise we say why not.
  * Confidence is shown only if the backend provides it.
  * The "not a medical diagnosis" disclaimer is always visible.
  * Users never see stack traces or secrets.
"""
from __future__ import annotations

import hashlib
import html

import streamlit as st

from api_client import (
    ApiError,
    DEFAULT_DISCLAIMER,
    check_health,
    load_settings,
    predict,
    validate_upload,
    with_mock,
)

st.set_page_config(page_title="Dysarthria AI", page_icon="🗣️", layout="centered")

# Atkinson Hyperlegible was designed for low-vision readers; it suits an
# accessibility-focused tool. Falls back to the system sans-serif if blocked.
st.markdown(
    """
<style>
@@import url('https://fonts.googleapis.com/css2?family=Atkinson+Hyperlegible:wght@400;700&display=swap');
html, body, p, h1, h2, h3, label, li, .stMarkdown, button { font-family: 'Atkinson Hyperlegible', system-ui, sans-serif; }

/* خلفية الصفحة: تدرج هادي من فوق */
.stApp { background: linear-gradient(180deg, #DCEBEE 0%, #F3F6F7 340px); }
.block-container { max-width: 760px; padding-top: 2rem; }

/* العناوين */
h1 { font-size: 2.8rem !important; color: #174E55; letter-spacing: -0.5px; }
h2, h3 { color: #174E55; }

/* صندوق رفع الملف */
[data-testid="stFileUploader"] section {
  background: #FFFFFF; border: 2px dashed #1F6F78; border-radius: 16px; padding: 1.2rem;
}
[data-testid="stFileUploader"] section:hover { background: #F0F8F9; }

/* زرار Predict */
.stButton > button[kind="primary"] {
  border-radius: 12px; padding: .6rem 1.8rem; font-weight: 700;
  box-shadow: 0 4px 14px rgba(31,111,120,.3); transition: transform .1s;
}
.stButton > button[kind="primary"]:hover { transform: translateY(-2px); }

/* الشريط الجانبي */
[data-testid="stSidebar"] { border-right: 1px solid #C8D9DC; }

/* صندوق النتيجة: كارت بظل */
.transcript {
  border-left: 6px solid #1F6F78; background: #FFFFFF;
  padding: 1.2rem 1.5rem; margin: .5rem 0 1rem 0; border-radius: 0 14px 14px 0;
  box-shadow: 0 6px 20px rgba(20,35,43,.08);
  font-size: 2rem; line-height: 1.3; font-weight: 700; color: #14232B;
  overflow-wrap: anywhere;
}
.transcript.mock { border-left-color: #B7791F; border-left-style: dashed; color: #5B4A1E; }

/* العلامات الصغيرة */
.tag { display: inline-block; padding: .2rem .7rem; border-radius: 999px; font-size: .85rem; font-weight: 700; }
.tag.mock { background: #FBEBC8; color: #7A5410; }
.tag.model { background: #DCEBEE; color: #174E55; }

/* إخفاء قايمة Deploy */
[data-testid="stToolbar"] { display: none; }
[data-testid="stHeader"] { background: transparent; }

:focus-visible { outline: 3px solid #1F6F78 !important; outline-offset: 2px; }
</style>
""",
    unsafe_allow_html=True,
)

settings = load_settings()

# ------------------------------ sidebar ------------------------------------ #
with st.sidebar:
    st.subheader("Connection")
    demo = st.toggle(
        "Demo mode",
        value=settings.mock,
        help="Uses placeholder results so you can test the interface without the backend.",
    )
    settings = with_mock(settings, demo)
    if demo:
        st.warning("Demo mode is on. Results are placeholders, not model predictions.")
    else:
        st.caption(f"Service: {settings.api_url}")
        if st.button("Test connection"):
            ok, msg = check_health(settings)
            (st.success if ok else st.error)(msg)
    st.divider()
    st.caption(
        "Research prototype. Accepts WAV files up to "
        f"{settings.max_upload_mb:g} MB and {settings.max_duration_seconds:g} seconds."
    )

# ------------------------------- header ------------------------------------ #
st.title("Dysarthria AI")
st.write(
    "Upload a recording and the system predicts the words the speaker is "
    "trying to say."
)
if settings.mock:
    st.markdown(
        '<span class="tag mock">Demo mode: placeholder results</span>',
        unsafe_allow_html=True,
    )

# ------------------------------- upload ------------------------------------ #
uploaded = st.file_uploader("Audio recording (.wav)", type=["wav"])

if uploaded is None:
    # Empty state: tell the person what to do next.
    st.info("Choose a WAV recording to begin. You can listen to it before analyzing.")
    st.caption(DEFAULT_DISCLAIMER)
    st.stop()

data = uploaded.getvalue()
problem = validate_upload(uploaded.name, data, settings)
if problem:
    st.error(problem)  # empty / unsupported / too large / unreadable / too long
    st.stop()

st.audio(data, format="audio/wav")

# One result per (file, mode) so Streamlit reruns don't re-call the backend.
key = hashlib.sha256(data).hexdigest() + ("|mock" if settings.mock else "|live")

if st.button("Predict transcript", type="primary"):
    with st.spinner("Analyzing the recording…"):  # loading state
        try:
            st.session_state["outcome"] = {
                "key": key,
                "result": predict(uploaded.name, data, settings),
            }
        except ApiError as e:
            st.session_state["outcome"] = {"key": key, "error": str(e)}
        except Exception:  # never leak a stack trace to the user
            st.session_state["outcome"] = {
                "key": key,
                "error": "Something went wrong in the app. Try again, or tell the team if it keeps happening.",
            }

outcome = st.session_state.get("outcome")
if not outcome or outcome["key"] != key:
    st.caption(DEFAULT_DISCLAIMER)
    st.stop()

# ------------------------------- results ------------------------------------ #
if "error" in outcome:
    st.error(outcome["error"])
    st.caption(DEFAULT_DISCLAIMER)
    st.stop()

res = outcome["result"]
st.subheader("Predicted transcript")
cls = "transcript mock" if res["is_mock"] else "transcript"
# Escape so model output can never inject HTML.
safe_text = html.escape(res["transcript"]) or "(no words predicted)"
st.markdown(f'<div class="{cls}">{safe_text}</div>', unsafe_allow_html=True)

if res["is_mock"]:
    st.markdown(
        '<span class="tag mock">Mock output, not a real prediction</span>',
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        f'<span class="tag model">Model: {html.escape(res["model_version"])}</span>',
        unsafe_allow_html=True,
    )

for w in res["warnings"]:
    st.warning(w)

# Confidence: only when the backend provides a meaningful score.
if res["confidence"] is not None:
    kind = res["confidence_type"] or "score type not specified"
    st.caption(
        f"Model score: {float(res['confidence']):.2f} ({kind}). "
        "A high score does not guarantee the transcript is right."
    )

# Explanation: only when the backend says it is available.
st.subheader("Why this prediction")
expl = res["explanation"]
if expl["available"]:
    if expl["method"]:
        st.caption(f"Method: {expl['method']}")
    if expl["summary"]:
        st.write(expl["summary"])
    url = expl["visualization_url"]
    if url:
        if url.startswith("/"):
            url = settings.api_url + url  # relative URLs point at the backend
        st.image(url, caption="Regions of the recording that influenced the prediction")
    if expl["limitations"]:
        with st.expander("What this explanation can't tell you"):
            for item in expl["limitations"]:
                st.write(f"- {item}")
else:
    reason = expl["limitations"][0] if expl["limitations"] else "The service did not give a reason."
    st.info(f"No explanation is available for this prediction. {reason}")

st.divider()
st.caption(res["disclaimer"])
