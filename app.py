"""Nayya — a source-grounded legal information assistant for women in India."""

import html
import os
import sys
import traceback

import streamlit as st

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Nayya | Understand your rights",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed",
)


def safe_import_rag():
    try:
        from query_database import EnhancedRAGPipeline

        return EnhancedRAGPipeline, None
    except Exception as exc:
        return None, f"{exc}\n\n{traceback.format_exc()}"


EnhancedRAGPipeline, import_error = safe_import_rag()

st.markdown(
    """
<style>
:root {
  --ink: #25101f;
  --plum: #4b193b;
  --berry: #842d51;
  --rose: #dba5aa;
  --ivory: #fbf6eb;
  --paper: #fffdf7;
  --gold: #d49a1d;
  --sage: #416b55;
  --line: #ddcfbf;
  --muted: #665b60;
}

html, body, [class*="css"] {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: var(--ink);
}
.stApp {
  background:
    radial-gradient(circle at 88% 10%, rgba(212,154,29,.12) 0 8rem, transparent 8.1rem),
    linear-gradient(110deg, rgba(132,45,81,.035), transparent 44%),
    var(--ivory);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppViewContainer"] > .main .block-container {
  max-width: 1180px;
  padding-top: 1.25rem;
  padding-bottom: 3rem;
}
#MainMenu, footer { visibility: hidden; }

.nayya-bar {
  display:flex; align-items:center; justify-content:space-between;
  padding:.75rem 0 1rem; border-bottom:1px solid var(--line);
}
.nayya-wordmark {
  font-family: Georgia, "Times New Roman", serif;
  font-size:1.55rem; font-weight:700; letter-spacing:-.03em; color:var(--plum);
}
.nayya-wordmark i { color:var(--gold); font-style:normal; margin-right:.38rem; }
.nayya-scope {
  color:var(--muted); font-size:.76rem; font-weight:650;
  letter-spacing:.08em; text-transform:uppercase;
}
.hero {
  position:relative; padding:2.25rem 0 1.45rem; max-width:930px;
}
.hero::after {
  content:""; position:absolute; width:110px; height:55px; right:0; top:1.75rem;
  border:2px solid var(--gold); border-bottom:0; border-radius:130px 130px 0 0;
  opacity:.65;
}
.eyebrow {
  color:var(--berry); text-transform:uppercase; letter-spacing:.13em;
  font-size:.72rem; font-weight:750; margin-bottom:1rem;
}
.hero h1 {
  font-family:Georgia, "Times New Roman", serif; color:var(--ink);
  font-size:clamp(2.55rem, 5.2vw, 4.25rem); line-height:1;
  max-width:820px; letter-spacing:-.055em; margin:0 0 1.35rem;
}
.hero p {
  max-width:680px; color:var(--muted); font-size:1.08rem; line-height:1.7; margin:0;
}
.safety {
  border-left:5px solid var(--gold); border-top:1px solid var(--line);
  border-bottom:1px solid var(--line); padding:1rem 1.15rem;
  background:rgba(255,253,247,.72); margin:.25rem 0 1.45rem;
  color:#3f3539; font-size:.92rem; line-height:1.55;
}
.safety strong { color:var(--ink); }
.section-label {
  color:var(--berry); text-transform:uppercase; letter-spacing:.12em;
  font-size:.7rem; font-weight:800; margin:0 0 .35rem;
}
.conversation-title {
  font-family:Georgia, "Times New Roman", serif; font-size:2rem;
  letter-spacing:-.025em; margin:0 0 .4rem; color:var(--ink);
}
.conversation-intro { color:var(--muted); margin:0 0 1.2rem; }

[data-testid="stChatMessage"] {
  background:var(--paper); border:1px solid var(--line); border-radius:3px;
  padding:1rem 1.1rem; margin:.75rem 0;
  box-shadow:0 8px 24px rgba(37,16,31,.045);
}
[data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
  background:#f4e6e6; border-color:#d8b9b7;
}
[data-testid="stChatMessage"] p { line-height:1.7; }
[data-testid="stChatInput"] {
  border:1.5px solid var(--plum); border-radius:3px;
  background:var(--paper);
}
[data-testid="stChatInput"]:focus-within {
  box-shadow:0 0 0 3px rgba(212,154,29,.28);
}

.stButton > button {
  border-radius:3px; min-height:2.85rem; border:1px solid #bba99b;
  background:var(--paper); color:var(--ink); font-weight:650;
  transition:transform .16s ease, border-color .16s ease, background .16s ease;
}
.stButton > button:hover {
  border-color:var(--berry); color:var(--plum); background:#fff9f2;
  transform:translateY(-1px);
}
.stButton > button:focus { box-shadow:0 0 0 3px rgba(212,154,29,.34) !important; }
.stButton > button[kind="primary"] {
  background:var(--plum); color:white; border-color:var(--plum);
}
.stButton > button[kind="primary"]:hover { background:#351128; color:white; }

.help-cluster, .trust-panel {
  margin-top:1.7rem; padding:1.25rem 1.35rem; border-top:3px solid var(--plum);
  background:var(--paper); box-shadow:0 8px 28px rgba(37,16,31,.05);
}
.trust-panel { border-top-color:var(--sage); }
.panel-title {
  font-family:Georgia, "Times New Roman", serif; color:var(--ink);
  font-size:1.28rem; margin:0 0 .75rem;
}
.help-list { display:grid; gap:.65rem; }
.help-item { border-bottom:1px solid var(--line); padding:0 0 .65rem; line-height:1.45; }
.help-item:last-child { border:0; padding-bottom:0; }
.help-item b { color:var(--berry); margin-right:.35rem; }
.trust-panel p { color:var(--muted); font-size:.88rem; line-height:1.6; margin:.55rem 0; }
.source-badge {
  display:inline-block; color:var(--sage); font-size:.72rem; font-weight:800;
  letter-spacing:.06em; text-transform:uppercase; margin-bottom:.35rem;
}
.answer-meta { color:var(--sage); font-size:.77rem; margin-top:.65rem; }
.unavailable {
  background:#f5e9e5; border:1px solid #d5b5ae; border-left:5px solid var(--berry);
  padding:1rem 1.1rem; margin:.85rem 0 1.15rem; color:#3d2731; line-height:1.55;
}
.unavailable strong { display:block; color:var(--ink); margin-bottom:.2rem; }
.site-footer {
  margin-top:3.5rem; padding-top:1.15rem; border-top:1px solid var(--line);
  color:var(--muted); font-size:.78rem; line-height:1.6;
}

@media (max-width: 900px) {
  [data-testid="stHorizontalBlock"] { flex-wrap:wrap; }
  [data-testid="column"] {
    flex:1 1 100% !important; width:100% !important; min-width:0 !important;
  }
}
@media (max-width: 700px) {
  [data-testid="stAppViewContainer"] > .main .block-container { padding: .75rem 1rem 2rem; }
  .nayya-scope { font-size:.62rem; letter-spacing:.04em; }
  .hero { padding:1.75rem 0 1.25rem; }
  .hero h1 { font-size:clamp(2.25rem, 11.5vw, 2.8rem); max-width:94%; margin-bottom:1rem; }
  .hero::after { width:62px; height:31px; right:0; top:1.4rem; }
  .hero p { font-size:.98rem; }
  .safety { margin-bottom:1.2rem; }
  [data-testid="stChatInput"] { width:100%; }
  .stButton > button { white-space:normal; height:auto; min-height:2.85rem; padding:.65rem .8rem; }
  [data-testid="stRadio"] > div { gap:.35rem 1rem; flex-wrap:wrap; }
  .help-cluster, .trust-panel { margin-top:1rem; }
}
@media (max-width: 380px) {
  .nayya-wordmark { font-size:1.35rem; }
  .nayya-scope { max-width:9.5rem; text-align:right; line-height:1.35; }
  .hero h1 { font-size:2.2rem; letter-spacing:-.045em; }
  .conversation-title { font-size:1.7rem; }
}
@media (prefers-reduced-motion: no-preference) {
  .hero, .safety { animation:rise .55s ease both; }
  .safety { animation-delay:.1s; }
  @keyframes rise { from { opacity:0; transform:translateY(9px); } to { opacity:1; transform:none; } }
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<header class="nayya-bar">
  <div class="nayya-wordmark"><i>✦</i>Nayya</div>
  <div class="nayya-scope">Legal information&nbsp; • &nbsp;India</div>
</header>
<section class="hero">
  <div class="eyebrow">Knowledge is a form of power</div>
  <h1>Understand your rights.<br>Choose your next step.</h1>
  <p>Plain-language information grounded in the Protection of Women from
  Domestic Violence Act, 2005—designed to help you find your footing.</p>
</section>
""",
    unsafe_allow_html=True,
)

verified_helpline = os.getenv("VERIFIED_WOMEN_HELPLINE", "").strip()
helpline_text = (
    f" A verified support line configured for this service is: <strong>{html.escape(verified_helpline)}</strong>."
    if verified_helpline
    else " A verified local helpline can be added by the service administrator."
)
st.markdown(
    f"""
<aside class="safety"><strong>Your safety comes first.</strong> If you are in immediate
danger, move to a safer place if you can and contact local emergency services.
{helpline_text}</aside>
""",
    unsafe_allow_html=True,
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "response_style" not in st.session_state:
    st.session_state.response_style = "Plain language"
if "pipeline_initialized" not in st.session_state:
    st.session_state.pipeline_initialized = False
if "pipeline_attempted" not in st.session_state:
    st.session_state.pipeline_attempted = False


@st.cache_resource
def initialize_pipeline():
    try:
        return EnhancedRAGPipeline(), None
    except Exception as exc:
        return None, f"{exc}\n\n{traceback.format_exc()}"


def process_question(question):
    if not getattr(st.session_state, "pipeline", None):
        return
    mode = "plain" if st.session_state.response_style == "Plain language" else "legal"
    try:
        with st.spinner("Looking through the source material…"):
            result = st.session_state.pipeline.query_with_sources(question, mode=mode)
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": result.get("answer", "I could not generate an answer."),
                "source_count": len(result.get("sources", [])),
                "sources": [
                    {
                        "page": source.metadata.get("page"),
                        "section": source.metadata.get("section"),
                        "excerpt": source.page_content[:500].strip(),
                    }
                    for source in result.get("sources", [])
                ],
            }
        )
    except Exception:
        st.session_state.chat_history.append(
            {
                "role": "assistant",
                "content": (
                    "I’m sorry—I couldn’t complete that search. Please try again later "
                    "or check with a qualified legal-aid professional."
                ),
                "source_count": 0,
            }
        )


service_error = import_error
if not import_error and not st.session_state.pipeline_initialized and not st.session_state.pipeline_attempted:
    st.session_state.pipeline_attempted = True
    pipeline, pipeline_error = initialize_pipeline()
    if pipeline_error:
        service_error = pipeline_error
        st.session_state.pipeline_error = pipeline_error
    else:
        st.session_state.pipeline = pipeline
        st.session_state.pipeline_initialized = True

if not service_error:
    service_error = st.session_state.get("pipeline_error")
pipeline_available = bool(
    st.session_state.pipeline_initialized and getattr(st.session_state, "pipeline", None)
)

main_col, side_col = st.columns([1.9, 1], gap="large")

with main_col:
    st.markdown(
        """
<div class="section-label">Ask Nayya</div>
<h2 class="conversation-title">What would you like to understand?</h2>
<p class="conversation-intro">Ask one question at a time. You do not need to share names,
addresses, case numbers, or other identifying details.</p>
""",
        unsafe_allow_html=True,
    )

    if not pipeline_available:
        st.markdown(
            """
<div class="unavailable" role="status">
  <strong>Answers are temporarily unavailable.</strong>
  You can still review the topics and safety guidance on this page. For legal help,
  contact a qualified legal-aid service, Protection Officer, or relevant local
  authority. If you are in immediate danger, contact local emergency services.
</div>
""",
            unsafe_allow_html=True,
        )

    control_col, reset_col = st.columns([2, 1])
    with control_col:
        st.radio(
            "Response style",
            ["Plain language", "Legal detail"],
            horizontal=True,
            key="response_style",
            help="Plain language explains concepts simply. Legal detail uses more statutory terminology.",
            disabled=not pipeline_available,
        )
    with reset_col:
        st.write("")
        if st.button("Reset conversation", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    suggestions = [
        "What counts as domestic violence under the Act?",
        "Can I ask for the right to stay in my shared home?",
        "What protection and monetary orders may be available?",
        "Who can help me make a complaint?",
    ]
    if not st.session_state.chat_history:
        st.caption("You could start with:")
        suggestion_cols = st.columns(2)
        for index, suggestion in enumerate(suggestions):
            with suggestion_cols[index % 2]:
                if st.button(
                    suggestion,
                    key=f"suggestion_{index}",
                    use_container_width=True,
                    disabled=not pipeline_available,
                ):
                    st.session_state.chat_history.append({"role": "user", "content": suggestion})
                    process_question(suggestion)
                    st.rerun()

    for message in st.session_state.chat_history:
        avatar = ":material/gavel:" if message["role"] == "assistant" else None
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                count = message.get("source_count", 0)
                if count:
                    st.markdown(
                        f'<div class="answer-meta">Source-grounded • {count} document excerpt{"s" if count != 1 else ""} consulted</div>',
                        unsafe_allow_html=True,
                    )
                    with st.expander("View source passages"):
                        for index, source in enumerate(message.get("sources", []), start=1):
                            label = f"S{index} · page {source.get('page', 'unknown')}"
                            if source.get("section"):
                                label += f" · section {source['section']}"
                            st.markdown(f"**{label}**")
                            st.caption(source.get("excerpt", ""))
                else:
                    st.caption("No matching source excerpt was found. Please verify this response.")

    input_placeholder = (
        "Ask about your rights or the complaint process"
        if pipeline_available
        else "Answers are temporarily unavailable"
    )
    if question := st.chat_input(input_placeholder, disabled=not pipeline_available):
        st.session_state.chat_history.append({"role": "user", "content": question})
        process_question(question)
        st.rerun()

with side_col:
    st.markdown(
        """
<section class="help-cluster">
  <h3 class="panel-title">What this can help with</h3>
  <div class="help-list">
    <div class="help-item"><b>01</b> Understand types of abuse recognised by the Act</div>
    <div class="help-item"><b>02</b> Explore protection, residence and monetary orders</div>
    <div class="help-item"><b>03</b> Learn how a complaint may move forward</div>
    <div class="help-item"><b>04</b> Find legal-aid and support pathways</div>
  </div>
</section>
<section class="trust-panel">
  <span class="source-badge">Source-aware by design</span>
  <h3 class="panel-title">A clear boundary builds trust.</h3>
  <p>Nayya provides legal information, not legal advice. Answers are generated from
  source material but may be incomplete or mistaken.</p>
  <p>Do not enter names, addresses, phone numbers, case IDs, or other sensitive
  personal details.</p>
  <p>Before acting, check important information with a qualified lawyer, legal-aid
  service, Protection Officer, or relevant authority.</p>
</section>
""",
        unsafe_allow_html=True,
    )

st.markdown(
    """
<footer class="site-footer">
  <strong>Scope:</strong> Information about the Protection of Women from Domestic
  Violence Act, 2005. Responses are generated from the service’s indexed source
  documents and should be independently verified. Nayya is not a government authority
  and does not replace a lawyer or emergency service.
</footer>
""",
    unsafe_allow_html=True,
)
