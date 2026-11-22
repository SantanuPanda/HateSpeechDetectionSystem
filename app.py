import os
import re
import time
import html
import joblib
import warnings
import streamlit as st

warnings.filterwarnings("ignore")

# ── Page Configuration ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hate Speech Detection System",
    page_icon="🗣️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Smart Stopword Filter: keep pronouns & directional words so "i kill you" doesn't reduce to just "kill"
custom_stops = set(ENGLISH_STOP_WORDS) - {
    "i", "you", "he", "she", "they", "we", "me", "him", "her", "them",
    "all", "not", "no", "against", "down", "out", "off", "up",
}


# ── Text Preprocessing ────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\S+|https\S+", "", str(text))
    text = re.sub(r"\@\w+|\#", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    text = text.lower()
    return " ".join(w for w in text.split() if w not in custom_stops)


# ── Load Model and Vectorizer ─────────────────────────────────────────────────
MODEL_PATH = "stacking_model.pkl"
VECTORIZER_PATH = "tfidf_vectorizer.pkl"


@st.cache_resource
def load_model_and_vectorizer():
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
        model = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        return model, vectorizer
    return None, None


model, vectorizer = load_model_and_vectorizer()


# ── Styling ───────────────────────────────────────────────────────────────────
CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=DM+Sans:wght@400;500;600&display=swap');

:root {
  --ink: #0e0a23;
  --plum: #241446;
  --violet: #8466ff;
  --rose: #ff5c8a;
  --mint: #3ddc97;
  --cloud: #f2eeff;
  --muted: #a99fd1;
  --glass: rgba(255, 255, 255, 0.06);
  --glass-edge: rgba(255, 255, 255, 0.14);
}

/* ---------- Base ---------- */
html, body, [class*="css"], .stApp {
  font-family: 'DM Sans', sans-serif;
  color: var(--cloud);
}
.stApp {
  background: var(--ink);
  background-image:
    radial-gradient(60rem 40rem at 15% -10%, rgba(132, 102, 255, 0.35), transparent 60%),
    radial-gradient(50rem 36rem at 95% 10%, rgba(255, 92, 138, 0.22), transparent 60%),
    radial-gradient(50rem 40rem at 50% 110%, rgba(61, 220, 151, 0.14), transparent 60%);
  background-attachment: fixed;
}
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { max-width: 760px; padding-top: 2.2rem; padding-bottom: 3rem; position: relative; z-index: 2; }

/* ---------- Floating orbs (page-load drift, then they just breathe) ---------- */
.orb {
  position: fixed; border-radius: 50%; filter: blur(70px);
  pointer-events: none; z-index: 0; opacity: .55;
  animation: drift 18s ease-in-out infinite alternate;
}
.orb.o1 { width: 340px; height: 340px; background: var(--violet); top: 8%; left: -6%; }
.orb.o2 { width: 280px; height: 280px; background: var(--rose); top: 55%; right: -8%; animation-duration: 22s; animation-delay: -6s; }
.orb.o3 { width: 220px; height: 220px; background: var(--mint); bottom: -6%; left: 30%; opacity: .3; animation-duration: 26s; animation-delay: -12s; }
@keyframes drift {
  0%   { transform: translate(0, 0) scale(1); }
  50%  { transform: translate(40px, -30px) scale(1.12); }
  100% { transform: translate(-30px, 40px) scale(0.94); }
}

/* ---------- Hero ---------- */
.hero { text-align: center; padding: 1.2rem 0 0.4rem; animation: rise .9s cubic-bezier(.2,.8,.2,1) both; }
@keyframes rise { from { opacity: 0; transform: translateY(24px); } to { opacity: 1; transform: none; } }
.bird {
  display: inline-block; font-size: 3.4rem; line-height: 1;
  filter: drop-shadow(0 10px 24px rgba(132, 102, 255, .55));
  animation: float 4.5s ease-in-out infinite;
}
@keyframes float {
  0%, 100% { transform: translateY(0) rotate(-4deg); }
  50%      { transform: translateY(-14px) rotate(5deg); }
}
.hero h1 {
  font-family: 'Bricolage Grotesque', sans-serif;
  font-weight: 800; letter-spacing: -0.03em;
  font-size: clamp(2.2rem, 6vw, 3.4rem); line-height: 1.05;
  margin: .6rem 0 .5rem; padding: 0;
  background: linear-gradient(100deg, #fff 0%, #cbbcff 35%, #ff9dbb 70%, #fff 100%);
  background-size: 220% auto;
  -webkit-background-clip: text; background-clip: text; color: transparent;
  animation: shimmer 7s linear infinite;
}
@keyframes shimmer { to { background-position: 220% center; } }
.hero p { color: var(--muted); font-size: 1.05rem; max-width: 34rem; margin: 0 auto 1rem; }

.pills { display: flex; flex-wrap: wrap; gap: .5rem; justify-content: center; margin-bottom: .4rem; }
.pill {
  padding: .3rem .85rem; border-radius: 999px; font-size: .8rem; font-weight: 500;
  background: var(--glass); border: 1px solid var(--glass-edge); color: var(--cloud);
  backdrop-filter: blur(8px);
}

/* ---------- Glass panel ---------- */
.panel-label { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.05rem; margin: 1.6rem 0 .5rem; color: var(--cloud); }

/* ---------- Buttons ---------- */
.stButton > button {
  border-radius: 14px; border: 1px solid var(--glass-edge);
  background: var(--glass); color: var(--cloud);
  font-weight: 600; font-family: 'DM Sans', sans-serif;
  padding: .6rem 1rem; backdrop-filter: blur(10px);
  transition: transform .18s ease, background .25s ease, border-color .25s ease, box-shadow .25s ease;
}
.stButton > button:hover {
  transform: translateY(-3px);
  border-color: rgba(132, 102, 255, .7);
  background: rgba(132, 102, 255, .16);
  box-shadow: 0 10px 28px rgba(132, 102, 255, .25);
  color: #fff;
}
.stButton > button:active { transform: translateY(0) scale(.97); }
.stButton > button:focus-visible { outline: 2px solid var(--violet); outline-offset: 2px; }

.stButton > button[kind="primary"] {
  border: none; color: #fff;
  background: linear-gradient(120deg, var(--violet), var(--rose), var(--violet));
  background-size: 200% auto;
  box-shadow: 0 10px 30px rgba(132, 102, 255, .4);
  animation: flow 5s linear infinite;
}
.stButton > button[kind="primary"]:hover {
  box-shadow: 0 14px 38px rgba(255, 92, 138, .45);
  background: linear-gradient(120deg, var(--violet), var(--rose), var(--violet));
  background-size: 200% auto;
}
@keyframes flow { to { background-position: 200% center; } }

/* ---------- Text area ---------- */
.stTextArea label p { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 700; font-size: 1.05rem; color: var(--cloud); }
.stTextArea textarea {
  background: rgba(255, 255, 255, 0.05) !important;
  border: 1px solid var(--glass-edge) !important;
  border-radius: 18px !important;
  padding: 1rem 1.1rem !important; font-size: 1.02rem !important; line-height: 1.55 !important;
  backdrop-filter: blur(12px);
  transition: border-color .25s ease, box-shadow .25s ease, background .25s ease;
}
.stTextArea textarea::placeholder { color: #8277b3 !important; }
.stTextArea textarea:focus {
  border-color: var(--violet) !important;
  box-shadow: 0 0 0 4px rgba(132, 102, 255, .22), 0 12px 40px rgba(132, 102, 255, .18) !important;
  background: rgba(255, 255, 255, 0.08) !important;
}

/* ---------- Result card ---------- */
.result {
  position: relative; overflow: hidden; margin-top: 1.4rem;
  padding: 1.6rem 1.6rem 1.4rem; border-radius: 24px;
  background: var(--glass); border: 1px solid var(--glass-edge);
  backdrop-filter: blur(16px);
  animation: pop .6s cubic-bezier(.2,1.2,.3,1) both;
}
@keyframes pop { from { opacity: 0; transform: translateY(18px) scale(.94); } to { opacity: 1; transform: none; } }
.result::before {  /* sweeping light, plays once on arrival */
  content: ""; position: absolute; inset: 0;
  background: linear-gradient(105deg, transparent 35%, rgba(255,255,255,.18) 50%, transparent 65%);
  transform: translateX(-120%); animation: sweep 1.2s .35s ease-out forwards; pointer-events: none;
}
@keyframes sweep { to { transform: translateX(120%); } }

.result.safe { border-color: rgba(61, 220, 151, .55); box-shadow: 0 0 60px rgba(61, 220, 151, .18); }
.result.hate { border-color: rgba(255, 92, 138, .6); box-shadow: 0 0 60px rgba(255, 92, 138, .22); animation: pop .6s cubic-bezier(.2,1.2,.3,1) both, shake .55s .5s ease; }
@keyframes shake {
  0%, 100% { transform: translateX(0); }
  20% { transform: translateX(-7px); } 40% { transform: translateX(7px); }
  60% { transform: translateX(-4px); } 80% { transform: translateX(4px); }
}

.verdict { display: flex; align-items: center; gap: 1rem; }
.badge {
  flex: none; width: 64px; height: 64px; border-radius: 50%;
  display: grid; place-items: center; font-size: 1.9rem;
}
.safe .badge { background: rgba(61, 220, 151, .16); animation: ring-mint 2.2s ease-out infinite; }
.hate .badge { background: rgba(255, 92, 138, .16); animation: ring-rose 1.4s ease-out infinite; }
@keyframes ring-mint { 0% { box-shadow: 0 0 0 0 rgba(61,220,151,.55); } 100% { box-shadow: 0 0 0 22px rgba(61,220,151,0); } }
@keyframes ring-rose { 0% { box-shadow: 0 0 0 0 rgba(255,92,138,.6); } 100% { box-shadow: 0 0 0 22px rgba(255,92,138,0); } }

.verdict h2 { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 1.7rem; letter-spacing: -0.02em; margin: 0; padding: 0; }
.safe h2 { color: var(--mint); }
.hate h2 { color: var(--rose); }
.verdict p { margin: .25rem 0 0; color: var(--muted); font-size: .98rem; }

/* ---------- Confidence bars ---------- */
.bars { margin-top: 1.4rem; display: grid; gap: .9rem; }
.bar-head { display: flex; justify-content: space-between; font-size: .9rem; margin-bottom: .35rem; color: var(--cloud); font-weight: 500; }
.track { height: 10px; border-radius: 999px; background: rgba(255,255,255,.08); overflow: hidden; }
.fill { height: 100%; width: 0; border-radius: 999px; animation: grow 1.3s .4s cubic-bezier(.2,.8,.2,1) forwards; }
.fill.normal { background: linear-gradient(90deg, #1fae73, var(--mint)); }
.fill.toxic  { background: linear-gradient(90deg, #d93a6a, var(--rose)); }
@keyframes grow { to { width: var(--w); } }

/* ---------- Processing details ---------- */
.detail { margin-top: .4rem; padding: 1rem 1.1rem; border-radius: 16px; background: rgba(0,0,0,.25); border: 1px solid var(--glass-edge); animation: pop .5s both; }
.detail .k { color: var(--muted); font-size: .82rem; margin-bottom: .2rem; }
.detail .v { font-size: .98rem; word-break: break-word; }
.detail code { background: rgba(132,102,255,.18); color: #d9cfff; padding: .15rem .5rem; border-radius: 8px; }
.detail + .detail { margin-top: .7rem; }
[data-testid="stExpander"] { border: 1px solid var(--glass-edge) !important; border-radius: 16px !important; background: var(--glass); margin-top: 1rem; }
[data-testid="stExpander"] summary:hover { color: #fff; }

/* ---------- Alerts / divider / footer ---------- */
[data-testid="stAlert"] { border-radius: 16px; animation: pop .45s both; }
hr { border-color: var(--glass-edge) !important; }
.foot { text-align: center; color: var(--muted); font-size: .85rem; margin-top: 2.4rem; }
.foot b { color: var(--cloud); font-weight: 600; }

/* ---------- Accessibility ---------- */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after { animation-duration: .01ms !important; animation-iteration-count: 1 !important; transition-duration: .01ms !important; }
  .fill { width: var(--w); }
}
@media (max-width: 640px) {
  .verdict { flex-direction: column; text-align: center; }
}
</style>

<div class="orb o1"></div>
<div class="orb o2"></div>
<div class="orb o3"></div>
"""


def html_block(markup: str) -> str:
    """Flatten HTML to one line so Streamlit's markdown never turns indents into code blocks."""
    return "".join(line.strip() for line in markup.splitlines())


st.markdown(CSS, unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    html_block(
        """
        <div class="hero">
            <div class="bird">🗣️</div>
            <h1>Hate Speech Detection System</h1>
            <p>Paste any tweet and find out in a moment whether it contains hate speech or is normal, safe conversation.</p>
            <div class="pills">
                <span class="pill">Random Forest</span>
                <span class="pill">XGBoost</span>
                <span class="pill">Logistic Regression</span>
                <span class="pill">TF-IDF</span>
            </div>
        </div>
        """
    ),
    unsafe_allow_html=True,
)

if model is None or vectorizer is None:
    st.error("⚠️ Model files (`stacking_model.pkl`, `tfidf_vectorizer.pkl`) not found in the directory.")
    st.stop()

# ── Session state + callbacks ─────────────────────────────────────────────────
if "tweet_input" not in st.session_state:
    st.session_state["tweet_input"] = ""

SAMPLES = {
    "😊 Friendly": "I love spending quality time with my friends and family! Such a beautiful day ❤️",
    "📚 Excited": "Can't wait for the new book release tomorrow, so excited!",
    "🚨 Aggressive": "I hate these disgusting people, they should all be eliminated.",
}


def set_sample(text: str):
    st.session_state["tweet_input"] = text


def clear_input():
    st.session_state["tweet_input"] = ""


# ── Samples ───────────────────────────────────────────────────────────────────
st.markdown('<div class="panel-label">Try a sample</div>', unsafe_allow_html=True)
cols = st.columns(3)
for col, (label, text) in zip(cols, SAMPLES.items()):
    with col:
        st.button(label, on_click=set_sample, args=(text,), use_container_width=True, key=f"sample_{label}")

# ── Input ─────────────────────────────────────────────────────────────────────
user_input = st.text_area(
    "Your tweet",
    key="tweet_input",
    placeholder="Type or paste a tweet here...",
    height=130,
)

col_btn1, col_btn2 = st.columns([3, 1])
with col_btn1:
    predict_clicked = st.button("🔍 Analyze tweet", type="primary", use_container_width=True)
with col_btn2:
    st.button("🗑️ Clear", on_click=clear_input, use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if predict_clicked:
    if not user_input.strip():
        st.warning("⚠️ Please enter a tweet before clicking Analyze.")
    else:
        with st.spinner("Reading the tweet..."):
            time.sleep(0.6)  # short pause so the result feels earned
            cleaned = clean_text(user_input)
            vec_features = vectorizer.transform([cleaned])
            prediction = model.predict(vec_features)[0]

            proba = None
            if hasattr(model, "predict_proba"):
                try:
                    proba = model.predict_proba(vec_features)[0]
                except Exception:
                    proba = None

        is_hate = prediction == 1

        if is_hate:
            card_class, icon = "hate", "🚨"
            title = "Hate speech detected"
            desc = "This tweet was flagged as hateful or toxic. It contains hostile or abusive patterns."
        else:
            card_class, icon = "safe", "✅"
            title = "Looks safe"
            desc = "This tweet reads as normal conversation. No hate speech patterns were found."

        bars_html = ""
        if proba is not None and len(proba) == 2:
            normal_pct = proba[0] * 100
            hate_pct = proba[1] * 100
            bars_html = f"""
            <div class="bars">
                <div>
                    <div class="bar-head"><span>Normal</span><span>{normal_pct:.1f}%</span></div>
                    <div class="track"><div class="fill normal" style="--w:{normal_pct:.1f}%"></div></div>
                </div>
                <div>
                    <div class="bar-head"><span>Hate speech</span><span>{hate_pct:.1f}%</span></div>
                    <div class="track"><div class="fill toxic" style="--w:{hate_pct:.1f}%"></div></div>
                </div>
            </div>
            """

        st.markdown(
            html_block(
                f"""
                <div class="result {card_class}">
                    <div class="verdict">
                        <div class="badge">{icon}</div>
                        <div>
                            <h2>{title}</h2>
                            <p>{desc}</p>
                        </div>
                    </div>
                    {bars_html}
                </div>
                """
            ),
            unsafe_allow_html=True,
        )

        with st.expander("🔎 See how the text was processed"):
            shown_clean = html.escape(cleaned) if cleaned else "(empty after removing stopwords and links)"
            st.markdown(
                html_block(
                    f"""
                    <div class="detail">
                        <div class="k">Original text</div>
                        <div class="v">{html.escape(user_input)}</div>
                    </div>
                    <div class="detail">
                        <div class="k">Cleaned and tokenized</div>
                        <div class="v"><code>{shown_clean}</code></div>
                    </div>
                    """
                ),
                unsafe_allow_html=True,
            )

st.markdown(
    '<div class="foot">Hate Speech Detection System, developed by <b>Santanu Panda</b> · Streamlit · Python 3.13 ready</div>',
    unsafe_allow_html=True,
)