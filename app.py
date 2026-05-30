import streamlit as st
import tensorflow as tf
import numpy as np
import joblib, pandas as pd, pickle
from groq import Groq
from sklearn.metrics.pairwise import cosine_similarity

# Konfigurasi Halaman
st.set_page_config(
    page_title="Skill Prediction",
    page_icon="assets/favicon.ico",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

:root {
    --bg-main:       #FFFFFF;
    --bg-secondary:  #F8FAFC;
    --accent-purple: #7C3AED;
    --accent-light:  #F5F3FF;
    --border:        #E2E8F0;
    --text-main:     #1E293B;
    --text-muted:    #64748B;
}

html, body, [class*="css"] {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    color: var(--text-main);
}

.stApp {
    background-color: var(--bg-main) !important;
}

[data-testid="stSidebar"] {
    background-color: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}

.main-header {
    background: #FFFFFF;
    padding: 2.5rem;
    border-radius: 24px;
    border: 1px solid var(--border);
    margin-bottom: 2rem;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.02);
}

.header-tag {
    color: var(--accent-purple);
    font-weight: 700;
    font-size: 0.9rem;
    margin-bottom: 0.5rem;
    display: block;
}

.skill-row {
    display: flex;
    align-items: center;
    padding: 1.2rem;
    background: white;
    border-radius: 12px;
    border: 1px solid var(--border);
    margin-bottom: 0.75rem;
    transition: all 0.2s ease;
}

.skill-row:hover {
    border-color: var(--accent-purple);
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(124, 58, 237, 0.08);
}

.rank-circle {
    background: var(--accent-light);
    color: var(--accent-purple);
    width: 32px;
    height: 32px;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 700;
    margin-right: 1rem;
    font-size: 0.8rem;
}

.status-badge {
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.5px;
}
.high-demand   { background: var(--accent-purple); color: white; }
.normal-demand { background: var(--accent-light);  color: var(--accent-purple); }

div[data-testid="stTextInput"] input {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 12px 16px !important;
    background: #FDFDFD !important;
}

div[data-testid="stTextInput"] input:focus {
    border-color: var(--accent-purple) !important;
}

button[kind="primary"] {
    background-color: var(--accent-purple) !important;
    border: none !important;
    height: 48px !important;
    border-radius: 12px !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(124, 58, 237, 0.25) !important;
}

.stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: var(--accent-purple);
}

#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# Registrasi Custom Components Model
@tf.keras.utils.register_keras_serializable(package="SkillRec")
class AttentionLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs): super().__init__(**kwargs)
    def call(self, inputs):
        return tf.reduce_sum(tf.nn.softmax(inputs, axis=1) * inputs, axis=1)
    def get_config(self): return super().get_config()

@tf.keras.utils.register_keras_serializable(package="SkillRec")
def weighted_bce(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    bce    = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    return tf.reduce_mean((y_true * 8.0 + (1 - y_true)) * bce)


@st.cache_resource
def load_assets():
    model    = tf.keras.models.load_model("skill_recommender_v1 (1).keras")
    mlb      = joblib.load("mlb_fix (1).pkl")

    # Load TF-IDF keyword mapper
    with open("tfidf_vec.pkl", "rb") as f:
        tfidf_vec = pickle.load(f)
    with open("keywords.pkl", "rb") as f:
        keywords = pickle.load(f)
    keyword_matrix = tfidf_vec.transform(keywords)

    return model, mlb, tfidf_vec, keywords, keyword_matrix

model, mlb, tfidf_vec, keywords, keyword_matrix = load_assets()


def map_to_keyword(user_input: str, threshold: float = 0.15) -> str:
    """Petakan input bebas user ke keyword terdekat via TF-IDF similarity."""
    vec    = tfidf_vec.transform([user_input.lower()])
    scores = cosine_similarity(vec, keyword_matrix).flatten()
    idx    = int(np.argmax(scores))
    if scores[idx] >= threshold:
        return keywords[idx]
    return user_input  # fallback kalau tidak ada yang cukup mirip


# Sidebar
with st.sidebar:
    st.title("System Control")
    st.markdown("---")

    threshold    = st.slider("Minimal Confidence", 0.05, 0.50, 0.20)
    top_n        = st.slider("Jumlah Rekomendasi", 5, 25, 10)
    show_roadmap = st.checkbox("Generate AI Roadmap", value=True)

    st.markdown("---")
    st.subheader("Data Insight")
    st.metric("Total Linkedin Jobs Analyzed", "4,542")
    st.metric("Categories", "105")


# Header
st.markdown("""
<div class="main-header">
    <h4 style="color: #2563EB; margin: 0;">Skill Recommendation</h4>
    <h1 style="margin: 10px 0; font-size: 2.5rem; font-weight: 800;">IT Skill Recommender</h1>
    <p style="color: #64748B; font-size: 1.1rem; max-width: 700px;">
        Temukan skill yang paling dicari berdasarkan data lowongan kerja LinkedIn.
    </p>
</div>
""", unsafe_allow_html=True)


# Input
c1, c2 = st.columns([3, 1])
with c1:
    user_input = st.text_input("Job Position", placeholder="Masukkan nama pekerjaan")
with c2:
    st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
    predict_btn = st.button("Prediksi dan Analisis", type="primary", use_container_width=True)


if predict_btn and user_input:
    with st.spinner("Processing..."):
        job_mapped = map_to_keyword(user_input)
        pred       = model.predict(tf.constant([job_mapped]), verbose=0)[0]

        p_min, p_max = pred.min(), pred.max()
        pred_cal     = (pred - p_min) / (p_max - p_min) if p_max > p_min else pred
        top_idx      = np.argsort(pred_cal)[::-1][:top_n]
        results      = [(mlb.classes_[i], float(pred_cal[i])) for i in top_idx if pred_cal[i] >= threshold]

    col_left, col_right = st.columns([1.2, 1], gap="large")

    with col_left:
        st.subheader("Rekomendasi Utama")

        # Tampilkan keyword hasil mapping — kalau berbeda dari input, beri tahu user
        if job_mapped.lower() != user_input.lower().strip():
            st.markdown(f"Pekerjaan dideteksi sebagai: **{job_mapped}**")
        else:
            st.markdown(f"Pekerjaan dideteksi sebagai: **{job_mapped}**")

        for rank, (skill, score) in enumerate(results, 1):
            badge_class = "high-demand" if score > 0.6 else "normal-demand"
            st.markdown(f"""
            <div class="skill-row">
                <div class="rank-number">{rank:02d}</div>
                <div style="flex-grow: 1;">
                    <div style="font-weight: 600;">{skill.upper()}</div>
                    <div style="font-size: 0.8rem; color: #94A3B8;">Confidence Score: {score:.2f}</div>
                </div>
                <div class="status-badge {badge_class}">{"Primary" if score > 0.6 else "Support"}</div>
            </div>
            """, unsafe_allow_html=True)

    with col_right:
        st.subheader("Data Visualization")
        chart_data = pd.DataFrame(results, columns=["Skill", "Score"])
        st.bar_chart(chart_data.set_index("Skill"), color="#2563EB")

        if show_roadmap:
            st.markdown("---")
            st.subheader("Roadmap Recommendation")
            try:
                client      = Groq(api_key=st.secrets["GROQ_API_KEY"])
                skills_list = ", ".join([s for s, _ in results])
                resp = client.chat.completions.create(
                    messages=[{
                        "role": "user",
                        "content": (
                            f"Berikan roadmap belajar profesional untuk {user_input} "
                            f"dengan skill utama: {skills_list}. Format Markdown, tanpa emotikon."
                            f"serta berikan rekomendasi sertifikasi dengan minimal memuat dicoding academy"
                        )
                    }],
                    model="llama-3.3-70b-versatile",
                )
                st.info(resp.choices[0].message.content)
            except Exception:
                st.warning("Roadmap AI sedang sibuk.")