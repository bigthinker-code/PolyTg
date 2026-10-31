
import streamlit as st

st.set_page_config(
    page_title="PolyTg | Polymer Intelligence",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- PAGE STYLE ----------

st.markdown("""
<style>
    .stApp {
        background-color: #07111f;
        color: #e5edf5;
    }

    [data-testid="stSidebar"] {
        background-color: #0b1929;
        border-right: 1px solid #1c3448;
    }

    .hero {
        padding: 65px 10px 45px 10px;
    }

    .eyebrow {
        color: #35d0c1;
        font-size: 13px;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
    }

    .hero h1 {
        color: #f3f7fb;
        font-size: 58px;
        font-weight: 750;
        line-height: 1.1;
        margin-top: 18px;
        margin-bottom: 20px;
    }

    .hero p {
        color: #a9bacb;
        font-size: 18px;
        line-height: 1.8;
        max-width: 760px;
    }

    .accent {
        color: #35d0c1;
    }

    .section-label {
        color: #35d0c1;
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 2px;
    }

    .info-card {
        background: #0d1d2e;
        border: 1px solid #1b354a;
        border-radius: 12px;
        padding: 24px;
        min-height: 175px;
    }

    .info-card h3 {
        color: #f0f5fa;
        font-size: 19px;
    }

    .info-card p {
        color: #a9bacb;
        font-size: 14px;
        line-height: 1.7;
    }

    .footer {
        color: #71869b;
        font-size: 12px;
        border-top: 1px solid #1b354a;
        padding-top: 20px;
        margin-top: 65px;
    }
</style>
""", unsafe_allow_html=True)


# ---------- SIDEBAR ----------

with st.sidebar:
    st.markdown("## 🧪 PolyTg")
    st.caption("POLYMER INTELLIGENCE")
    st.divider()

    st.markdown("### Navigation")
    st.page_link("app.py", label="Home", icon="🏠")
    st.page_link(
        "pages/Tg_Predictor.py",
        label="Tg Predictor",
        icon="🧬"
    )


# ---------- HERO SECTION ----------

st.markdown("""
<div class="hero">
    <div class="eyebrow">Machine Learning · Polymer Science</div>

    <h1>
        Understanding polymers.<br>
        Predicting <span class="accent">possibilities.</span>
    </h1>

    <p>
        PolyTg is a machine-learning platform for estimating the
        glass transition temperature of polymers from their
        polymer SMILES structures.
        
        By combining polymer descriptors, predictive modelling,
        uncertainty estimation and applicability-domain assessment,
        PolyTg provides a research-oriented approach to polymer
        property prediction.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------- CALL TO ACTION ----------

left, right = st.columns([1, 1])

with left:
    if st.button(
        "Open Tg Predictor →",
        type="primary",
        use_container_width=True
    ):
        st.switch_page("pages/Tg_Predictor.py")

with right:
    st.markdown(
        "<p style='color:#8da3b8; padding-top:8px;'>"
        "Explore the platform and its scientific methodology."
        "</p>",
        unsafe_allow_html=True
    )


st.divider()


# ---------- PLATFORM OVERVIEW ----------

st.markdown(
    '<div class="section-label">PLATFORM OVERVIEW</div>',
    unsafe_allow_html=True
)

st.markdown("## What PolyTg offers")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div class="info-card">
        <h3>🧬 Polymer Representation</h3>
        <p>
            Submit polymer SMILES structures and generate
            molecular and polymer-level descriptors for
            machine-learning inference.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="info-card">
        <h3>📊 Tg Prediction</h3>
        <p>
            Estimate polymer glass transition temperature
            in Kelvin and Celsius using a trained ExtraTrees
            regression model.
        </p>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="info-card">
        <h3>🎯 Prediction Assessment</h3>
        <p>
            Examine conformal prediction intervals and
            applicability-domain assessments to understand
            prediction uncertainty and limitations.
        </p>
    </div>
    """, unsafe_allow_html=True)


# ---------- FOOTER ----------

st.markdown("""
<div class="footer">
    PolyTg · Research Prototype<br>
    Machine Learning for Polymer Property Prediction
</div>
""", unsafe_allow_html=True)
