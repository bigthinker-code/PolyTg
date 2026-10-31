
import streamlit as st

st.set_page_config(
    page_title="Model Architecture | PolyTg",
    page_icon="🧠",
    layout="wide"
)

st.title("Model Architecture")

st.write(
    "PolyTg uses a descriptor-based machine-learning pipeline "
    "to estimate the glass transition temperature of polymers."
)

st.divider()

st.header("The prediction pipeline")

st.markdown("""
<style>
.pipeline-step {
    background: #0d1d2e;
    border: 1px solid #1b354a;
    border-radius: 12px;
    padding: 20px;
    margin-bottom: 10px;
}
.pipeline-step h4 {
    color: #35d0c1;
    margin-bottom: 8px;
}
.pipeline-step p {
    color: #c0cedb;
    margin-bottom: 0;
}
</style>

<div class="pipeline-step">
    <h4>01 — Polymer Input</h4>
    <p>Submit a polymer structure represented using polymer SMILES (PSMILES).</p>
</div>

<div style="text-align:center; color:#35d0c1; font-size:22px;">↓</div>

<div class="pipeline-step">
    <h4>02 — Polymer Representation</h4>
    <p>The PSMILES string is converted into a polymer object using PolyMetriX.</p>
</div>

<div style="text-align:center; color:#35d0c1; font-size:22px;">↓</div>

<div class="pipeline-step">
    <h4>03 — Descriptor Generation</h4>
    <p>PolyTg calculates 100 descriptors across backbone, sidechain and full-polymer feature groups.</p>
</div>

<div style="text-align:center; color:#35d0c1; font-size:22px;">↓</div>

<div class="pipeline-step">
    <h4>04 — ExtraTrees Regression</h4>
    <p>The trained ExtraTrees model uses the generated descriptors to estimate glass transition temperature.</p>
</div>

<div style="text-align:center; color:#35d0c1; font-size:22px;">↓</div>

<div class="pipeline-step">
    <h4>05 — Prediction Assessment</h4>
    <p>A conformal prediction interval and nearest-neighbour applicability-domain assessment provide additional context.</p>
</div>

<div style="text-align:center; color:#35d0c1; font-size:22px;">↓</div>

<div class="pipeline-step">
    <h4>06 — Results</h4>
    <p>The platform reports predicted Tg in Kelvin and Celsius, the 90% conformal interval and applicability-domain status.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

st.header("Machine-learning model")

st.subheader("ExtraTrees Regressor")

st.write(
    "PolyTg uses an ExtraTrees regression model trained to "
    "predict experimental polymer glass transition temperatures "
    "from polymer descriptors."
)

st.markdown("""
- **Model:** ExtraTrees Regressor
- **Number of estimators:** 700
- **Input:** 100 polymer descriptors
- **Prediction target:** Experimental glass transition temperature (Kelvin)
""")

st.divider()

st.header("Uncertainty estimation")

st.write(
    "PolyTg uses split conformal prediction. A calibration "
    "dataset is used to determine a threshold that defines "
    "the prediction interval."
)

st.metric(
    "Nominal interval coverage",
    "90%"
)

st.metric(
    "Calibration threshold",
    "62.447 K"
)

st.caption(
    "The nominal coverage relies on the exchangeability "
    "assumption and is not an individual prediction guarantee."
)

st.divider()

st.header("Applicability-domain assessment")

st.write(
    "The model uses a StandardScaler and a nearest-neighbour "
    "distance calculation to assess whether a polymer's "
    "descriptor profile is close to the training data."
)

st.metric(
    "Applicability-domain threshold",
    "5.208"
)

st.caption(
    "A prediction outside this threshold is flagged for "
    "additional caution. It does not automatically mean "
    "the prediction is incorrect."
)

st.divider()

st.caption("PolyTg | Research Prototype")
