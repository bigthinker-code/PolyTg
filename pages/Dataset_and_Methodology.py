
import streamlit as st

st.set_page_config(
    page_title="Dataset & Methodology | PolyTg",
    page_icon="🔬",
    layout="wide"
)

st.title("Dataset & Methodology")

st.write(
    "An overview of the data, descriptor engineering, "
    "model development and evaluation strategy behind PolyTg."
)

st.divider()

st.header("1. Dataset")

st.write(
    "PolyTg was developed using a curated polymer glass "
    "transition temperature dataset from the PolyMetriX "
    "research ecosystem."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Original records", "7,367")

with col2:
    st.metric("Records retained", "7,363")

with col3:
    st.metric("Prediction target", "Experimental Tg")

st.write(
    "The input representation is polymer SMILES (PSMILES), "
    "while the prediction target is experimental glass "
    "transition temperature measured in Kelvin."
)

st.divider()

st.header("2. Data preparation")

st.write(
    "Four unreliable records were excluded during data "
    "preparation, leaving 7,363 records for model development."
)

st.write(
    "The dataset contains a substantial imbalance in polymer "
    "class representation. This is an important consideration "
    "when interpreting model performance and generalisation."
)

st.divider()

st.header("3. Feature engineering")

st.write(
    "PolyTg uses 100 descriptors organised into three groups."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Backbone descriptors", "30")

with col2:
    st.metric("Sidechain descriptors", "42")

with col3:
    st.metric("Full-polymer descriptors", "28")

st.write(
    "These descriptors capture structural and chemical "
    "characteristics of the polymer, including molecular "
    "composition, ring systems, hydrogen-bonding properties "
    "and polymer structural features."
)

st.divider()

st.header("4. Model development")

st.write(
    "An ExtraTrees regression model with 700 estimators "
    "was trained to predict experimental Tg."
)

st.markdown("""
**Data allocation**

- Training set: 70%
- Calibration set: 15%
- Test set: 15%
""")

st.write(
    "The calibration partition was used for conformal "
    "uncertainty estimation, while the test partition "
    "was used for performance evaluation."
)

st.divider()

st.header("5. Predictive performance")

st.write(
    "The following results describe performance on the "
    "project's held-out test partition."
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("MAE", "26.47 K")

with col2:
    st.metric("RMSE", "38.90 K")

with col3:
    st.metric("R²", "0.879")

st.caption(
    "These results are specific to the evaluation split and "
    "should not be interpreted as a guarantee of performance "
    "on new polymer families."
)

st.divider()

st.header("6. Conformal uncertainty estimation")

st.write(
    "Split conformal prediction was used with a nominal "
    "90% coverage level."
)

col1, col2 = st.columns(2)

with col1:
    st.metric("Calibration samples", "1,104")

with col2:
    st.metric("Conformal threshold", "62.447 K")

st.write(
    "The empirical coverage observed on the test partition "
    "was 90.23%, with an average interval width of 124.893 K."
)

st.divider()

st.header("7. Applicability domain")

st.write(
    "A StandardScaler and nearest-neighbour distance method "
    "were used to identify polymers that are structurally "
    "less represented in the training data."
)

col1, col2 = st.columns(2)

with col1:
    st.metric("Distance threshold", "5.208")

with col2:
    st.metric("Test samples within domain", "1,045 / 1,105")

st.write(
    "The applicability-domain assessment provides an "
    "additional caution mechanism. It does not establish "
    "whether an individual prediction is experimentally correct."
)

st.divider()

st.header("8. Generalisation considerations")

st.write(
    "A separate GroupKFold evaluation grouped by polymer "
    "class produced an MAE of 41.46 K, RMSE of 54.13 K "
    "and R² of 0.639."
)

st.warning(
    "The difference between random-split and polymer-class "
    "grouped evaluation highlights the challenge of predicting "
    "Tg for polymer classes that may be poorly represented "
    "during model training."
)

st.divider()

st.caption("PolyTg | Research Prototype")
