
import streamlit as st

st.set_page_config(
    page_title="Understanding Results | PolyTg",
    page_icon="📊",
    layout="wide"
)

st.title("Understanding Your Results")

st.write(
    "PolyTg reports more than a predicted glass transition "
    "temperature. This page explains how to interpret each result."
)

st.divider()

st.header("1. Predicted Glass Transition Temperature")

st.write(
    "The predicted Tg is the model's estimate of the temperature "
    "at which a polymer transitions between a hard, glassy state "
    "and a softer, more rubber-like state."
)

st.info(
    "PolyTg reports Tg in Kelvin (K) and Celsius (°C). "
    "The Celsius value is obtained by subtracting 273.15 from "
    "the Kelvin prediction."
)

st.divider()

st.header("2. 90% Conformal Prediction Interval")

st.write(
    "The conformal interval provides a range around the predicted "
    "Tg. It is calculated using a calibration threshold obtained "
    "from calibration data."
)

st.latex(r"[\,\hat{T}_g - q,\ \hat{T}_g + q\,]")

st.write(
    "Here, the predicted Tg is the model estimate and q is the "
    "stored conformal threshold."
)

st.warning(
    "The 90% coverage is a marginal coverage property under "
    "the exchangeability assumption. It is not a guarantee "
    "that an individual prediction has a 90% probability of "
    "being correct."
)

st.divider()

st.header("3. Applicability Domain")

st.write(
    "The applicability-domain assessment examines how far the "
    "submitted polymer's descriptor profile is from the training "
    "data, using its nearest-neighbour distance after scaling."
)

col1, col2 = st.columns(2)

with col1:
    st.success("Within the applicability domain")
    st.write(
        "The nearest-neighbour distance is at or below the "
        "model's established threshold."
    )

with col2:
    st.warning("Outside the applicability domain")
    st.write(
        "The distance exceeds the threshold. The prediction "
        "should be treated with additional caution."
    )

st.caption(
    "Applicability-domain status is a caution indicator, "
    "not a probability of prediction accuracy."
)

st.divider()

st.header("4. Important Limitations")

st.write(
    """
    - Predictions depend on the quality and coverage of the training data.
    - Polymers poorly represented in the training data may be harder to predict.
    - A conformal interval does not eliminate model error.
    - Applicability-domain status does not establish experimental validity.
    - Predictions should be interpreted as computational estimates,
      not substitutes for experimental measurements.
    """
)

st.divider()

st.caption("PolyTg | Research Prototype")
