
import os
import traceback

import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import numpy as np
import joblib

from huggingface_hub import hf_hub_download
from featurizer import get_featurizer
from polymetrix.featurizers.polymer import Polymer


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="PolyTg | Polymer Intelligence",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --------------------------------------------------
# CUSTOM FRONTEND COMPONENT
# --------------------------------------------------

FRONTEND_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "frontend"
)

polytg_frontend = components.declare_component(
    "polytg_frontend",
    path=FRONTEND_PATH
)


# --------------------------------------------------
# LOAD EXISTING MODEL ASSETS
# --------------------------------------------------

@st.cache_resource
def load_assets():

    repo_id = "Abdulroheem/PolyTg"

    model_path = hf_hub_download(
        repo_id=repo_id,
        filename="polymer_tg_extra_trees.pkl"
    )

    ad_path = hf_hub_download(
        repo_id=repo_id,
        filename="applicability_domain.pkl"
    )

    uncertainty_path = hf_hub_download(
        repo_id=repo_id,
        filename="uncertainty_configuration.pkl"
    )

    model = joblib.load(model_path)
    ad = joblib.load(ad_path)
    uncertainty = joblib.load(uncertainty_path)

    featurizer, feature_columns = get_featurizer()

    return (
        model,
        ad,
        uncertainty,
        featurizer,
        feature_columns
    )


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict_tg(
    psmiles,
    model,
    ad,
    uncertainty,
    featurizer,
    feature_columns
):

    polymer = Polymer.from_psmiles(psmiles)

    features = np.asarray(
        featurizer.featurize(polymer)
    )

    if features.ndim == 1:
        features = features.reshape(1, -1)

    generated_columns = list(feature_columns)
    expected_columns = list(ad["feature_columns"])

    if features.shape[1] != len(generated_columns):
        raise ValueError(
            "Generated feature count does not match "
            "the featurizer's feature labels."
        )

    if len(set(generated_columns)) != len(generated_columns):
        raise ValueError(
            "Duplicate feature names were generated."
        )

    if set(generated_columns) != set(expected_columns):

        missing = [
            name for name in expected_columns
            if name not in generated_columns
        ]

        unexpected = [
            name for name in generated_columns
            if name not in expected_columns
        ]

        raise ValueError(
            f"Feature names do not match the trained model. "
            f"Missing: {missing}. Unexpected: {unexpected}."
        )

    feature_positions = {
        name: index
        for index, name in enumerate(generated_columns)
    }

    reorder_indices = [
        feature_positions[name]
        for name in expected_columns
    ]

    features = features[:, reorder_indices]

    X = pd.DataFrame(
        features,
        columns=expected_columns
    )

    # Tg prediction
    prediction = float(model.predict(X)[0])

    # Conformal interval
    margin = float(
        uncertainty["conformal_threshold"]
    )

    lower = prediction - margin
    upper = prediction + margin

    # Applicability-domain assessment
    scaled = ad["scaler"].transform(X)

    distance = float(
        ad["nearest_neighbors"].kneighbors(
            scaled,
            n_neighbors=1
        )[0][0][0]
    )

    threshold = float(ad["ad_threshold"])

    within_domain = distance <= threshold

    return {
        "success": True,
        "psmiles": psmiles,
        "prediction_kelvin": round(prediction, 2),
        "prediction_celsius": round(
            prediction - 273.15, 2
        ),
        "lower_bound": round(lower, 2),
        "upper_bound": round(upper, 2),
        "interval_width": round(2 * margin, 2),
        "nearest_distance": round(distance, 3),
        "ad_threshold": round(threshold, 3),
        "within_domain": within_domain
    }


# --------------------------------------------------
# MAIN APPLICATION
# --------------------------------------------------

try:

    (
        model,
        ad,
        uncertainty,
        featurizer,
        feature_columns
    ) = load_assets()

except Exception as e:

    st.error(f"Could not load model assets: {e}")
    st.code(traceback.format_exc())
    st.stop()


# --------------------------------------------------
# COMPONENT STATE
# --------------------------------------------------

if "polytg_result" not in st.session_state:
    st.session_state.polytg_result = None

if "polytg_error" not in st.session_state:
    st.session_state.polytg_error = None


# --------------------------------------------------
# SEND DATA TO FRONTEND
# --------------------------------------------------

component_value = polytg_frontend(
    results=st.session_state.polytg_result,
    error=st.session_state.polytg_error,
    height=1100,
    key="polytg_interface"
)


# --------------------------------------------------
# HANDLE FRONTEND REQUESTS
# --------------------------------------------------

if isinstance(component_value, dict):

    action = component_value.get("action")

    if action == "predict":

        psmiles = str(
            component_value.get("psmiles", "")
        ).strip()

        if not psmiles:

            st.session_state.polytg_error = (
                "Please enter a polymer SMILES."
            )

            st.session_state.polytg_result = None

        else:

            try:

                result = predict_tg(
                    psmiles,
                    model,
                    ad,
                    uncertainty,
                    featurizer,
                    feature_columns
                )

                st.session_state.polytg_result = result
                st.session_state.polytg_error = None

            except Exception as e:

                st.session_state.polytg_result = None

                st.session_state.polytg_error = str(e)

        st.rerun()
