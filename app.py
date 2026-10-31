import streamlit as st
import pandas as pd
import numpy as np
import joblib

from huggingface_hub import hf_hub_download
from featurizer import get_featurizer
from polymetrix.featurizers.polymer import Polymer


st.set_page_config(
    page_title="Polymer Tg Predictor",
    page_icon="🧪",
    layout="wide"
)

st.title("Polymer Property Prediction AI")

st.write(
    "Predict the glass transition temperature (Tg) "
    "of polymers from their polymer SMILES."
)


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


try:
    (
        model,
        ad,
        uncertainty,
        featurizer,
        feature_columns
    ) = load_assets()

except Exception as e:
    import traceback
    st.error(f"Prediction failed: {e}")
    st.code(traceback.format_exc())
    st.stop()


st.subheader("Enter polymer structure")

psmiles = st.text_input(
    "Polymer SMILES (PSMILES)",
    placeholder="Enter a valid polymer SMILES"
)


if st.button("Predict Tg", type="primary"):

    if not psmiles.strip():

        st.warning("Please enter a polymer SMILES.")

    else:

        try:

            polymer = Polymer.from_psmiles(psmiles)

            features = np.asarray(
                featurizer.featurize(polymer)
            )

            if features.ndim == 1:
                features = features.reshape(1, -1)

            if features.shape[1] != len(feature_columns):
                raise ValueError(
                    "Generated feature count does not match "
                    "the expected feature count."
                )

            expected_columns = ad["feature_columns"]

            if list(feature_columns) != list(expected_columns):
                raise ValueError(
                    "Feature order does not match the training model."
                )

            X = pd.DataFrame(
                features,
                columns=feature_columns
            )

            prediction = float(model.predict(X)[0])

            margin = float(
                uncertainty["conformal_threshold"]
            )

            lower = prediction - margin
            upper = prediction + margin

            st.subheader("Prediction")

            col1, col2 = st.columns(2)

            with col1:

                st.metric(
                    "Predicted Tg (Kelvin)",
                    f"{prediction:.2f} K"
                )

                st.metric(
                    "Predicted Tg (Celsius)",
                    f"{prediction - 273.15:.2f} °C"
                )

            with col2:

                st.metric(
                    "90% Conformal Interval",
                    f"{lower:.2f} – {upper:.2f} K"
                )

                st.metric(
                    "Interval Width",
                    f"{2 * margin:.2f} K"
                )

            scaled = ad["scaler"].transform(X)

            distance = float(
                ad["nearest_neighbors"].kneighbors(
                    scaled,
                    n_neighbors=1
                )[0][0][0]
            )

            threshold = float(ad["ad_threshold"])

            st.subheader("Applicability Domain")

            st.write(
                f"Nearest-neighbour distance: {distance:.3f}"
            )

            st.write(
                f"Applicability-domain threshold: {threshold:.3f}"
            )

            if distance <= threshold:

                st.success(
                    "Within the model's applicability domain."
                )

            else:

                st.warning(
                    "Outside the model's applicability domain. "
                    "Treat this prediction with extra caution."
                )

            st.caption(
                "The conformal interval is based on calibration data "
                "and provides marginal coverage under exchangeability. "
                "It is not a guarantee for an individual polymer. "
                "Applicability-domain status is a caution flag, "
                "not a probability."
            )

        except Exception as e:

            st.error(f"Prediction failed: {e}")

            st.info(
                "Check that the PSMILES is valid and supported "
                "by the featurizer."
            )


st.divider()

st.caption(
    "Research prototype | ExtraTrees | 100 polymer descriptors"
)
