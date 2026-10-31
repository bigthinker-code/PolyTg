
import numpy as np
import pandas as pd
import joblib

from functools import lru_cache
from huggingface_hub import hf_hub_download

from featurizer import get_featurizer
from polymetrix.featurizers.polymer import Polymer


MODEL_REPO = "Abdulroheem/PolyTg"


@lru_cache(maxsize=1)
def load_assets():
    """
    Load the trained model, applicability-domain configuration,
    uncertainty configuration, and polymer featurizer.

    Assets are loaded once and reused for subsequent predictions.
    """

    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="polymer_tg_extra_trees.pkl"
    )

    ad_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename="applicability_domain.pkl"
    )

    uncertainty_path = hf_hub_download(
        repo_id=MODEL_REPO,
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


def predict_tg(psmiles: str):
    """
    Predict the glass transition temperature of a polymer.

    Returns:
        Predicted Tg in Kelvin and Celsius,
        90% conformal interval,
        and applicability-domain assessment.
    """

    if not isinstance(psmiles, str) or not psmiles.strip():
        raise ValueError("Please provide a polymer SMILES string.")

    (
        model,
        ad,
        uncertainty,
        featurizer,
        feature_columns
    ) = load_assets()

    # Convert PSMILES into a polymer object.
    polymer = Polymer.from_psmiles(psmiles.strip())

    # Generate polymer descriptors.
    features = np.asarray(
        featurizer.featurize(polymer)
    )

    if features.ndim == 1:
        features = features.reshape(1, -1)

    generated_columns = list(feature_columns)
    expected_columns = list(ad["feature_columns"])

    # Validate feature count.
    if features.shape[1] != len(generated_columns):
        raise ValueError(
            "Generated feature count does not match "
            "the featurizer's feature labels."
        )

    # Validate feature names.
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

    # Reorder descriptors to match the trained model.
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

    # Predict Tg.
    prediction = float(model.predict(X)[0])

    # Calculate conformal prediction interval.
    margin = float(
        uncertainty["conformal_threshold"]
    )

    lower = prediction - margin
    upper = prediction + margin

    # Assess applicability domain.
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
        "psmiles": psmiles.strip(),
        "predicted_tg_kelvin": prediction,
        "predicted_tg_celsius": prediction - 273.15,
        "conformal_interval_90_kelvin": {
            "lower": lower,
            "upper": upper
        },
        "interval_width_kelvin": 2 * margin,
        "applicability_domain": {
            "nearest_neighbor_distance": distance,
            "threshold": threshold,
            "within_domain": within_domain,
            "status": (
                "within_domain"
                if within_domain
                else "outside_domain"
            )
        }
    }
