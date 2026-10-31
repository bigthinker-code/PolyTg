import os
import numpy as np
from polymetrix.featurizers.base_featurizer import BaseFeatureCalculator
import logging
from typing import List, Optional
import pandas as pd

from polymetrix.featurizers.polymer import Polymer
from polymetrix.featurizers.chemical_featurizer import (
    NumHBondDonors,
    NumHBondAcceptors,
    NumRotatableBonds,
    NumRings,
    NumNonAromaticRings,
    NumAromaticRings,
    NumAtoms,
    TopologicalSurfaceArea,
    FractionBicyclicRings,
    NumAliphaticHeterocycles,
    SlogPVSA1,
    BalabanJIndex,
    MolecularWeight,
    Sp3CarbonCountFeaturizer,
    Sp2CarbonCountFeaturizer,
    MaxEStateIndex,
    SmrVSA5,
    FpDensityMorgan1,
    BondCounts,
    BridgingRingsCount,
    MaxRingSize,
    HeteroatomCount,
    HeteroatomDensity,
)
from polymetrix.featurizers.sidechain_backbone_featurizer import (
    SideChainFeaturizer,
    NumSideChainFeaturizer,
    BackBoneFeaturizer,
    NumBackBoneFeaturizer,
    FullPolymerFeaturizer,
    SidechainLengthToStarAttachmentDistanceRatioFeaturizer,
    StarToSidechainMinDistanceFeaturizer,
    SidechainDiversityFeaturizer,
)
from polymetrix.featurizers.multiple_featurizer import MultipleFeaturizer


class DatasetHalogenCounts(BaseFeatureCalculator):
    def calculate(self, mol, sanitize=True):
        self._sanitize(mol, sanitize)
        counts = {9: 0, 17: 0, 35: 0}

        for atom in mol.GetAtoms():
            atomic_num = atom.GetAtomicNum()
            if atomic_num in counts:
                counts[atomic_num] += 1

        return np.array([
            sum(counts.values()),
            counts[9],
            counts[17],
            counts[35]
        ])

    def feature_base_labels(self):
        return [
            "total_halogens",
            "fluorine_count",
            "chlorine_count",
            "bromine_count"
        ]


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

DEFAULT_META_COLUMNS = [
    "polymer",
    "source",
    "tg_range",
    "tg_values",
    "num_of_points",
    "std",
    "reliability",
]

DEFAULT_LABEL_COLUMNS = ["Exp_Tg(K)"]


FULL_POLYMER_FEATURIZERS = [
    (FullPolymerFeaturizer, BalabanJIndex, {}),
    (FullPolymerFeaturizer, NumHBondDonors, {}),
    (FullPolymerFeaturizer, NumHBondAcceptors, {}),
    (FullPolymerFeaturizer, NumRotatableBonds, {}),
    (FullPolymerFeaturizer, NumRings, {}),
    (FullPolymerFeaturizer, NumNonAromaticRings, {}),
    (FullPolymerFeaturizer, NumAromaticRings, {}),
    (FullPolymerFeaturizer, TopologicalSurfaceArea, {}),
    (FullPolymerFeaturizer, FractionBicyclicRings, {}),
    (FullPolymerFeaturizer, NumAliphaticHeterocycles, {}),
    (FullPolymerFeaturizer, SlogPVSA1, {}),
    (FullPolymerFeaturizer, MolecularWeight, {}),
    (FullPolymerFeaturizer, Sp3CarbonCountFeaturizer, {}),
    (FullPolymerFeaturizer, Sp2CarbonCountFeaturizer, {}),
    (FullPolymerFeaturizer, MaxEStateIndex, {}),
    (FullPolymerFeaturizer, SmrVSA5, {}),
    (FullPolymerFeaturizer, FpDensityMorgan1, {}),
    (FullPolymerFeaturizer, DatasetHalogenCounts, {}),
    (FullPolymerFeaturizer, BondCounts, {}),
    (FullPolymerFeaturizer, BridgingRingsCount, {}),
    (FullPolymerFeaturizer, MaxRingSize, {}),
    (FullPolymerFeaturizer, HeteroatomDensity, {}),
    (FullPolymerFeaturizer, HeteroatomCount, {}),
]

BACKBONE_LEVEL_FEATURIZERS = [
    (BackBoneFeaturizer, NumAtoms, {}),
    (NumBackBoneFeaturizer, None, {}),
    (BackBoneFeaturizer, BalabanJIndex, {}),
    (BackBoneFeaturizer, NumHBondDonors, {}),
    (BackBoneFeaturizer, NumHBondAcceptors, {}),
    (BackBoneFeaturizer, NumRotatableBonds, {}),
    (BackBoneFeaturizer, NumRings, {}),
    (BackBoneFeaturizer, NumNonAromaticRings, {}),
    (BackBoneFeaturizer, NumAromaticRings, {}),
    (BackBoneFeaturizer, TopologicalSurfaceArea, {}),
    (BackBoneFeaturizer, FractionBicyclicRings, {}),
    (BackBoneFeaturizer, NumAliphaticHeterocycles, {}),
    (BackBoneFeaturizer, SlogPVSA1, {}),
    (BackBoneFeaturizer, MolecularWeight, {}),
    (BackBoneFeaturizer, Sp3CarbonCountFeaturizer, {}),
    (BackBoneFeaturizer, Sp2CarbonCountFeaturizer, {}),
    (BackBoneFeaturizer, MaxEStateIndex, {}),
    (BackBoneFeaturizer, SmrVSA5, {}),
    (BackBoneFeaturizer, FpDensityMorgan1, {}),
    (BackBoneFeaturizer, DatasetHalogenCounts, {}),
    (BackBoneFeaturizer, BondCounts, {}),
    (BackBoneFeaturizer, BridgingRingsCount, {}),
    (BackBoneFeaturizer, MaxRingSize, {}),
    (BackBoneFeaturizer, HeteroatomDensity, {}),
    (BackBoneFeaturizer, HeteroatomCount, {}),
]

SIDECHAIN_LEVEL_FEATURIZERS = [
    (SideChainFeaturizer, NumAtoms, {"agg": ["sum", "mean", "max", "min"]}),
    (NumSideChainFeaturizer, None, {}),
    (
        SidechainLengthToStarAttachmentDistanceRatioFeaturizer,
        None,
        {"agg": ["mean", "min", "max", "sum"]},
    ),
    (
        StarToSidechainMinDistanceFeaturizer,
        None,
        {"agg": ["mean", "min", "max", "sum"]},
    ),
    (SidechainDiversityFeaturizer, None, {}),
    (SideChainFeaturizer, BalabanJIndex, {}),
    (SideChainFeaturizer, NumHBondDonors, {}),
    (SideChainFeaturizer, NumHBondAcceptors, {}),
    (SideChainFeaturizer, NumRotatableBonds, {}),
    (SideChainFeaturizer, NumRings, {}),
    (SideChainFeaturizer, NumNonAromaticRings, {}),
    (SideChainFeaturizer, NumAromaticRings, {}),
    (SideChainFeaturizer, TopologicalSurfaceArea, {}),
    (SideChainFeaturizer, FractionBicyclicRings, {}),
    (SideChainFeaturizer, NumAliphaticHeterocycles, {}),
    (SideChainFeaturizer, SlogPVSA1, {}),
    (SideChainFeaturizer, MolecularWeight, {}),
    (SideChainFeaturizer, Sp3CarbonCountFeaturizer, {}),
    (SideChainFeaturizer, Sp2CarbonCountFeaturizer, {}),
    (SideChainFeaturizer, MaxEStateIndex, {}),
    (SideChainFeaturizer, SmrVSA5, {}),
    (SideChainFeaturizer, FpDensityMorgan1, {}),
    (SideChainFeaturizer, DatasetHalogenCounts, {}),
    (SideChainFeaturizer, BondCounts, {}),
    (SideChainFeaturizer, BridgingRingsCount, {}),
    (SideChainFeaturizer, MaxRingSize, {}),
    (SideChainFeaturizer, HeteroatomDensity, {}),
    (SideChainFeaturizer, HeteroatomCount, {}),
]

def create_featurizer_set(featurizer_config: list) -> List[object]:
    """Instantiate featurizers from configuration tuples."""
    featurizers = []
    for container_cls, feature_cls, params in featurizer_config:
        if feature_cls is None:
            featurizers.append(container_cls(**params))
        else:
            inner_feat = feature_cls(**params)
            featurizers.append(container_cls(inner_feat))
    return featurizers

def get_featurizer() -> tuple:
    """Create and configure the MultipleFeaturizer with prefixed feature labels."""
    featurizers = []
    
    # Define feature categories
    feature_categories = {
        "fullpolymerlevel.features": FULL_POLYMER_FEATURIZERS,
        "backbonelevel.features": BACKBONE_LEVEL_FEATURIZERS,
        "sidechainlevel.features": SIDECHAIN_LEVEL_FEATURIZERS
    }

    feature_labels = []

    for prefix, feature_set in feature_categories.items():
        featurizer_list = create_featurizer_set(feature_set)
        featurizers.extend(featurizer_list)
        feature_labels.extend(
            [f"{prefix}.{label}" for f in featurizer_list for label in f.feature_labels()]
        )

    return MultipleFeaturizer(featurizers), feature_labels

def compute_features(psmiles: str, featurizer: MultipleFeaturizer) -> pd.Series:
    """Compute features for a single polymer SMILES string."""
    try:
        polymer = Polymer.from_psmiles(psmiles)
        return pd.Series(featurizer.featurize(polymer))
    except Exception as e:
        logger.error(f"Error processing {psmiles}: {str(e)}")
        return pd.Series([None] * len(featurizer.feature_labels()))

def process_dataframe(
    df: pd.DataFrame,
    psmiles_column: str,
    meta_columns: List[str],
    label_columns: List[str],
    output_path: str,
) -> None:
    """Process DataFrame and save results."""
    featurizer, feature_labels = get_featurizer()

    pandarallel.initialize(progress_bar=True)

    features_df = df[psmiles_column].parallel_apply(
        lambda x: compute_features(x, featurizer)
    )
    features_df.columns = feature_labels

    column_renames = {}

    # Check for and handle missing columns
    for col_type, columns in {"meta": meta_columns, "label": label_columns}.items():
        missing_cols = [col for col in columns if col not in df.columns]
        if missing_cols:
            logger.warning(f"Warning: Missing {col_type} columns: {missing_cols}. These will be filled with NaN.")
            for col in missing_cols:
                df[col] = None 

    # Rename existing meta and label columns
    for col in meta_columns + label_columns:
        if col in df.columns:
            column_renames[col] = f"{'meta' if col in meta_columns else 'labels'}.{col}"

    df.rename(columns=column_renames, inplace=True)

    # Combine with computed features
    result_df = pd.concat([df, features_df], axis=1)
    result_df.to_csv(output_path, index=False)
    
    logger.info(f"Saved processed data to {output_path}")

def main(
    input_path: str,
    output_path: str,
    psmiles_column: str = "PSMILES",
    meta_columns: Optional[List[str]] = None,
    label_columns: Optional[List[str]] = None,
) -> None:
    """Main processing pipeline."""
    if not os.path.exists(input_path):
        logger.error(f"Input file not found: {input_path}")
        return

    meta_cols = meta_columns or DEFAULT_META_COLUMNS
    label_cols = label_columns or DEFAULT_LABEL_COLUMNS

    df = pd.read_csv(input_path)

    # Handle missing PSMILES column
    if psmiles_column not in df.columns:
        logger.error(f"Missing required column: {psmiles_column}")
        return

    process_dataframe(
        df=df,
        psmiles_column=psmiles_column,
        meta_columns=meta_cols,
        label_columns=label_cols,
        output_path=output_path,
    )



def get_featurizer():
    configs = [
        ("backbonelevel.features", BACKBONE_LEVEL_FEATURIZERS),
        ("sidechainlevel.features", SIDECHAIN_LEVEL_FEATURIZERS),
        ("fullpolymerlevel.features", FULL_POLYMER_FEATURIZERS)
    ]

    featurizers = []
    feature_labels = []

    for prefix, config in configs:
        items = create_featurizer_set(config)
        featurizers.extend(items)

        for item in items:
            feature_labels.extend(
                [f"{prefix}.{label}" for label in item.feature_labels()]
            )

    return MultipleFeaturizer(featurizers), feature_labels
