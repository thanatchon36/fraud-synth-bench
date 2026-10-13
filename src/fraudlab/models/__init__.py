"""Model zoo behind the common ``FraudModel`` interface.

To add a model: subclass :class:`FraudModel` (``fit``, ``predict_proba``;
``explain`` has a permutation-importance default), give it ``name`` and
``family``, and register it in ``MODELS`` below.
"""

from fraudlab.models.anomaly import AutoEncoder, IForest
from fraudlab.models.base import FEATURE_SETS, FraudModel, ModelData, build_model_data
from fraudlab.models.classical import LightGBMDefault, LightGBMTuned, LogReg, RandomForest
from fraudlab.models.ftt import FTTransformer
from fraudlab.models.graph import GraphSAGEModel
from fraudlab.models.semi import LightGBMSemi
from fraudlab.models.sequence import GRUModel
from fraudlab.models.tabpfn_model import TabPFNModel, tabpfn_available

MODELS: dict[str, type[FraudModel]] = {
    m.name: m
    for m in (
        LogReg,
        RandomForest,
        LightGBMDefault,
        LightGBMTuned,
        LightGBMSemi,
        IForest,
        AutoEncoder,
        GRUModel,
        GraphSAGEModel,
        FTTransformer,
        TabPFNModel,
    )
}

__all__ = [
    "FEATURE_SETS",
    "MODELS",
    "FraudModel",
    "ModelData",
    "build_model_data",
    "tabpfn_available",
]
