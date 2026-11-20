from importlib import metadata

from ml_inference.baseline import (
    BASELINE_TYPES, check_baseline, current_from_baseline
)
from ml_inference.bundle import (
    BUNDLE_FORMAT_VERSION, FORMS, ROLES, Bundle, load_bundle, make_bundle
)
from ml_inference.conclude import (
    NO_PREDICTION, UNDECIDED, check_rules, conclude
)
from ml_inference.decision import backtest, check_decision, decide
from ml_inference.explain import explain
from ml_inference.forecast import forecast
from ml_inference.predict import OK, SKIPPED, describe_bundle, predict
from ml_inference.procedures import (
    PROCEDURES, get_procedure, make_generator, register_procedure,
    run_procedure, start_noise
)
from ml_inference.remeasure import remeasure, restore_uncertainty
from ml_inference.result import (
    RESULT_FORMAT_VERSION, artifact_path, infer, read_result, write_result
)
from ml_inference.samples import make_sample_bundle
from ml_inference.units import UnregisteredUnitError, register_units

try:
    __version__ = metadata.version("ml-inference")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "BASELINE_TYPES",
    "BUNDLE_FORMAT_VERSION",
    "Bundle",
    "FORMS",
    "NO_PREDICTION",
    "OK",
    "PROCEDURES",
    "RESULT_FORMAT_VERSION",
    "ROLES",
    "SKIPPED",
    "UNDECIDED",
    "UnregisteredUnitError",
    "artifact_path",
    "backtest",
    "check_baseline",
    "check_decision",
    "check_rules",
    "conclude",
    "current_from_baseline",
    "decide",
    "describe_bundle",
    "explain",
    "forecast",
    "get_procedure",
    "infer",
    "load_bundle",
    "make_bundle",
    "make_generator",
    "make_sample_bundle",
    "predict",
    "read_result",
    "register_procedure",
    "register_units",
    "remeasure",
    "restore_uncertainty",
    "run_procedure",
    "start_noise",
    "write_result",
]
