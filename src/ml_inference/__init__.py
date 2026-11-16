from importlib import metadata

from ml_inference.bundle import (
    BUNDLE_FORMAT_VERSION, Bundle, load_bundle, make_bundle
)
from ml_inference.conclude import (
    NO_PREDICTION, UNDECIDED, check_rules, conclude
)
from ml_inference.explain import explain
from ml_inference.forecast import forecast
from ml_inference.predict import OK, SKIPPED, describe_bundle, predict
from ml_inference.result import infer, write_result

try:
    __version__ = metadata.version("ml-inference")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "BUNDLE_FORMAT_VERSION",
    "Bundle",
    "NO_PREDICTION",
    "OK",
    "SKIPPED",
    "UNDECIDED",
    "check_rules",
    "conclude",
    "describe_bundle",
    "explain",
    "forecast",
    "infer",
    "load_bundle",
    "make_bundle",
    "predict",
    "write_result",
]
