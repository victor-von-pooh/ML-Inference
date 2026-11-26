from importlib import metadata

from ml_inference.anomaly import make_anomaly_bundle
from ml_inference.baseline import (
    BASELINE_TYPES, check_baseline, current_from_baseline
)
from ml_inference.bundle import (
    BUNDLE_FORMAT_VERSION, FORMS, ROLES, Bundle, load_bundle, make_bundle,
    make_ready_bundle
)
from ml_inference.conclude import (
    NO_PREDICTION, RULE_KEYS, UNDECIDED, check_rules, conclude,
    conclude_outputs, origin_sentence, summary_sentences
)
from ml_inference.decision import (
    GUARD_COUNTS, GUARD_UNCOUNTED, backtest, check_decision, decide
)
from ml_inference.demand import make_demand_bundle
from ml_inference.explain import explain, explain_rows
from ml_inference.forecast import forecast
from ml_inference.grouping import (
    group_representatives, group_summary, make_group_bundle
)
from ml_inference.items import POINT_KINDS, backproject, describe_items
from ml_inference.policy import (
    POLICY_FORMAT_VERSION, GuardedPolicy, apply_guards, load_policy,
    read_policy_spec
)
from ml_inference.predict import (
    OK, SKIPPED, describe_bundle, predict, read_source, target_of
)
from ml_inference.price import make_price_bundle
from ml_inference.procedures import (
    PROCEDURES, get_procedure, make_generator, procedure_options,
    register_procedure, run_procedure, start_noise
)
from ml_inference.ranking import make_rank_bundle, rank_metrics
from ml_inference.ready import READY_LIBRARIES
from ml_inference.remeasure import remeasure, restore_uncertainty
from ml_inference.result import (
    RESULT_FORMAT_VERSION, artifact_path, infer, read_result, write_result
)
from ml_inference.reviews import read_reviews, record_review
from ml_inference.samples import make_sample_bundle
from ml_inference.search import make_search_bundle
from ml_inference.units import UnregisteredUnitError, register_units
from ml_inference.uplift import make_uplift_bundle

try:
    __version__ = metadata.version("ml-inference")
except metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "BASELINE_TYPES",
    "BUNDLE_FORMAT_VERSION",
    "Bundle",
    "FORMS",
    "GUARD_COUNTS",
    "GUARD_UNCOUNTED",
    "NO_PREDICTION",
    "GuardedPolicy",
    "OK",
    "POINT_KINDS",
    "POLICY_FORMAT_VERSION",
    "PROCEDURES",
    "READY_LIBRARIES",
    "RESULT_FORMAT_VERSION",
    "RULE_KEYS",
    "ROLES",
    "SKIPPED",
    "UNDECIDED",
    "UnregisteredUnitError",
    "apply_guards",
    "artifact_path",
    "backproject",
    "backtest",
    "check_baseline",
    "check_decision",
    "check_rules",
    "conclude",
    "conclude_outputs",
    "current_from_baseline",
    "decide",
    "describe_bundle",
    "describe_items",
    "explain",
    "explain_rows",
    "forecast",
    "get_procedure",
    "group_representatives",
    "group_summary",
    "infer",
    "load_bundle",
    "load_policy",
    "make_anomaly_bundle",
    "make_bundle",
    "make_demand_bundle",
    "make_generator",
    "make_group_bundle",
    "make_price_bundle",
    "make_rank_bundle",
    "make_ready_bundle",
    "make_sample_bundle",
    "make_search_bundle",
    "make_uplift_bundle",
    "origin_sentence",
    "predict",
    "procedure_options",
    "rank_metrics",
    "read_policy_spec",
    "read_result",
    "read_reviews",
    "read_source",
    "record_review",
    "register_procedure",
    "register_units",
    "remeasure",
    "restore_uncertainty",
    "run_procedure",
    "start_noise",
    "summary_sentences",
    "target_of",
    "write_result",
]
