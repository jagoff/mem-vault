"""Closed-loop adaptive ranker — fits a logistic regression on search-event
telemetry, then re-scores ``memory_search`` candidates with the learned model.

Architecture (v0.6.0 closed-loop):

    memory_search → record one row per hit to telemetry.search_events
                       │
                       ▼
                  Stop hook detects citations
                       │
                       ▼
              flips ``was_cited=1`` on rows
                       │
                       ▼
    ``mem-vault ranker train`` reads the rows, fits a logistic regression
    over [score_dense, score_bm25_norm, helpful_ratio, recency_days,
    usage_count_log, project_match, agent_id_match, rank], saves the
    pickle + metrics to ``state_dir/ranker_v{N}.pkl``.

    next memory_search loads the latest pickle, calls ``predict_proba`` on
    each candidate's feature vector, and re-sorts by the learned score
    (multiplied by the existing usage_boost so positive feedback compounds).

Why logistic regression and not a neural net:

- The dataset is small (167 memorias × ~few searches/day = O(1k) rows
  per month). A 2-layer MLP would overfit immediately.
- Inference must run inline on every search (target <2 ms per hit).
  Logreg is one dot product; an ONNX runtime is a 50 MB dep we don't
  need.
- The features are already engineered (we know which signals matter
  from the existing usage_boost heuristic). What we want is the
  optimal *weighting* — that's what logreg gives us.
- Calibrated probabilities out of the box (``predict_proba``), so we
  can blend with the existing ``score`` cleanly instead of comparing
  apples to oranges.

Feature flag: ``MEM_VAULT_LEARNED_RANKER=1`` enables inference at
search time. Default OFF so existing users with no training data get
the same behavior as v0.5.0.

Kill-switch: every train run writes a metric ("auc", "n_train",
"n_pos", "n_neg") next to the pickle. ``mem-vault ranker eval`` runs
the harness in ``eval.py`` against both ``ranker_v{N}.pkl`` and
``ranker_v{N-1}.pkl``; if hit@5 dropped, the user can manually
``mem-vault ranker rollback``.

Optional dependency: ``[learning]`` extra installs scikit-learn +
numpy. Without them, every public function returns a graceful no-op
("ranker disabled").
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from mem_vault import telemetry

logger = logging.getLogger(__name__)

# State-dir layout. The active pickle is what ``server.search`` reads
# at startup (or on first inference); train writes a new versioned
# file + atomically swaps the symlink.
RANKER_DIR_NAME = "ranker"
ACTIVE_FILENAME = "active.pkl"
META_FILENAME = "meta.json"


# ---------------------------------------------------------------------------
# Feature engineering — pure helpers, importable without sklearn.
# ---------------------------------------------------------------------------

#: Feature column order. The model is fit + queried in this exact order;
#: changes here are breaking (require retrain).
FEATURE_COLUMNS = (
    "score_dense",        # raw bi-encoder score (or post-rerank if available)
    "score_final",        # what the user saw (post-boost). 0 if missing.
    "rank",               # zero-based position in the response.
    "helpful_ratio",      # supervised feedback signal at search-time.
    "usage_count_log",    # log1p(usage_count) — diminishing returns on popularity.
    "recency_days",       # capped at 365 — older isn't more "old".
    "project_match",      # 0/1.
    "agent_id_match",     # 0/1.
)


def featurize(row: dict[str, Any]) -> list[float]:
    """Project a search-event row to the model's feature vector.

    Robust to None columns: missing values become 0 for scores / 365 for
    recency / 0 for binary flags. The choice mirrors the search path:
    "no signal" should never bias a memory positively.
    """
    import math

    def _f(v: Any, default: float = 0.0) -> float:
        if v is None:
            return default
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    usage = _f(row.get("usage_count"), default=0.0)
    return [
        _f(row.get("score_dense")),
        _f(row.get("score_final")),
        _f(row.get("rank")),
        _f(row.get("helpful_ratio")),
        math.log1p(max(0.0, usage)),
        min(_f(row.get("recency_days"), default=365.0), 365.0),
        _f(row.get("project_match")),
        _f(row.get("agent_id_match")),
    ]


# ---------------------------------------------------------------------------
# Trained-model wrapper — instances live in memory; pickles on disk.
# ---------------------------------------------------------------------------


@dataclass
class TrainedRanker:
    """A fitted scikit-learn classifier + the metadata to evaluate it.

    The pickle stores the dataclass directly; ``load`` returns the
    instance ready to call ``score()`` per candidate at search-time.
    """

    model: Any                              # sklearn.linear_model.LogisticRegression
    feature_columns: tuple[str, ...]
    feature_means: list[float]              # for missing-feature imputation at predict time
    feature_stds: list[float]               # same; used to z-normalize live features
    n_train: int
    n_positive: int
    n_negative: int
    auc: float | None
    trained_at: float = field(default_factory=time.time)
    schema_version: int = telemetry.SCHEMA_VERSION

    def score(self, row: dict[str, Any]) -> float:
        """Return the predicted probability that ``row`` would be cited.

        Result is in [0, 1]. The search path multiplies this by the
        existing pre-rank score (and the usage_boost factor) so the
        ranker composes with rather than replaces the heuristic.
        """
        x = featurize(row)
        # z-normalize the same way training did
        z = [
            (xi - mu) / (sigma if sigma > 1e-9 else 1.0)
            for xi, mu, sigma in zip(x, self.feature_means, self.feature_stds, strict=True)
        ]
        # ``predict_proba`` returns shape (1, 2): [[P(class=0), P(class=1)]].
        # Class 1 = "was cited", which is what we want to maximize.
        return float(self.model.predict_proba([z])[0][1])


# ---------------------------------------------------------------------------
# Disk layout helpers.
# ---------------------------------------------------------------------------


def ranker_dir(state_dir: Path) -> Path:
    return Path(state_dir) / RANKER_DIR_NAME


def active_pickle_path(state_dir: Path) -> Path:
    return ranker_dir(state_dir) / ACTIVE_FILENAME


def meta_path(state_dir: Path) -> Path:
    return ranker_dir(state_dir) / META_FILENAME


def _versioned_pickle(state_dir: Path, version: int) -> Path:
    return ranker_dir(state_dir) / f"ranker_v{version}.pkl"


def _next_version(state_dir: Path) -> int:
    rd = ranker_dir(state_dir)
    if not rd.exists():
        return 1
    versions = []
    for p in rd.glob("ranker_v*.pkl"):
        try:
            versions.append(int(p.stem.split("_v", 1)[1]))
        except (IndexError, ValueError):
            continue
    return (max(versions) + 1) if versions else 1


# ---------------------------------------------------------------------------
# Training.
# ---------------------------------------------------------------------------


@dataclass
class TrainResult:
    """What ``train()`` returns; serialized into ``meta.json`` for inspection."""

    version: int
    n_train: int
    n_positive: int
    n_negative: int
    auc: float | None
    pickle_path: str
    trained_at: float
    feature_columns: tuple[str, ...]
    notes: str | None = None


def train(
    state_dir: Path,
    *,
    min_rows: int = 50,
    min_positives: int = 5,
    test_size: float = 0.2,
    random_state: int = 1337,
) -> TrainResult:
    """Fit a logistic regression on the recorded search events.

    Args:
        state_dir: where ``search_events.db`` + the ranker pickles live.
        min_rows: refuse to train below this row count — too noisy.
        min_positives: refuse below this many ``was_cited=1`` rows —
            without positive examples the classifier collapses.
        test_size: held-out fraction for the AUC report.
        random_state: deterministic split for reproducibility.

    Raises:
        RuntimeError: when the [learning] extra isn't installed.
        ValueError: when there isn't enough data to fit.
    """
    try:
        import numpy as np  # noqa: F401  (used implicitly by sklearn)
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise RuntimeError(
            "scikit-learn not installed. Install the [learning] extra: "
            "uv tool install --editable '.[learning]'"
        ) from exc

    rows = telemetry.fetch_training_rows(state_dir)
    n_rows = len(rows)
    n_pos = sum(1 for r in rows if r.get("was_cited"))
    n_neg = n_rows - n_pos

    if n_rows < min_rows:
        raise ValueError(
            f"not enough training data: {n_rows} rows < min_rows={min_rows}. "
            "Use mem-vault for a few sessions to gather signal first."
        )
    if n_pos < min_positives:
        raise ValueError(
            f"not enough positive examples: {n_pos} citations < "
            f"min_positives={min_positives}. The Stop hook needs to "
            "have observed at least that many cited memorias before "
            "training is meaningful."
        )

    X = [featurize(r) for r in rows]
    y = [int(r.get("was_cited") or 0) for r in rows]

    # Standardize: keep means + stds so we can normalize live features
    # the same way at inference. We do it manually rather than wrap a
    # ``Pipeline`` to keep the pickle dependency-light at load time.
    means = [sum(col) / len(col) for col in zip(*X, strict=True)]
    stds = [
        max(
            1e-9,
            (sum((v - mu) ** 2 for v in col) / len(col)) ** 0.5,
        )
        for col, mu in zip(zip(*X, strict=True), means, strict=True)
    ]
    X_norm = [
        [(xi - mu) / sigma for xi, mu, sigma in zip(x, means, stds, strict=True)]
        for x in X
    ]

    # Held-out AUC. With small datasets stratify keeps the positive class
    # represented in both splits.
    auc: float | None = None
    if min(n_pos, n_neg) >= 2 and n_rows >= 10:
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                X_norm,
                y,
                test_size=test_size,
                random_state=random_state,
                stratify=y,
            )
            probe = LogisticRegression(
                max_iter=200,
                class_weight="balanced",
                random_state=random_state,
            )
            probe.fit(X_tr, y_tr)
            preds = probe.predict_proba(X_te)[:, 1]
            auc = float(roc_auc_score(y_te, preds))
        except Exception as exc:
            logger.warning("ranker.train: AUC computation failed: %s", exc)
            auc = None

    # Final fit on the full dataset (we want every row's signal in production).
    model = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=random_state,
    )
    model.fit(X_norm, y)

    fitted = TrainedRanker(
        model=model,
        feature_columns=FEATURE_COLUMNS,
        feature_means=means,
        feature_stds=stds,
        n_train=n_rows,
        n_positive=n_pos,
        n_negative=n_neg,
        auc=auc,
    )

    rd = ranker_dir(state_dir)
    rd.mkdir(parents=True, exist_ok=True)

    version = _next_version(state_dir)
    versioned = _versioned_pickle(state_dir, version)
    with versioned.open("wb") as f:
        pickle.dump(fitted, f, protocol=pickle.HIGHEST_PROTOCOL)

    # Atomic swap of the active pointer: write a temp file with the same
    # contents and ``os.replace`` it. Avoids a half-written ``active.pkl``
    # if the process gets killed mid-write.
    active = active_pickle_path(state_dir)
    tmp = active.with_suffix(active.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(fitted, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, active)

    result = TrainResult(
        version=version,
        n_train=n_rows,
        n_positive=n_pos,
        n_negative=n_neg,
        auc=auc,
        pickle_path=str(versioned),
        trained_at=fitted.trained_at,
        feature_columns=FEATURE_COLUMNS,
    )
    meta_path(state_dir).write_text(
        json.dumps(
            {
                **result.__dict__,
                "feature_columns": list(result.feature_columns),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    return result


def load_active(state_dir: Path) -> TrainedRanker | None:
    """Load the currently-active ranker, if any.

    Returns None on any failure (no pickle yet, corrupt pickle, sklearn
    missing, etc.) — the search path falls back to the heuristic boost.
    """
    path = active_pickle_path(state_dir)
    if not path.exists():
        return None
    try:
        # Lazy import: don't pay the sklearn import cost just to discover
        # there's no pickle. Loading the pickle itself implicitly
        # imports sklearn (the model class is referenced inside).
        import sklearn  # noqa: F401
    except ImportError:
        return None
    try:
        with path.open("rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, TrainedRanker):
            return obj
        logger.warning("ranker.load_active: pickle is not a TrainedRanker")
    except Exception as exc:
        logger.warning("ranker.load_active: failed to load pickle: %s", exc)
    return None


def rollback(state_dir: Path) -> int | None:
    """Demote the active pickle and promote the previous version.

    Returns the version now active, or None if there's no prior version
    to fall back to (ranker remains disabled).
    """
    rd = ranker_dir(state_dir)
    if not rd.exists():
        return None
    versions = sorted(
        (
            int(p.stem.split("_v", 1)[1])
            for p in rd.glob("ranker_v*.pkl")
            if p.stem.split("_v", 1)[1].isdigit()
        ),
        reverse=True,
    )
    if len(versions) < 2:
        return None
    target = versions[1]
    src = _versioned_pickle(state_dir, target)
    dst = active_pickle_path(state_dir)
    tmp = dst.with_suffix(dst.suffix + ".tmp")
    with src.open("rb") as f_in, tmp.open("wb") as f_out:
        f_out.write(f_in.read())
    os.replace(tmp, dst)
    return target


def is_enabled() -> bool:
    """Whether ``server.search`` should call into the ranker on each hit.

    Cheap env-var check, evaluated on every search to allow toggling
    without a restart.
    """
    return os.environ.get("MEM_VAULT_LEARNED_RANKER", "0").lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


__all__ = [
    "FEATURE_COLUMNS",
    "TrainResult",
    "TrainedRanker",
    "active_pickle_path",
    "featurize",
    "is_enabled",
    "load_active",
    "meta_path",
    "ranker_dir",
    "rollback",
    "train",
]
