"""Load and persist hallucination-detector rules from the database."""

from __future__ import annotations

import logging

from app.schemas.settings import HallucinationRuleSetting, HallucinationRuleUpdate
from stt_hallucination import (
    HALLUCINATION_RULE_SPECS,
    HALLUCINATION_RULE_SPECS_BY_ID,
    HallucinationConfig,
    default_hallucination_config,
    hallucination_config_from_rows,
)

log = logging.getLogger(__name__)


def _get_db():
    from app.core.config import settings as _config
    from database import DatabaseManager

    return DatabaseManager(_config.database_url)


def get_hallucination_config() -> HallucinationConfig:
    """Return runtime detector config from the DB, falling back to factory defaults."""
    try:
        db = _get_db()
        rows = db.list_hallucination_rules()
    except Exception:
        log.warning(
            "Could not read hallucination_rules from the database; "
            "using factory defaults",
            exc_info=True,
        )
        return default_hallucination_config()
    if not rows:
        log.warning(
            "hallucination_rules table is empty after init; using factory defaults"
        )
        return default_hallucination_config()
    return hallucination_config_from_rows(rows)


def list_hallucination_rule_settings() -> list[HallucinationRuleSetting]:
    """Catalog + stored enable/threshold values for the settings API."""
    config = get_hallucination_config()
    result: list[HallucinationRuleSetting] = []
    for spec in HALLUCINATION_RULE_SPECS:
        state = config.rules.get(spec.id)
        enabled = True if state is None else state.enabled
        threshold = spec.default_threshold
        if state is not None and state.threshold is not None:
            threshold = state.threshold
        result.append(
            HallucinationRuleSetting(
                id=spec.id,
                label=spec.label,
                description=spec.description,
                enabled=enabled,
                threshold=threshold,
                default_threshold=spec.default_threshold,
                threshold_min=spec.threshold_min,
                threshold_max=spec.threshold_max,
                threshold_step=spec.threshold_step,
                unit=spec.unit,
                comparison=spec.comparison,
            )
        )
    return result


def save_hallucination_rules(updates: list[HallucinationRuleUpdate]) -> None:
    """Validate and persist enable flags and thresholds for known rules.

    Raises:
        ValueError: Unknown rule id or threshold outside the allowed range.
    """
    if not updates:
        return

    current = {row["rule_id"]: row for row in _get_db().list_hallucination_rules()}
    rows: list[dict] = []
    for item in updates:
        spec = HALLUCINATION_RULE_SPECS_BY_ID.get(item.id)
        if spec is None:
            raise ValueError(f"Unknown hallucination rule: {item.id}")

        previous = current.get(
            item.id,
            {"enabled": 1, "threshold": spec.default_threshold},
        )
        enabled = previous.get("enabled", True) if item.enabled is None else item.enabled

        if spec.default_threshold is None:
            threshold = None
        elif item.threshold is None:
            threshold = previous.get("threshold", spec.default_threshold)
            if threshold is None:
                threshold = spec.default_threshold
        else:
            threshold = float(item.threshold)
            if spec.threshold_min is not None and threshold < spec.threshold_min:
                raise ValueError(
                    f"{spec.id} threshold {threshold} is below minimum {spec.threshold_min}"
                )
            if spec.threshold_max is not None and threshold > spec.threshold_max:
                raise ValueError(
                    f"{spec.id} threshold {threshold} is above maximum {spec.threshold_max}"
                )

        rows.append(
            {
                "rule_id": item.id,
                "enabled": bool(enabled),
                "threshold": threshold,
            }
        )

    _get_db().upsert_hallucination_rules(rows)
