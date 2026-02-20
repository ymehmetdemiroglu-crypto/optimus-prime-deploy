"""
Adaptive Threshold Manager — Per-Account Intent Threshold Profiles

Provides a configurable layer on top of the global INTENT_THRESHOLDS
so individual accounts can tune bleed detection, opportunity discovery,
and keyword graduation sensitivity per intent type.

Flow:
  1. Global defaults live in intent_classifier.INTENT_THRESHOLDS
  2. Account overrides are stored in-memory (and optionally persisted to DB)
  3. get_thresholds(intent, account_id) merges both, account wins

This lets an account that sells impulse-buy products use tighter
transactional thresholds, while a brand-building account loosens
Rufus/discovery thresholds even further.
"""
import logging
from copy import deepcopy
from typing import Dict, Optional

from app.services.ml.intent_classifier import (
    ShoppingIntent,
    INTENT_THRESHOLDS,
)

logger = logging.getLogger(__name__)

# Allowed threshold keys and their valid ranges
THRESHOLD_KEYS = {
    "bleed_threshold":              (0.01, 0.95),
    "opportunity_floor":            (0.01, 0.95),
    "min_orders_to_graduate":       (1, 50),
    "prob_acos_threshold":          (0.10, 0.99),
    "min_clicks_to_negate":         (3, 100),
    "acos_ceiling_for_negate":      (0.10, 2.0),
    "min_spend_for_negate_by_acos": (1.0, 100.0),
}


class AdaptiveThresholdManager:
    """
    Manages per-account threshold overrides on top of global defaults.

    Usage:
        manager = AdaptiveThresholdManager()
        # Get thresholds (merges global + account overrides)
        t = manager.get_thresholds(ShoppingIntent.INFORMATIONAL_RUFUS, account_id=42)

        # Set per-account override
        manager.set_override(42, ShoppingIntent.INFORMATIONAL_RUFUS, "bleed_threshold", 0.25)

        # View all overrides for an account
        overrides = manager.get_account_overrides(42)

        # Reset an account to global defaults
        manager.reset_account(42)
    """

    def __init__(self):
        # account_id -> intent -> key -> value
        self._overrides: Dict[int, Dict[ShoppingIntent, Dict[str, float]]] = {}

    def get_thresholds(
        self,
        intent: ShoppingIntent,
        account_id: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Return merged thresholds for (intent, account_id).
        Account overrides take precedence over global defaults.
        """
        base = deepcopy(
            INTENT_THRESHOLDS.get(intent, INTENT_THRESHOLDS[ShoppingIntent.TRANSACTIONAL])
        )
        if account_id is not None and account_id in self._overrides:
            account_overrides = self._overrides[account_id].get(intent, {})
            base.update(account_overrides)
        return base

    def set_override(
        self,
        account_id: int,
        intent: ShoppingIntent,
        key: str,
        value: float,
    ) -> None:
        """Set a single threshold override for an account+intent."""
        if key not in THRESHOLD_KEYS:
            raise ValueError(
                f"Unknown threshold key '{key}'. Valid keys: {list(THRESHOLD_KEYS.keys())}"
            )
        lo, hi = THRESHOLD_KEYS[key]
        if not (lo <= value <= hi):
            raise ValueError(
                f"Value {value} for '{key}' is out of range [{lo}, {hi}]"
            )

        if account_id not in self._overrides:
            self._overrides[account_id] = {}
        if intent not in self._overrides[account_id]:
            self._overrides[account_id][intent] = {}

        self._overrides[account_id][intent][key] = value
        logger.info(
            f"Threshold override set: account={account_id} "
            f"intent={intent.value} {key}={value}"
        )

    def set_overrides_bulk(
        self,
        account_id: int,
        intent: ShoppingIntent,
        overrides: Dict[str, float],
    ) -> None:
        """Set multiple threshold overrides at once."""
        for key, value in overrides.items():
            self.set_override(account_id, intent, key, value)

    def get_account_overrides(
        self,
        account_id: int,
    ) -> Dict[str, Dict[str, float]]:
        """Return all overrides for an account, keyed by intent."""
        if account_id not in self._overrides:
            return {}
        return {
            intent.value: thresholds
            for intent, thresholds in self._overrides[account_id].items()
        }

    def reset_account(self, account_id: int) -> None:
        """Remove all overrides for an account (revert to global defaults)."""
        self._overrides.pop(account_id, None)
        logger.info(f"Threshold overrides reset for account={account_id}")

    def reset_intent(self, account_id: int, intent: ShoppingIntent) -> None:
        """Remove overrides for a specific intent on an account."""
        if account_id in self._overrides:
            self._overrides[account_id].pop(intent, None)

    def get_global_defaults(self) -> Dict[str, Dict[str, float]]:
        """Return the global default thresholds for all intents."""
        return {
            intent.value: deepcopy(thresholds)
            for intent, thresholds in INTENT_THRESHOLDS.items()
        }

    def get_full_profile(
        self,
        account_id: Optional[int] = None,
    ) -> Dict[str, Dict[str, float]]:
        """Return the effective thresholds for all intents (merged with account overrides)."""
        return {
            intent.value: self.get_thresholds(intent, account_id)
            for intent in ShoppingIntent
        }


# Module-level singleton
_manager: Optional[AdaptiveThresholdManager] = None


def get_threshold_manager() -> AdaptiveThresholdManager:
    """Get or create the global AdaptiveThresholdManager singleton."""
    global _manager
    if _manager is None:
        _manager = AdaptiveThresholdManager()
    return _manager
