"""Fraud scenario injectors. Importing this package registers all built-in scenarios."""

from fraudlab.synth.scenarios import (  # noqa: F401
    account_takeover,
    bust_out,
    card_testing,
    friendly_fraud,
    merchant_collusion,
    mule_network,
    synthetic_identity,
    velocity_abuse,
)
from fraudlab.synth.scenarios.base import REGISTRY, Scenario, Window, register

__all__ = ["REGISTRY", "Scenario", "Window", "register"]
