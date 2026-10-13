"""Static vocabularies: merchant categories, channels, countries.

Values are stylised rather than calibrated to any real issuer. The relative
ordering (groceries cheap and frequent, travel and electronics expensive and
rare) follows the qualitative descriptions in the card-fraud literature
(Correa Bahnsen et al. 2016; Le Borgne et al. 2022).
"""

from __future__ import annotations

import numpy as np

# name, popularity weight, log-amount offset, share of transactions online
MCC_TABLE: tuple[tuple[str, float, float, float], ...] = (
    ("grocery", 0.22, -0.2, 0.10),
    ("restaurant", 0.16, -0.3, 0.05),
    ("fuel", 0.09, 0.0, 0.00),
    ("retail", 0.13, 0.3, 0.30),
    ("pharmacy", 0.05, -0.4, 0.10),
    ("utilities", 0.04, 0.6, 0.90),
    ("entertainment", 0.06, 0.0, 0.60),
    ("online_marketplace", 0.10, 0.4, 1.00),
    ("digital_goods", 0.06, -0.8, 1.00),
    ("electronics", 0.04, 1.4, 0.50),
    ("travel", 0.03, 1.8, 0.80),
    ("gift_cards", 0.02, 0.8, 0.70),
)
MCC_NAMES: tuple[str, ...] = (*(m[0] for m in MCC_TABLE), "p2p_transfer")
MCC_INDEX: dict[str, int] = {name: i for i, name in enumerate(MCC_NAMES)}
MCC_POPULARITY = np.array([m[1] for m in MCC_TABLE]) / sum(m[1] for m in MCC_TABLE)
MCC_LOG_OFFSET = np.array([m[2] for m in MCC_TABLE] + [0.8])
MCC_ONLINE_SHARE = np.array([m[3] for m in MCC_TABLE] + [1.0])
P2P_MCC = MCC_INDEX["p2p_transfer"]

CHANNELS: tuple[str, ...] = ("pos", "online", "transfer")
CH_POS, CH_ONLINE, CH_TRANSFER = 0, 1, 2

N_COUNTRIES = 12
COUNTRIES: tuple[str, ...] = tuple(f"C{i:02d}" for i in range(N_COUNTRIES))
#: Home-country distribution of accounts (one dominant domestic market).
HOME_COUNTRY_P = np.array([0.82] + [0.18 / (N_COUNTRIES - 1)] * (N_COUNTRIES - 1))

#: Categories favoured by fraudsters cashing out stolen credentials.
HIGH_RISK_MCC: tuple[int, ...] = (
    MCC_INDEX["electronics"],
    MCC_INDEX["gift_cards"],
    MCC_INDEX["online_marketplace"],
    MCC_INDEX["digital_goods"],
)

SEGMENTS: tuple[str, ...] = ("standard", "premium", "student")
SEGMENT_P = np.array([0.75, 0.10, 0.15])
SEGMENT_LOG_AMOUNT = np.array([3.4, 4.2, 2.9])  # median ~$30 / $67 / $18

ACCOUNT_ROLES: tuple[str, ...] = ("customer", "mule", "collector", "synthetic")
