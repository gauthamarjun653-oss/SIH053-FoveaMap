"""Class taxonomy and the namespace trap (spec section 5).

Pretrained SalsaNext outputs LEARNING-SPACE indices 0-19, not raw SemanticKITTI
IDs. Both are small integers, so applying the raw-ID table to argmax output
would produce silently wrong labels that look plausible and never raise. The
chain is:

    logits[20] -> argmax -> learning idx (0-19) -> learning_map_inv -> raw ID -> 3-class

collapsed here into one precomputed lookup, derived from the YAML and asserted
against a literal so either source being wrong aborts loudly (T-04).
"""

from __future__ import annotations

import numpy as np
import yaml

IGNORE, ROAD, STATIC, DYNAMIC = 0, 1, 2, 3
CLASS_NAMES = ("IGNORE", "ROAD", "STATIC", "DYNAMIC")

ROAD_RAW = {40, 44, 48, 49, 60, 72}
STATIC_RAW = {50, 51, 52, 70, 71, 80, 81, 99}
DYNAMIC_RAW = {10, 11, 13, 15, 16, 18, 20, 30, 31, 32} | set(range(252, 260))

# The table the spec was written against. Notes recorded so nobody debugs them
# later (spec section 5.4): raw 252-259 are unreachable after learning_map has
# merged moving classes (harmless); raw 52 and 99 map to learning idx 0 and so
# land in IGNORE, not STATIC — an accepted trade.
EXPECTED_TABLE = np.array([0, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 2, 2, 2, 2, 1, 2, 2],
                          dtype=np.uint8)


def _raw_to_class3(raw: int) -> int:
    if raw in ROAD_RAW:
        return ROAD
    if raw in STATIC_RAW:
        return STATIC
    if raw in DYNAMIC_RAW:
        return DYNAMIC
    return IGNORE


def build_learning_to_class3(yaml_path: str) -> np.ndarray:
    """Derive the 20-entry learning-idx -> 3-class table from semantic-kitti.yaml
    and assert it against the literal. Either failure aborts loudly."""
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    inv = cfg["learning_map_inv"]  # learning idx -> raw id

    table = np.zeros(20, dtype=np.uint8)
    for i in range(20):
        table[i] = _raw_to_class3(int(inv[i]))

    assert np.array_equal(table, EXPECTED_TABLE), (
        f"learning_map_inv produced {table.tolist()}, expected {EXPECTED_TABLE.tolist()}. "
        "Do not proceed — inspect semantic-kitti.yaml before continuing."
    )
    return table


def build_raw_to_class3() -> np.ndarray:
    """Raw SemanticKITTI ID -> 3-class table (IDs 0..259), for GROUND-TRUTH
    labels only. Ground truth is in raw-ID space (including moving classes
    252-259); predictions are in learning space and must go through
    build_learning_to_class3 instead."""
    table = np.zeros(260, dtype=np.uint8)
    for raw in range(260):
        table[raw] = _raw_to_class3(raw)
    return table
