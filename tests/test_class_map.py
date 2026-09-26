"""T-04: build_learning_to_class3 derived from YAML and asserted vs literal."""

import numpy as np
import pytest
import yaml

from src.segmentation.class_map import (EXPECTED_TABLE, build_learning_to_class3,
                                        build_raw_to_class3,
                                        IGNORE, ROAD, STATIC, DYNAMIC)


def test_t04_table_matches_literal():
    table = build_learning_to_class3("configs/semantic-kitti.yaml")
    assert np.array_equal(table, EXPECTED_TABLE)


def test_t04_corrupted_yaml_aborts(tmp_path):
    inv = {i: v for i, v in enumerate([0, 10, 11, 15, 18, 20, 30, 31, 32, 40,
                                       44, 48, 49, 50, 51, 70, 71, 72, 80, 81])}
    inv[9], inv[13] = inv[13], inv[9]  # swap road and building
    p = tmp_path / "bad.yaml"
    p.write_text(yaml.safe_dump({"learning_map_inv": inv}), encoding="utf-8")
    with pytest.raises(AssertionError):
        build_learning_to_class3(str(p))


def test_raw_table_for_ground_truth():
    t = build_raw_to_class3()
    assert t[40] == ROAD and t[48] == ROAD and t[72] == ROAD
    assert t[50] == STATIC and t[80] == STATIC
    assert t[10] == DYNAMIC and t[30] == DYNAMIC
    assert t[252] == DYNAMIC and t[259] == DYNAMIC  # moving classes, GT space
    assert t[0] == IGNORE and t[1] == IGNORE
