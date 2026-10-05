# Copyright OpenSearch Contributors
# SPDX-License-Identifier: Apache-2.0
#
# The OpenSearch Contributors require contributions made to
# this file be licensed under the Apache-2.0 license or a
# compatible open source license.

import pytest

from core.index_builder.index_builder_utils import (
    GRAPH_DEGREE_PER_M,
    SMALL_DATASET_MAX_BYTES,
    calculate_effective_m,
    calculate_ivf_pq_n_lists,
)

N_PROBES = 5
SMALL = 1024  # well under SMALL_DATASET_MAX_BYTES


def _effective_m(m, doc_count, vector_data_bytes=SMALL):
    return calculate_effective_m(
        m,
        doc_count,
        calculate_ivf_pq_n_lists(doc_count),
        N_PROBES,
        vector_data_bytes,
    )


@pytest.mark.parametrize(
    "m,doc_count,expected",
    [
        (16, 5, 1),  # smallest build k-NN sends
        (16, 131, 7),  # segment sizes that failed with m=16
        (16, 192, 9),
        (16, 400, 12),
        (16, 1_000, 16),  # can already supply m * 4 neighbors: requested m is kept
        (4, 131, 4),  # small m already fits
        (128, 1_000, 20),  # large m on a small dataset is lowered too
    ],
)
def test_small_dataset_m_is_lowered_to_fit(m, doc_count, expected):
    assert _effective_m(m, doc_count) == expected


@pytest.mark.parametrize(
    "m,doc_count",
    [
        (16, 131),  # would be lowered if it were a small dataset
        (128, 17_000),  # e.g. dim 768 float at the default k-NN threshold
        (64, 3_200),  # e.g. dim 4096 float at the default k-NN threshold
        (16, 1_000_000),
    ],
)
def test_dataset_at_or_above_threshold_keeps_requested_m(m, doc_count):
    assert _effective_m(m, doc_count, SMALL_DATASET_MAX_BYTES) == m
    assert _effective_m(m, doc_count, SMALL_DATASET_MAX_BYTES + 1) == m


def test_dataset_just_below_threshold_is_still_eligible():
    assert _effective_m(128, 1_000, SMALL_DATASET_MAX_BYTES - 1) == 20


@pytest.mark.parametrize("doc_count", [5, 6, 7, 8, 10, 25, 50, 131, 192, 400, 1_000])
def test_effective_graph_degree_fits_the_dataset(doc_count):
    n_lists = calculate_ivf_pq_n_lists(doc_count)
    degree = _effective_m(16, doc_count) * GRAPH_DEGREE_PER_M
    candidates = min(N_PROBES, n_lists) * doc_count // n_lists

    # Never more neighbors than there are other vectors
    assert degree <= doc_count - 1
    # Never more than the IVF-PQ search can see, except at the minimum m of 1
    assert degree <= candidates or degree == GRAPH_DEGREE_PER_M


def test_effective_m_never_exceeds_requested_or_drops_below_one():
    for doc_count in (5, 50, 500, 5_000, 500_000):
        for m in (1, 2, 16, 64):
            assert 1 <= _effective_m(m, doc_count) <= m
