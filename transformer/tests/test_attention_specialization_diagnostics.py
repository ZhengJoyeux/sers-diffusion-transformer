import numpy as np

from Inference import _attention_specialization_tables


def test_uniform_attention_has_max_entropy_and_high_query_similarity():
    attention = np.full((6, 3, 10), 0.1, dtype=np.float64)
    y_true = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 1, 0],
            [1, 0, 1],
            [0, 1, 1],
        ],
        dtype=np.int64,
    )

    summary, pairwise, details = _attention_specialization_tables(
        attention, y_true, top_k=2
    )

    all_rows = summary[summary["scope"] == "all"]
    assert np.allclose(all_rows["normalized_entropy_mean"], 1.0)
    assert np.allclose(all_rows["top_k_mass_mean"], 0.2)
    assert np.allclose(pairwise["cosine_similarity_mean"], 1.0)
    assert np.allclose(pairwise["jensen_shannon_divergence_mean"], 0.0)
    assert len(details) == 6 * 3


def test_distinct_concentrated_queries_reduce_entropy_and_similarity():
    attention = np.full((4, 3, 10), 1e-6, dtype=np.float64)
    attention[:, 0, 1] = 1.0
    attention[:, 1, 4] = 1.0
    attention[:, 2, 7] = 1.0
    attention /= attention.sum(axis=-1, keepdims=True)
    y_true = np.ones((4, 3), dtype=np.int64)

    summary, pairwise, _ = _attention_specialization_tables(
        attention, y_true, top_k=2
    )
    all_rows = summary[summary["scope"] == "all"]

    assert (all_rows["normalized_entropy_mean"] < 0.1).all()
    assert (all_rows["top_k_mass_mean"] > 0.99).all()
    assert (pairwise["cosine_similarity_mean"] < 0.01).all()
    assert (pairwise["jensen_shannon_divergence_mean"] > 0.5).all()
