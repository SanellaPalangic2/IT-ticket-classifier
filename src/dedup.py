"""Find near-duplicate tickets and group them into clusters.

Used by profile_data.py (to count them) and clean_data.py (to give each
ticket a group_id so the split can keep a whole cluster on one side).
"""
import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.feature_extraction.text import TfidfVectorizer

NEAR_DUP_THRESHOLD = 0.9  # cosine similarity at or above this counts as "near duplicate"


def find_near_duplicates(texts, threshold=NEAR_DUP_THRESHOLD, chunk_size=1000):
    """Return pairs (i, j, similarity) with i < j and TF-IDF cosine >= threshold.

    TF-IDF rows are L2-normalised, so a dot product of two rows is their
    cosine similarity. Comparing all pairs at once would need too much
    memory, so we compare one chunk of rows at a time against all rows.
    """
    X = TfidfVectorizer().fit_transform(texts)
    pairs = []
    for start in range(0, X.shape[0], chunk_size):
        sims = (X[start:start + chunk_size] @ X.T).tocoo()
        rows = sims.row + start
        # Filter with numpy first: almost every pair shares some common word,
        # so looping over all nonzero similarities in Python is far too slow.
        keep = (sims.data >= threshold) & (rows < sims.col)
        pairs += zip(rows[keep].tolist(), sims.col[keep].tolist(), sims.data[keep].tolist())
    return pairs


def assign_groups(texts, threshold=NEAR_DUP_THRESHOLD):
    """Give every text a cluster id. Near-duplicates share an id.

    Pairs are chained: if A~B and B~C, then A, B and C share one id, even if
    A and C are not directly similar. A text with no near-duplicate gets an
    id of its own.
    """
    return groups_from_pairs(find_near_duplicates(texts, threshold), len(texts))


def groups_from_pairs(pairs, n):
    """Turn similar pairs into cluster ids (connected components of a graph
    whose nodes are texts and whose edges are the similar pairs)."""
    i_idx = [p[0] for p in pairs]
    j_idx = [p[1] for p in pairs]
    graph = coo_matrix((np.ones(len(pairs)), (i_idx, j_idx)), shape=(n, n))
    _, group_ids = connected_components(graph, directed=False)
    return group_ids
