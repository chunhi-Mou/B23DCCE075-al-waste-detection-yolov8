"""PPAL CCMS diversity (Yang et al., CVPR 2024, arXiv:2211.11612).
Pure numpy."""
import numpy as np


def _l2(x: np.ndarray) -> np.ndarray:
    """L2-normalize along last axis."""
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.where(n > 0, n, 1.0)


def image_distance_matrix(feats: list, labels: list, scores: list,
                          score_thr: float) -> np.ndarray:
    # category-conditioned image distance (PPAL Eq.6/7):
    # nearest same-class box cosine dist, score-weighted, symmetrized
    N = len(feats)
    nf = [_l2(np.asarray(f, float)) if len(f) else np.zeros((0, 1))
          for f in feats]
    lab = [np.asarray(l) for l in labels]
    sc = [np.asarray(s, float) for s in scores]
    D = np.zeros((N, N), float)
    for i in range(N):
        fi, li, si = nf[i], lab[i], sc[i]
        for j in range(N):
            if i == j or len(fi) == 0 or len(nf[j]) == 0:
                continue
            fj, lj, sj = nf[j], lab[j], sc[j]
            d = 1.0 - fi @ fj.T                     # (ni, nj) cosine dist
            d = np.where(li[:, None] == lj[None, :], d, 2.0)  # diff class -> 2
            d = np.where(sj[None, :] > score_thr, d, np.inf)  # below thr -> inf
            md = d.min(axis=1)                      # nearest same-class match
            md = np.where(md > 2.0, 0.0, md)        # no valid box in j -> 0
            norm = (si.sum() if (sj > score_thr).any() else 0.0) + 1e-5
            D[i, j] = float((md * si).sum() / norm)
    return 0.5 * (D + D.T)


def kcenter_greedy_matrix(D: np.ndarray, K: int, rng) -> list:
    # random first center, then farthest-point greedy on D
    N = D.shape[0]
    centroids = [rng.randrange(N)]
    while len(centroids) < K:
        md = D[:, centroids].min(axis=1)
        md[centroids] = -1.0
        centroids.append(int(np.argmax(md)))
    return centroids


def kmedoids(D: np.ndarray, K: int, n_iter: int, rng) -> list:
    # k-Center greedy init, then n_iter Lloyd-style medoid updates
    N = D.shape[0]
    centroids = np.array(kcenter_greedy_matrix(D, K, rng))
    idx = np.arange(N)
    for _ in range(n_iter):
        assign = np.argmin(D[:, centroids], axis=1)
        new = []
        for k in range(K):
            members = idx[assign == k]
            if len(members) == 0:
                new.append(centroids[k])
                continue
            sub = D[np.ix_(members, members)]
            new.append(int(members[np.argmin(sub.sum(axis=1))]))
        centroids = np.array(new)
    return centroids.tolist()
