import numpy as np

# Normalize features to unit length
def l2_normalize(x, eps=1e-8):
    n = np.linalg.norm(x, axis=1, keepdims=True) + eps
    return x / n

# Cosine top-k nearest neighbor search
def cosine_topk_indices(embeddings, k):
    E = l2_normalize(embeddings.astype(np.float32))
    sim = E @ E.T  # (N, N)

    N = sim.shape[0]
    # Exclude self-similarity
    sim[np.arange(N), np.arange(N)] = -1.0

    # Get top-k indices
    rank = np.argsort(-sim, axis=1)
    return rank[:, :k]

# Evaluate retrieval performance from top-k indices
def evaluate_retrieval_from_topk(topk, items, k):
    N = len(items)
    style = [it["style"] for it in items]

    precisions, recalls, hits = [], [], []
    per_query = []  

    # Build style to index mapping
    style_to_idx = {}
    for i, s in enumerate(style):
        style_to_idx.setdefault(s, []).append(i)

    # Evaluate each query image
    for i in range(N):
        s = style[i]
        gt = [j for j in style_to_idx[s] if j != i]
        if len(gt) == 0:
            continue

        retrieved = topk[i].tolist()
        hit_cnt = len(set(retrieved) & set(gt))

        precisions.append(hit_cnt / k)
        recalls.append(hit_cnt / len(gt))
        hits.append(1.0 if hit_cnt > 0 else 0.0)

        per_query.append({
            "query_idx": i,
            "style": s,
            "gt_size": len(gt),
            "hit_cnt": hit_cnt,
        })

    return {
        f"P@{k}": float(np.mean(precisions)) if precisions else 0.0,
        f"R@{k}": float(np.mean(recalls)) if recalls else 0.0,
        f"Hit@{k}": float(np.mean(hits)) if hits else 0.0,
        "num_queries": len(precisions),
        "per_query": per_query,
    }
