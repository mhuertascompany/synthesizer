"""Compare mock and observed AstroPT DESI embedding distributions."""

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mocks", type=Path, required=True)
    parser.add_argument("--observed", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-per-domain", type=int, default=5000)
    return parser.parse_args()


def load_embedding_file(path):
    with np.load(path) as data:
        embeddings = np.asarray(data["embeddings"], dtype=np.float64)
        metadata = {
            key: np.asarray(data[key]) for key in ("redshift", "mass") if key in data
        }
    if embeddings.ndim != 2 or not np.all(np.isfinite(embeddings)):
        raise ValueError(f"Invalid embedding matrix in {path}: {embeddings.shape}")
    return embeddings, metadata


def sample_rows(array, metadata, maximum, rng):
    if len(array) <= maximum:
        return array, metadata
    indices = np.sort(rng.choice(len(array), maximum, replace=False))
    return array[indices], {key: value[indices] for key, value in metadata.items()}


def l2_normalize(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Found a zero-norm embedding")
    return x / norms


def nearest_cosine(query, reference, chunk_size=512, exclude_self=False):
    query = l2_normalize(query)
    reference = l2_normalize(reference)
    maxima = []
    for start in range(0, len(query), chunk_size):
        similarity = query[start:start + chunk_size] @ reference.T
        if exclude_self:
            rows = np.arange(len(similarity))
            cols = start + rows
            inside = cols < similarity.shape[1]
            similarity[rows[inside], cols[inside]] = -np.inf
        maxima.append(np.max(similarity, axis=1))
    return np.concatenate(maxima)


def rbf_mmd(x, y, maximum=2000):
    x = x[:maximum]
    y = y[:maximum]
    pooled = np.concatenate((x, y))
    norm2 = np.sum(pooled**2, axis=1)
    squared = norm2[:, None] + norm2[None, :] - 2.0 * (pooled @ pooled.T)
    squared = np.maximum(squared, 0.0)
    positive = squared[squared > 0]
    bandwidth2 = float(np.median(positive)) if positive.size else 1.0
    kernel = np.exp(-squared / (2.0 * bandwidth2))
    nx = len(x)
    kxx = kernel[:nx, :nx]
    kyy = kernel[nx:, nx:]
    kxy = kernel[:nx, nx:]
    # Biased estimate is non-negative and stable for small diagnostic samples.
    return float(kxx.mean() + kyy.mean() - 2.0 * kxy.mean()), bandwidth2


def roc_auc(labels, scores):
    """Return ROC AUC using the rank-sum definition (scores are continuous)."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    positive = labels == 1
    n_positive = np.count_nonzero(positive)
    n_negative = len(labels) - n_positive
    rank_sum = np.sum(ranks[positive])
    return float(
        (rank_sum - n_positive * (n_positive + 1) / 2)
        / (n_positive * n_negative)
    )


def cross_validated_domain_scores(features, labels, folds, rng, alpha=1.0):
    """Fit a leakage-free ridge linear domain classifier in stratified folds."""
    fold_members = [[] for _ in range(folds)]
    for label in (0, 1):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        for fold, subset in enumerate(np.array_split(indices, folds)):
            fold_members[fold].extend(subset.tolist())
    scores = np.empty(len(labels), dtype=np.float64)
    all_indices = np.arange(len(labels))
    for members in fold_members:
        test = np.asarray(members, dtype=int)
        train = np.setdiff1d(all_indices, test, assume_unique=False)
        mean = features[train].mean(axis=0)
        scale = features[train].std(axis=0)
        scale[scale == 0] = 1.0
        x_train = (features[train] - mean) / scale
        x_test = (features[test] - mean) / scale
        target_mean = labels[train].mean()
        target = labels[train] - target_mean
        # Use the smaller primal/dual system for stable high-dimensional fits.
        if x_train.shape[1] <= x_train.shape[0]:
            matrix = x_train.T @ x_train + alpha * np.eye(x_train.shape[1])
            weights = np.linalg.solve(matrix, x_train.T @ target)
        else:
            matrix = x_train @ x_train.T + alpha * np.eye(x_train.shape[0])
            weights = x_train.T @ np.linalg.solve(matrix, target)
        scores[test] = x_test @ weights + target_mean
    return scores


def pca_two_components(features):
    centered = features - features.mean(axis=0)
    _, singular_values, right = np.linalg.svd(centered, full_matrices=False)
    projected = centered @ right[:2].T
    denominator = np.sum(singular_values**2)
    explained = singular_values[:2] ** 2 / denominator
    return projected, explained


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    mocks, mock_meta = load_embedding_file(args.mocks)
    observed, observed_meta = load_embedding_file(args.observed)
    if mocks.shape[1] != observed.shape[1]:
        raise ValueError("Mock and observed embedding dimensions differ")
    mocks, mock_meta = sample_rows(mocks, mock_meta, args.max_per_domain, rng)
    observed, observed_meta = sample_rows(
        observed, observed_meta, args.max_per_domain, rng
    )

    features = np.concatenate((mocks, observed))
    labels = np.concatenate((np.ones(len(mocks)), np.zeros(len(observed))))
    feature_mean = features.mean(axis=0)
    feature_scale = features.std(axis=0)
    feature_scale[feature_scale == 0] = 1.0
    scaled = (features - feature_mean) / feature_scale
    folds = min(5, len(mocks), len(observed))
    if folds < 2:
        raise ValueError("Need at least two mocks and two observed objects")
    domain_scores = cross_validated_domain_scores(features, labels, folds, rng)
    domain_auc = roc_auc(labels, domain_scores)

    normalized_mock = l2_normalize(mocks)
    normalized_observed = l2_normalize(observed)
    mock_to_observed = nearest_cosine(normalized_mock, normalized_observed)
    observed_to_observed = nearest_cosine(
        normalized_observed, normalized_observed, exclude_self=True
    )
    mmd, bandwidth2 = rbf_mmd(normalized_mock, normalized_observed)
    projected, explained_variance = pca_two_components(scaled)

    metrics = {
        "n_mocks": len(mocks),
        "n_observed": len(observed),
        "embedding_dimension": mocks.shape[1],
        "domain_classifier_auc": domain_auc,
        "mock_to_observed_nearest_cosine_median": float(np.median(mock_to_observed)),
        "observed_leave_one_out_nearest_cosine_median": float(
            np.median(observed_to_observed)
        ),
        "rbf_mmd_biased": mmd,
        "rbf_bandwidth_squared": bandwidth2,
        "pca_explained_variance": explained_variance.tolist(),
    }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_prefix.with_suffix(".json")
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    split = len(mocks)
    axes[0].scatter(
        projected[split:, 0], projected[split:, 1], s=8, alpha=0.45,
        label="Observed", color="tab:blue",
    )
    axes[0].scatter(
        projected[:split, 0], projected[:split, 1], s=12, alpha=0.65,
        label="Mocks", color="tab:orange",
    )
    axes[0].set(xlabel="PCA 1", ylabel="PCA 2", title="Joint PCA projection")
    axes[0].legend()
    bins = np.linspace(-1, 1, 60)
    axes[1].hist(
        observed_to_observed, bins=bins, density=True, histtype="step",
        linewidth=2, label="Observed to observed",
    )
    axes[1].hist(
        mock_to_observed, bins=bins, density=True, histtype="step",
        linewidth=2, label="Mock to observed",
    )
    axes[1].set(
        xlabel="Nearest-neighbour cosine similarity", ylabel="Density",
        title=f"Domain AUC = {domain_auc:.3f}",
    )
    axes[1].legend()
    figure_path = args.output_prefix.with_suffix(".png")
    fig.savefig(figure_path, dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, indent=2))
    print(f"Wrote {metrics_path}")
    print(f"Wrote {figure_path}")


if __name__ == "__main__":
    main()
