"""\
This script preps runs Kmedoids according to params

Usage: prep_data.py
"""
from sklearn.metrics import silhouette_score
from sparsemedoid import clustering
from sklearn import metrics
import pandas as pd
import numpy as np

# number of clusters (int), list of distance types ([str]), ...
def run_kmedoids_clustering(clusters, distance_types, normalization_param, X_df):
    total_runs = (len(clusters) * len(distance_types) * len(normalization_param))
    Scores = np.zeros((1, total_runs))
    barcodes = X_df.index.to_list()
    X = X_df.to_numpy()
    P = X.shape[0]
    N = X.shape[1]
    prefix_cols = []
    all_feature_weights = np.zeros((N, total_runs))
    all_cluster_labels = np.zeros((P, total_runs))
    iter1 = 0
    for K in clusters:
        for distance in distance_types:
            for S in normalization_param:
                results_path_prefix = f"K={K}_dist={distance}_S={S}"
                prefix_col = f"N={N}_K={K}_dist={distance}_nparam={S}"
                prefix_cols.append(results_path_prefix)
                (
                    cluster_labels,
                    feature_weights,
                    feature_order,
                    weighted_distances,
                ) = clustering.sparse_kmedoids(
                    X,
                    distance_type=distance,
                    k=K,
                    s=S,
                    max_attempts=6,
                    method="pam",
                    init="build",
                    max_iter=100,
                    random_state=None,
                )
                Scores[0, iter1] += metrics.silhouette_score(
                    weighted_distances, cluster_labels, metric="precomputed"
                )
                all_feature_weights[:, iter1] = feature_weights
                all_cluster_labels[:, iter1] = cluster_labels
                iter1 += 1
    feature_weights_df = pd.DataFrame(all_feature_weights)
    cluster_labels_df = pd.DataFrame(all_cluster_labels)
    cluster_labels_df.index = barcodes
    cluster_labels_df.columns = prefix_cols
    feature_weights_df.index = X_df.columns.to_list()
    feature_weights_df.columns = prefix_cols
    scores_df = pd.DataFrame(Scores)
    scores_df.columns = prefix_cols
    return scores_df, cluster_labels_df, feature_weights_df
