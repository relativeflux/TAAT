import numpy as np
from sklearn.cluster import AgglomerativeClustering


# Create linkage matrix
def get_linkage_matrix(model, **kwargs):
    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1 # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count
    # Build the matrix
    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)
    return linkage_matrix

def get_clusters(X):
    # distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(distance_threshold=0, n_clusters=None)
    
    labels = model.fit_predict(X, y=None)
    linkage_matrix = get_linkage_matrix(model)

    return labels.tolist(), linkage_matrix.tolist()