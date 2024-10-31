from ClustAssessPy._common_imports import np, coo_matrix

def element_sim_elscore(clustering1, clustering2):
    """
    Calculates the element-wise element-centric similarity between two clustering results.
    
    ### Args:
        clustering1 (array-like): The first clustering partition, where each element represents a cluster assignment.
        clustering2 (array-like): The second clustering partition to compare against 'clustering1'.
        
    ### Returns:
        ndarray: A NumPy array of ECS values for each element in the clustering partitions. The length of this array is equal to the number of elements in the clustering partitions.

    ### Raises:
        ValueError: If 'clustering1' and 'clustering2' are not of the same length.

    ### Example:
    >>> clustering1 = np.array([1, 1, 2, 2, 3])
    >>> clustering2 = np.array([1, 2, 2, 3, 3])
    >>> element_sim_elscore(clustering1, clustering2)
    """

    clustering1 = np.asarray(clustering1)
    clustering2 = np.asarray(clustering2)

    if clustering1.shape != clustering2.shape:
        raise ValueError("clustering1 and clustering2 must have the same length")

    # Map cluster labels to indices starting from 0
    labels_a, inverse_a = np.unique(clustering1, return_inverse=True)
    labels_b, inverse_b = np.unique(clustering2, return_inverse=True)
    n_labels_a = labels_a.size
    n_labels_b = labels_b.size

    # Compute the contingency table C
    data = np.ones(clustering1.size)
    C = coo_matrix((data, (inverse_a, inverse_b)), shape=(n_labels_a, n_labels_b)).toarray()

    len_A_eq_ca = C.sum(axis=1)  # Number of elements in each cluster in clustering1
    len_B_eq_cb = C.sum(axis=0)  # Number of elements in each cluster in clustering2

    # Compute the inverse cluster sizes, handling zero divisions
    len_A_eq_ca_inv = np.divide(1.0, len_A_eq_ca, where=len_A_eq_ca != 0)
    len_B_eq_cb_inv = np.divide(1.0, len_B_eq_cb, where=len_B_eq_cb != 0)

    # Compute the absolute difference of inverse cluster sizes
    delta_inv = np.abs(len_A_eq_ca_inv[:, None] - len_B_eq_cb_inv[None, :])

    # Compute the terms used in the ECS calculation
    term1 = C * delta_inv
    term2 = (len_A_eq_ca[:, None] - C) * len_A_eq_ca_inv[:, None]
    term3 = (len_B_eq_cb[None, :] - C) * len_B_eq_cb_inv[None, :]

    # Compute the ECS score matrix
    score_matrix = 1 - 0.5 * (term1 + term2 + term3)
    score_matrix = np.clip(score_matrix, 0, 1)

    # Assign ECS values to each element based on their cluster assignments
    ecs_element_values = score_matrix[inverse_a, inverse_b]

    return ecs_element_values

def element_sim(clustering1, clustering2):
    """
    Calculate the average element-centric similarity between two clustering results

    ### Args:
        clustering1 (array-like): The first clustering partition, where each element represents a cluster assignment.
        clustering2 (array-like): The second clustering partition to compare against 'clustering1'.
        
    ### Returns:
        float: The average ECS value for all elements across the two clustering partitions.

    ### Raises:
        ValueError: If 'clustering1' and 'clustering2' are not of the same length or if they are not appropriate vector types (i.e., NumPy arrays, lists, or tuples).

    ### Example:
    >>> clustering1 = np.array([1, 1, 2, 2, 3])
    >>> clustering2 = np.array([1, 2, 2, 3, 3])
    >>> element_sim(clustering1, clustering2)
    """
    return np.mean(element_sim_elscore(clustering1, clustering2))

def element_agreement(reference_clustering, clustering_list):
    """
    Inspect how consistently a set of clusterings agree with a reference clustering by calculating their element-wise average agreement.

    Optimized using precomputed data.

    ### Args:
        reference_clustering (array-like): The reference clustering, that each clustering in clustering_list is compared to
        clustering_list (list of array-like): A list where each element is a clustering partition represented as an array-like structure (e.g., lists, NumPy arrays).
        
    ### Returns:
        ndarray: A vector containing the element-wise average agreement.

    ### Raises:
        ValueError: If any of the clusterings in the clustering_list are not of the same length as reference_clustering or if they are not appropriate array-like structures.

    ### Example:
    >>> reference_clustering = np.array([1, 1, 2])
    >>> clustering_list = [np.array([1, 1, 2]), np.array([1, 2, 2]), np.array([2, 2, 1])]
    >>> element_agreement(reference_clustering, clustering_list)
    """
    reference_clustering = np.asarray(reference_clustering)
    num_elements = len(reference_clustering)
    num_clusterings = len(clustering_list)

    # Precompute data for the reference clustering
    labels_ref, inverse_ref = np.unique(reference_clustering, return_inverse=True)
    data_ref = {
        'clustering': reference_clustering,
        'labels': labels_ref,
        'inverse': inverse_ref
    }

    # Initialize ECS accumulator
    ecs_element_values = np.zeros(num_elements)

    # Compute ECS values for each clustering
    for clustering in clustering_list:
        clustering = np.asarray(clustering)
        labels, inverse = np.unique(clustering, return_inverse=True)
        data = {
            'clustering': clustering,
            'labels': labels,
            'inverse': inverse
        }
        ecs_values = _element_sim_elscore_with_precomputed(data_ref, data)
        ecs_element_values += ecs_values

    # Compute average ECS values
    ecs_element_values /= num_clusterings

    return ecs_element_values

def element_consistency(clustering_list):
    """
    Inspects the consistency of a set of clusterings by calculating their element-wise clustering consistency.

    Optimized version using precomputed data and efficient looping.

    ### Args:
        clustering_list (list of array-like): A list where each element is a clustering partition represented as an array-like structure (e.g., lists, NumPy arrays). Each clustering partition should have the same length, with each element representing a cluster assignment.

    ### Returns:
        ndarray: A NumPy array of average consistency scores for each element in the clustering partitions.
        The length of this array is equal to the number of elements in the individual clustering partitions.

    ### Raises:
        ValueError: If elements of the 'clustering_list' are not of the same length or if they are not appropriate array-like structures.

    ### Example:
        >>> import numpy as np
        >>> clustering_list = [np.array([1, 1, 2]), np.array([1, 2, 2]), np.array([2, 2, 1])]
        >>> ecs_values = element_consistency(clustering_list)
        >>> print(ecs_values)
    """

    # Precompute data for each clustering
    clustering_data = _precompute_clustering_data(clustering_list)
    num_clusterings = len(clustering_data)
    num_elements = len(clustering_data[0]['clustering'])

    # Check that all clusterings have the same length
    if not all(len(data['clustering']) == num_elements for data in clustering_data):
        raise ValueError("All clusterings must have the same number of elements")

    ecs_element_values = np.zeros(num_elements)
    num_pairs = num_clusterings * (num_clusterings - 1) // 2

    # Generate all unique pairs of indices
    indices_i, indices_j = np.triu_indices(num_clusterings, k=1)

    # Loop over pairs using precomputed data
    for idx in range(len(indices_i)):
        i = indices_i[idx]
        j = indices_j[idx]
        ecs_element_values += _element_sim_elscore_with_precomputed(
            clustering_data[i], clustering_data[j]
        )

    # Compute the average ECS values
    ecs_element_values /= num_pairs

    return ecs_element_values

def element_sim_matrix(clustering_list):
    """
    Compare a set of clusterings by calculating their pairwise average element-centric clustering similarities.

    Optimized to avoid redundant computations and utilize vectorized operations.

    ### Args:
        clustering_list (list of array-like): A list where each element is a clustering partition represented as an array-like structure (e.g., lists, NumPy arrays). Each clustering partition should have the same length, with each element representing a cluster assignment.

    ### Returns:
        ndarray: A NumPy matrix containing the pairwise ECS values.

    ### Raises:
        ValueError: If elements of the 'clustering_list' are not of the same length or if they are not appropriate array-like structures.

    ### Example:
    >>> clustering_list = [np.array([1, 1, 2]), np.array([1, 2, 2]), np.array([2, 2, 1])]
    >>> element_sim_matrix(clustering_list)
    """

    # Precompute data for all clusterings
    clustering_data = _precompute_clustering_data(clustering_list)
    num_clusterings = len(clustering_data)
    sim_matrix = np.ones((num_clusterings, num_clusterings))  # Diagonal elements are 1

    # Generate upper triangle indices (excluding diagonal)
    indices_i, indices_j = np.triu_indices(num_clusterings, k=1)

    # Compute pairwise similarities
    for idx in range(len(indices_i)):
        i = indices_i[idx]
        j = indices_j[idx]
        sim_score = _element_sim_with_precomputed(clustering_data[i], clustering_data[j])
        sim_matrix[i, j] = sim_score
        sim_matrix[j, i] = sim_score  # Symmetric matrix

    return sim_matrix

def weighted_element_consistency(clustering_list, weights=None, calculate_sim_matrix=False):
    """
    Calculate the weighted element-wise consistency of a set of clusterings.

    This function computes per-element weighted consistency scores, aligning with the old implementation.

    ### Args:
        clustering_list (list of array-like): A list where each element is a clustering partition.
        weights (list of float, optional): A list of weights corresponding to each clustering in 'clustering_list'.
        calculate_sim_matrix (bool, optional): Whether to return the similarity matrix.

    ### Returns:
        ndarray: An array of per-element weighted consistency scores.
        (optional) ndarray: Similarity matrix if calculate_sim_matrix is True.

    ### Example:
        >>> clustering_list = [np.array([1, 1, 2]), np.array([1, 2, 2]), np.array([2, 2, 1])]
        >>> weighted_ecc = weighted_element_consistency(clustering_list, weights=[1, 0.5, 0.5])
        >>> print(weighted_ecc)
    """
    n_clusterings = len(clustering_list)
    num_elements = len(clustering_list[0])

    # Precompute data for all clusterings
    clustering_data = _precompute_clustering_data(clustering_list)

    if weights is None:
        weights = np.ones(n_clusterings)
    else:
        weights = np.asarray(weights)
        if weights.shape[0] != n_clusterings:
            raise ValueError("Length of 'weights' must match the number of clusterings.")

    total_weights = np.sum(weights)
    num_pairs = total_weights * (total_weights - 1) / 2

    # Initialize consistency accumulator as an array
    consistency = np.zeros(num_elements)

    if calculate_sim_matrix:
        sim_matrix = np.zeros((n_clusterings, n_clusterings))  # Initialize with zeros

    # Generate all unique pairs of indices
    indices_i, indices_j = np.triu_indices(n_clusterings, k=1)

    # Compute pairwise ECS values and accumulate weighted consistency
    for idx in range(len(indices_i)):
        i = indices_i[idx]
        j = indices_j[idx]
        weight_pair = weights[i] * weights[j]

        ecs_values = _element_sim_elscore_with_precomputed(clustering_data[i], clustering_data[j])

        consistency += ecs_values * weight_pair

        if calculate_sim_matrix:
            mean_ecs = np.mean(ecs_values)
            sim_matrix[i, j] = mean_ecs  # Only assign upper triangle

    # Add self-consistency contributions (element-wise)
    for i in range(n_clusterings):
        weight = weights[i]
        # Since ECS of a clustering with itself is 1 for all elements,
        # and weights[i] * (weights[i] - 1) / 2 is the weight of self-consistency,
        # we multiply it by 1 (the ECS value) for each element.
        consistency += (weight * (weight - 1) / 2) * np.ones(num_elements)

    # Normalize consistency
    normalized_consistency = consistency / num_pairs

    if calculate_sim_matrix:
        return normalized_consistency, sim_matrix  # sim_matrix remains upper triangular with zeros elsewhere

    return normalized_consistency

# Private helper functions
def _element_sim_with_precomputed(data_a, data_b):
    """
    Computes the average ECS between two clusterings using precomputed data.
    """
    ecs_values = _element_sim_elscore_with_precomputed(data_a, data_b)
    return np.mean(ecs_values)

def _precompute_clustering_data(clustering_list):
    """
    Precomputes necessary data for each clustering to optimize computations.

    This function processes each clustering to extract:
    - 'labels': The unique cluster labels.
    - 'inverse': An array mapping each element to its cluster index.

    By precomputing this data, we avoid redundant computations during pairwise comparisons.

    ### Args:
        clustering_list (list of array-like): List of clustering partitions.

    ### Returns:
        list of dict: List containing precomputed data for each clustering.
    """
    clustering_data = []
    for clustering in clustering_list:
        clustering = np.asarray(clustering)
        labels, inverse = np.unique(clustering, return_inverse=True)
        clustering_data.append({
            'clustering': clustering,
            'labels': labels,
            'inverse': inverse
        })
    return clustering_data

def _element_sim_elscore_with_precomputed(data_a, data_b):
    """
    Calculates ECS values using precomputed clustering data.

    ### Args:
        data_a (dict): Precomputed data for clustering A.
        data_b (dict): Precomputed data for clustering B.

    ### Returns:
        ndarray: A NumPy array of ECS values for each element.
    """
    inverse_a = data_a['inverse']
    inverse_b = data_b['inverse']
    labels_a = data_a['labels']
    labels_b = data_b['labels']
    n_labels_a = labels_a.size
    n_labels_b = labels_b.size

    # Efficient contingency table computation
    indices = inverse_a * n_labels_b + inverse_b
    C = np.bincount(indices, minlength=n_labels_a * n_labels_b).reshape(n_labels_a, n_labels_b)

    len_A_eq_ca = np.bincount(inverse_a)
    len_B_eq_cb = np.bincount(inverse_b)
    len_A_eq_ca_inv = np.divide(1.0, len_A_eq_ca, where=len_A_eq_ca != 0)
    len_B_eq_cb_inv = np.divide(1.0, len_B_eq_cb, where=len_B_eq_cb != 0)

    delta_inv = np.abs(len_A_eq_ca_inv[:, None] - len_B_eq_cb_inv[None, :])

    term1 = C * delta_inv
    term2 = (len_A_eq_ca[:, None] - C) * len_A_eq_ca_inv[:, None]
    term3 = (len_B_eq_cb[None, :] - C) * len_B_eq_cb_inv[None, :]

    score_matrix = 1 - 0.5 * (term1 + term2 + term3)
    score_matrix = np.clip(score_matrix, 0, 1)

    ecs_element_values = score_matrix[inverse_a, inverse_b]

    return ecs_element_values