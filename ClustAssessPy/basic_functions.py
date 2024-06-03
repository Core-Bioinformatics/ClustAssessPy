from ClustAssessPy._common_imports import np

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
    # Convert inputs to numpy arrays
    clustering1 = np.asarray(clustering1)
    clustering2 = np.asarray(clustering2)
    
    if clustering1.shape != clustering2.shape:
        raise ValueError("clustering1 and clustering2 must have the same length")
    
    # Function to calculate ECS for a pair.
    def _cluster_overlap_ecs_score(A, B, c_a, c_b):
        A_eq_ca = A == c_a
        B_eq_cb = B == c_b
        A_neq_ca = ~A_eq_ca
        B_neq_cb = ~B_eq_cb

        len_A_eq_ca = np.sum(A_eq_ca)
        len_B_eq_cb = np.sum(B_eq_cb)

        if len_A_eq_ca == 0 or len_B_eq_cb == 0:
            return 0

        score = 1 - 0.5 * (
            np.sum(A_eq_ca & B_eq_cb) * abs((1 / len_A_eq_ca) - (1 / len_B_eq_cb)) +
            np.sum(A_eq_ca & B_neq_cb) / len_A_eq_ca +
            np.sum(B_eq_cb & A_neq_ca) / len_B_eq_cb
        )
        return score
    
    # Identifying all unique pairs
    unique_pairs, indices = np.unique(np.vstack([clustering1, clustering2]).T, axis=0, return_inverse=True)
    
    ecs_element_values = np.zeros(clustering1.size)
    
    # Compute ECS scores for each unique pair and assign
    for i, (c_a, c_b) in enumerate(unique_pairs):
        ecs_val = _cluster_overlap_ecs_score(clustering1, clustering2, c_a, c_b)
        mask = indices == i  # Identify elements belonging to the current pair
        ecs_element_values[mask] = ecs_val
    
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
    return(np.mean(element_sim_elscore(clustering1, clustering2)))

def element_agreement(reference_clustering, clustering_list):
    """
    Inspect how consistently of a set of clusterings agree with a reference clustering by calculating their element-wise average agreement.

    ### Args:
        reference_clustering (array-like): The reference clustering, that each clustering in clustering_list is compared to
        clustering_list (array-like): (list of array-like): A list where each element is a clustering partition represented as an array-like structure (e.g., lists, NumPy arrays).
        
    ### Returns:
        float: A vector containing the element-wise average agreement.

    ### Raises:
        ValueError: If any of the clusterings in the clustering_list are not of the same length as reference_clustering or if they are not appropriate vector types (i.e., NumPy arrays, lists, or tuples).

    ### Example:
    >>> reference_clustering = np.array([1, 1, 2])
    >>> clustering_list = [np.array([1, 1, 2]), np.array([1, 2, 2]), np.array([2, 2, 1])]
    >>> element_agreement(reference_clustering, clustering_list)
    """
    ecs_element_values = np.zeros(len(reference_clustering))

    # Calculate element agreement for each point.
    for clustering in clustering_list:
        ecs_element_values += element_sim_elscore(reference_clustering, clustering)

    return ecs_element_values / len(clustering_list)

def element_consistency(clustering_list):
    """
    Inspect the consistency of a set of clusterings by calculating their element-wise clustering consistency (also known as element-wise frustration).

    ### Args:
        clustering_list (list of array-like): A list where each element is a clustering partition represented as an array-like structure (e.g., lists, NumPy arrays). Each clustering partition should have the same length, with each element representing a cluster assignment.

    ### Returns:
        ndarray: A NumPy array of average consistency scores for each element in the clustering partitions. The length of this array is equal to the number of elements in the individual clustering partitions.

    ### Raises:
        ValueError: If elements of the 'clustering_list' are not of the same length or if they are not appropriate array-like structures.

    ### Example:
    >>> clustering_list = [np.array([1, 1, 2]), np.array([1, 2, 2]), np.array([2, 2, 1])]
    >>> element_consistency(clustering_list)
    """
    ecs_element_values = np.zeros(len(clustering_list[0]))

    for i in range(0, len(clustering_list)):
        for j in range(i + 1, len(clustering_list)):
            ecs_element_values += element_sim_elscore(clustering_list[i], clustering_list[j])

    return ecs_element_values / (len(clustering_list) * (len(clustering_list) - 1) / 2)


def element_sim_matrix(clustering_list):
    """
    Compare a set of clusterings by calculating their pairwise average element-centric clustering similarities.

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
    n = len(clustering_list)
    sim_matrix = np.zeros((n, n))
    
    for i in range(n):
        sim_matrix[i, i] = 1
        for j in range(i + 1, n):
            sim_score = element_sim(clustering_list[i], clustering_list[j])
            sim_matrix[i, j] = sim_score
            sim_matrix[j, i] = sim_score  # Mirror the score to the lower triangle

    return sim_matrix

def weighted_element_consistency(clustering_list, weights=None, calculate_sim_matrix=False):
    """
    Calculate the weighted element-wise consistency of a set of clusterings.

    ### Args:
        clustering_list (list of array-like): A list where each element is a clustering partition.
        weights (list of float, optional): A list of weights corresponding to each clustering in 'clustering_list'.

    ### Returns:
        float: The weighted element-wise consistency score.
        
    ### Example:
    >>> clustering_list = [np.array([1, 1, 2]), np.array([1, 2, 2]), np.array([2, 2, 1])]
    >>> weighted_element_consistency(clustering_list, [1, 0.5, 0.5])
    """

    n_clusterings = len(clustering_list)
    
    if calculate_sim_matrix:
        sim_matrix = np.zeros((n_clusterings, n_clusterings))

    if weights is None:
        weights = np.ones(n_clusterings)

    total_weights = np.sum(weights)
    consistency = 0

    for i in range(n_clusterings - 1):
        # Add self-consistency contribution
        consistency += weights[i] * (weights[i] - 1) / 2
        for j in range(i + 1, n_clusterings):
            current_ecs = element_sim_elscore(clustering_list[i], clustering_list[j])
            
            if calculate_sim_matrix:
                sim_matrix[i, j] = np.mean(current_ecs)
            
            # Weighted contribution to consistency
            consistency += current_ecs * weights[i] * weights[j]

    consistency += weights[-1] * (weights[-1] - 1) / 2
    normalized_consistency = consistency / (total_weights * (total_weights - 1) / 2)
    
    if calculate_sim_matrix:
        return normalized_consistency, sim_matrix
    
    return normalized_consistency
