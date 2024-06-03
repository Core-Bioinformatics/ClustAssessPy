from ClustAssessPy._common_imports import np, warnings, multiprocessing, pd, UMAP
from ClustAssessPy.basic_functions import weighted_element_consistency
from ClustAssessPy._private_helper_functions import nn2_indices, compute_partitions, merge_partitions, get_graph_types_list, process_neigh_matrix, get_highest_prune_param, get_partitionings
from snn_functions import getNNmatrix
import concurrent.futures

def _validate_parameters_stability(n_neigh_sequence, n_repetitions, ecs_thresh, graph_type, embedding, algorithm, graph_reduction_type, flavor, ncores):
    if not isinstance(n_neigh_sequence, (np.ndarray, list, tuple)):
        raise ValueError("n_neigh_sequence parameter should be numeric")

    if not isinstance(n_repetitions, int):
        raise ValueError("n_repetitions parameter should be numeric")

    if not isinstance(ecs_thresh, (int, float)):
        raise ValueError("ecs_thresh parameter should be numeric")

    if not isinstance(graph_type, int) or not (graph_type in [0, 1, 2]):
        raise ValueError("graph_type parameter should be a either 0 (nn) or 1 (snn) or 2 (both)")

    if not isinstance(embedding, (np.ndarray, pd.DataFrame)):
        raise ValueError("The embedding parameter should be a matrix")

    if algorithm not in ['louvain', 'leiden']:
        raise ValueError("algorithm should be either 'louvain' or 'leiden'")

    if not isinstance(graph_reduction_type, str) or not (graph_reduction_type in ["PCA", "UMAP"]):
        raise ValueError("graph_reduction_type parameter should take one of these values: 'PCA' or 'UMAP'")
    
    if flavor != "default" and flavor != "igraph":
        raise ValueError("flavor parameter should be 'default' or 'igraph'")
    
    cores_available = multiprocessing.cpu_count()
    if ncores > cores_available:
        raise ValueError(f"The number of cores should be less than or equal to {cores_available}.")
    

def _compute_neigh_matrices(n_neigh_sequence, nn2_res, graph_type, ncores):
    with concurrent.futures.ProcessPoolExecutor(max_workers = ncores) as executor:
        future_to_n_neigh = {executor.submit(process_neigh_matrix, n_neigh, nn2_res, graph_type): n_neigh for n_neigh in n_neigh_sequence}

        results = {}
        for future in concurrent.futures.as_completed(future_to_n_neigh.keys()):
            n_neigh = future_to_n_neigh[future]
            results[n_neigh] = future.result()

    return results

# create an object showing the number of clusters obtained for each number of neighbours.
def _get_number_of_clusters_obtained_for_each_neighbor(partitions_list):
    nn_object_n_clusters= {}
    
    for config_name, config_data in partitions_list.items():
        
        nn_object_n_clusters[config_name] = {}
        for n_neigh, neigh_data in config_data.items():
            # Pre-calculate set lengths and frequencies for vectorization
            set_lengths = np.array([len(set(x['mb'])) for x in neigh_data['partitions']])
            freqs = np.array([x['freq'] for x in neigh_data['partitions']])
            
            nn_object_n_clusters[config_name][n_neigh] = np.repeat(set_lengths, freqs).tolist()
            
    return nn_object_n_clusters

def _get_number_different_partitions(partitions_list):
    n_different_partitions = {}
    for config_name, config in partitions_list.items():
        if config_name not in n_different_partitions:
            n_different_partitions[config_name] = {}
            
        for n_neigh_key, n_neigh_value in config.items():
            n_different_partitions[config_name][n_neigh_key] = len(n_neigh_value['partitions'])

    return n_different_partitions

def _compute_ecc_for_n_neigh(config_name, n_neigh_key, n_neigh_value):
    mb_values = [x['mb'] for x in n_neigh_value['partitions']]
    freq_values = [x['freq'] for x in n_neigh_value['partitions']]
    return config_name, n_neigh_key, weighted_element_consistency(mb_values, freq_values)

def _ecc_of_partitions_for_each_neighbor(partitions_list, ncores):
    nn_ecs_object = {config_name: {} for config_name in partitions_list}
    tasks = []
    
    for config_name, config in partitions_list.items():
        for n_neigh_key, n_neigh_value in config.items():
            tasks.append((config_name, n_neigh_key, n_neigh_value))

    with concurrent.futures.ProcessPoolExecutor(max_workers=ncores) as executor:
        futures = [executor.submit(_compute_ecc_for_n_neigh, *task) for task in tasks]

        for future in concurrent.futures.as_completed(futures):
            config_name, n_neigh_key, ecc = future.result()
            nn_ecs_object[config_name][n_neigh_key] = ecc
    
    return nn_ecs_object

def _compute_single_seed_result(embedding, seed, n_neigh_sequence, graph_types_list, umap_arguments, algorithm, flavor, show_warnings):
    np.random.seed(seed)
    
    umap_model = UMAP(**umap_arguments)
    reduced_embedding = umap_model.fit_transform(embedding)

    nn2_res = nn2_indices(reduced_embedding, max(n_neigh_sequence))
    
    seed_result = {gt: {} for gt in graph_types_list}
    for n_neigh in n_neigh_sequence:
        neigh_matrix = getNNmatrix(nn2_res, n_neigh, 0, -1)

        for graph_type in graph_types_list:
            if graph_type == "snn":
                neigh_matrix[graph_type] = get_highest_prune_param(neigh_matrix['nn'], n_neigh)['adj_matrix']

            seed_result[graph_type][str(n_neigh)] = get_partitionings(neigh_matrix, algorithm, seed, graph_type, flavor, show_warnings)

    return seed_result

def _compute_umap_seed_results(embedding, seed_sequence, n_neigh_sequence, graph_types_list, umap_arguments, algorithm, flavor, ncores, show_warnings):
    # Compute results for each seed in parallel.
    with concurrent.futures.ProcessPoolExecutor(max_workers=ncores) as executor:
        futures = [executor.submit(_compute_single_seed_result, embedding, seed, n_neigh_sequence, graph_types_list, umap_arguments, algorithm, flavor, show_warnings) for seed in seed_sequence]
        seed_list = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    return seed_list

def _assess_nn_stability_pca(embedding, n_neigh_sequence, seed_sequence, ecs_thresh, graph_type, algorithm, flavor, ncores, show_warnings):
    
    nn2_res = nn2_indices(embedding, k = max(n_neigh_sequence))
    
    # Parallel execution
    neigh_matrices = _compute_neigh_matrices(n_neigh_sequence, nn2_res, graph_type, ncores)
    
    # Get the partitions for each configuration
    partitions_list = compute_partitions(n_neigh_sequence, neigh_matrices, graph_type, algorithm, seed_sequence, ncores, ecs_thresh, flavor, show_warnings)
    
    # Get an showing the number of clusters obtained for each number of neighbours.
    nn_object_n_clusters = _get_number_of_clusters_obtained_for_each_neighbor(partitions_list)
    
    # Get an object showing the ECC of partitions for each neighbor.
    nn_ecs_object = _ecc_of_partitions_for_each_neighbor(partitions_list, ncores)
    
    # Get number of different partitions for each configuration.
    n_different_partitions = _get_number_different_partitions(partitions_list)
    
    return {
        "n_neigh_k_corresp": nn_object_n_clusters,
        "n_neigh_ec_consistency": nn_ecs_object, 
        "n_different_partitions": n_different_partitions
    }
    
def _assess_nn_stability_umap(embedding, n_neigh_sequence, seed_sequence, ecs_thresh, graph_type, algorithm, umap_arguments, flavor, ncores, show_warnings):
    # Set default UMAP arguments
    if umap_arguments is None:
        umap_arguments = {}
        
    if 'n_neighbors' not in umap_arguments:
        umap_arguments['n_neighbors'] = min(15, len(embedding) - 1)

    graph_types_list = get_graph_types_list(graph_type)
    
    seed_list = _compute_umap_seed_results(embedding, seed_sequence, n_neigh_sequence, graph_types_list, umap_arguments, algorithm, flavor, ncores, show_warnings)
    
    # Merge partitions for each configuration
    partitions_list = {f"UMAP_{graph_type}": {} for graph_type in graph_types_list}
    for graph_type in graph_types_list:
        for n_neigh in n_neigh_sequence:
            partitions = [seed[graph_type][str(n_neigh)] for seed in seed_list]
            partitions_list[f"UMAP_{graph_type}"][str(n_neigh)] = merge_partitions(partitions, ecs_thresh)

    # Number of clusters for each number of neighbors
    nn_object_n_clusters = _get_number_of_clusters_obtained_for_each_neighbor(partitions_list)

    # ECC of partitions for each neighbor
    nn_ecs_object = _ecc_of_partitions_for_each_neighbor(partitions_list, ncores)

    # Number of different partitions for each configuration
    n_different_partitions = _get_number_different_partitions(partitions_list)

    return {
        "n_neigh_k_corresp": nn_object_n_clusters,
        "n_neigh_ec_consistency": nn_ecs_object,
        "n_different_partitions": n_different_partitions
    }
    
def assess_nn_stability(embedding, n_neigh_sequence, ncores, n_repetitions = 100, seed_sequence = None,
                        graph_reduction_type = "PCA", ecs_thresh = 1, graph_type = 1, algorithm = 'leiden', umap_arguments = None, flavor="default", show_warnings = False):
    '''
    Evaluates clustering stability when changing the values of different parameters involved in the graph building step, namely the base embedding, the graph type and the number of neighbours.
    
    ### Args:
        embedding (np.ndarray or pd.DataFrame): A matrix associated with a PCA embedding. Embeddings from other dimensionality reduction techniques can be used.
        n_neigh_sequence (int array): An integer array of the number of nearest neighbours.
        ncores (int): The number of cores to be used for parallel processing.
        n_repetitions (int): The number of repetitions of applying the pipeline with different seeds; ignored if seed_sequence is provided by the user.
        seed_sequence (int array): A custom seed sequence; if the value is None, the sequence will be built starting from 1 with a step of 100.
        graph_reduction_type (str): The graph reduction type, denoting if the graph should be built on either the PCA or the UMAP embedding.
        ecs_thresh (float): The ECS threshold used for merging similar clusterings.
        graph_type (int): Argument indicating whether the graph should be NN (0), SNN (1) or both (2).
        algorithm (str): The community detection algorithm to be used: 'louvain', or 'leiden'. Default and recommended is 'leiden'.
        umap_arguments (dict): Arguments to be used in UMAP construction. Used when graph_reduction_type = "UMAP".
        flavor (str): The implementation of the graph clustering algorithm to be used. Default is 'default'. You can also use 'igraph'.
    
    ### Raises:
        ValueError: If embedding is not a numpy array or pandas DataFrame.
        ValueError: If n_neigh_sequence is not a list or np.ndarray.
        ValueError: If n_repetitions is not an integer.
        ValueError: If graph_reduction_type is not 'PCA' or 'UMAP'.
        ValueError: If ecs_thresh is not an integer or a float.
        ValueError: If graph_type is not an array of integers or if it contain values not between 0 and 2.
        ValueError: If algorithm is not 'louvain' or 'leiden'.
        ValueError: If flavor is not 'default' or 'igraph'.
    
    ### Returns:
        # dict: A dictionary with the following structure:
        #     n_neigh_k_corresp (dict): A dict containing the number of the clusters obtained by running the pipeline multiple times with different seed, number of neighbors and graph type (weighted vs unweigted).
        #     n_neigh_ec_consistency (dict): A dict containing the EC consistency of the partitions obtained at multiple runs when changing the number of neighbors or the graph type.
        #     n_different_partitions (dict): The number of different partitions obtained by each number of neighbors.
    
    ### Example:
    >>> import numpy as np
    >>> import pandas as pd
    >>> import ClustAssessPy as ca
    >>> # Create a PCA embedding.
    >>> pca_emb = np.random.uniform(low=0.0, high=1.0, size=(100, 30))
    >>> pca_emb_df = pd.DataFrame(pca_emb, index=map(str, range(1, 101)), columns=["PCA_" + str(i) for i in range(1, 31)])
    >>> # Assess nn stability using Leiden algorithm
    >>> nn_stability_obj_leiden = ca.assess_nn_stability(pca_emb_df, n_neigh_sequence=[10, 15, 20], n_repetitions = 10, graph_reduction_type="PCA" ,algorithm = 'leiden', graph_type = 2)
    '''
    if not show_warnings:
        if algorithm == "leiden":
            print("FutureWarnings (caused by ScanPy for Leiden) will be suppressed. Set show_warnings=True to see them.")
        
    # Validate parameters stability.
    _validate_parameters_stability(n_neigh_sequence, n_repetitions, ecs_thresh, graph_type, embedding, algorithm, graph_reduction_type, flavor, ncores)

    # Convert number of neighbors to integers.
    n_neigh_sequence = np.sort(n_neigh_sequence).astype(int)

    # convert n_repetitions to integers
    n_repetitions = int(n_repetitions)

    n_cells = embedding.shape[0]
    n_neigh_sequence = n_neigh_sequence[np.where(n_neigh_sequence < n_cells)]

    if len(n_neigh_sequence) == 0:
        warnings.warn(f"The provided values for the `n_neigh_sequence` are greater than the number of cells ({n_cells}). For the downstream analysis, we will set `n_neigh` to {n_cells - 1}.")
        n_neigh_sequence = np.array([n_cells - 1])

    # Create a seed sequence if it's not provided.
    if seed_sequence is None:
        seed_sequence = np.arange(1, n_repetitions * 100, 100)
    else:
        if not isinstance(seed_sequence, (list, np.ndarray)):
            raise ValueError("seed_sequence parameter should be numeric")
        seed_sequence = np.array(seed_sequence).astype(int)

    if graph_reduction_type == "PCA":
        return _assess_nn_stability_pca(embedding, n_neigh_sequence, seed_sequence, ecs_thresh, graph_type, algorithm, flavor, ncores, show_warnings)
    else:
        return _assess_nn_stability_umap(embedding, n_neigh_sequence, seed_sequence, ecs_thresh, graph_type, algorithm, umap_arguments, flavor, ncores, show_warnings)
    
def get_adjacency_matrix_wrapper(embedding, n_neighs):
    '''
    Get the adjacency matrix for the given number of neighbours. 
    This function gives the ability to the scanpy community to get the SNN graph similar to Seurat.
    
    Args:
        embedding: np.ndarray or pd.DataFrame
            A matrix associated with a PCA embedding. Embeddings from other dimensionality reduction techniques can be used.
        n_neighs: int
            The number of nearest neighbours.
    
    Returns:
    --------
        # dict: A dictionary with the following structure:
        #     nn: np.ndarray
        #         The adjacency matrix for the nearest neighbour graph.
        #     snn: np.ndarray
        #         The adjacency matrix for the shared nearest neighbour graph.
    
    Example:
    >>> pca_emb = np.random.uniform(low=0.0, high=1.0, size=(100, 30))
    >>> pca_emb_df = pd.DataFrame(pca_emb, index=map(str, range(1, 101)), columns=["PCA_" + str(i) for i in range(1, 31)])
    >>> # Get SNN matrix from pca using 20 neighbours
    >>> snn_matrix = ca.get_adjacency_matrix_wrapper(pca_emb_df, n_neighs=20)['snn']
    '''
    nn2_res = nn2_indices(embedding, k = n_neighs)
    return process_neigh_matrix(n_neighs, nn2_res, 1)