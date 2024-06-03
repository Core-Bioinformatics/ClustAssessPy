from ClustAssessPy._common_imports import AnnData, csr_matrix, multiprocessing, np
from ClustAssessPy.basic_functions import weighted_element_consistency
from ClustAssessPy._private_helper_functions import merge_resolutions, compute_partition, merge_partitions
import concurrent.futures

def _validate_parameters_clustering_stability(resolution, n_repetitions, ecs_thresh, algorithm, flavor, ncores):
    if not isinstance(resolution, (np.ndarray, list, tuple)) or not all(isinstance(r, (int, float)) for r in resolution):
        raise ValueError("resolution parameter should be a list of floats")

    if not isinstance(n_repetitions, int):
        raise ValueError("n_repetitions parameter should be numeric")
    
    if not isinstance(ecs_thresh, (int, float)):
        raise ValueError("ecs_thresh parameter should be numeric")

    if not isinstance(algorithm, (list, tuple)) or not all(isinstance(item, str) for item in algorithm):
        raise ValueError("algorithm should be either ['louvain'] or ['leiden'] or both ['louvain', 'leiden']")
    
    if flavor != "default" and flavor != "igraph":
        raise ValueError("flavor parameter should be 'default' or 'igraph'")
    
    cores_available = multiprocessing.cpu_count()
    if ncores > cores_available:
        raise ValueError(f"The number of cores should be less than or equal to {cores_available}.")
    

def _evaluate_clustering_resolution(graph_adjacency_matrix, adata, alg, res, seed_sequence, ncores, ecs_thresh, flavor, show_warnings):
    different_partitions = []
            
    args = [(graph_adjacency_matrix, adata, alg, res, seed, flavor, show_warnings) for seed in seed_sequence]

    # Parallel processing
    with concurrent.futures.ProcessPoolExecutor(max_workers=ncores) as executor:
        partitions = executor.map(compute_partition, args)
            
    different_partitions.extend(partitions)

    different_partitions_grouped = {}   
            
    # get partition directly but also index as well
    for i, partition in enumerate(different_partitions):
        k = len(np.unique(partition))
                
        if str(k) not in different_partitions_grouped:
            different_partitions_grouped[str(k)] = []
                    
        different_partitions_grouped[str(k)].append({
            'mb': partition.tolist(),
            'freq': 1,
            'seed': seed_sequence[i]
        })
            
    # Merge partitions    
    for k in different_partitions_grouped.keys():
        different_partitions_grouped[k] = merge_partitions(different_partitions_grouped[k], ecs_thresh=ecs_thresh, order=True, check_ties=True)
            
    # Calculate weighted ECC for runs of current algorithm and resolution.        
    partitions_values = list(different_partitions_grouped.values())
    partitions_array = [item['partitions'] for item in partitions_values]
    flattened_partitions = [partition for sublist in partitions_array for partition in sublist]
    weights = [partition['freq'] for partition in flattened_partitions]     
    partitions_mb = [partition['mb'] for partition in flattened_partitions]
            
    ecc = weighted_element_consistency(partitions_mb, weights)

    return {'clusters': different_partitions_grouped, 'ecc': ecc}


def assess_clustering_stability(graph_adjacency_matrix, resolution, ncores,
                                n_repetitions=100,
                                seed_sequence=None, ecs_thresh=1,
                                algorithms=['louvain', 'leiden'],
                                flavor="default", show_warnings=False):
    '''
    Evaluates the stability of different graph clustering methods in the clustering pipeline. 
    The method will iterate through different values of the resolution parameter and compare, using the EC Consistency score, the partitions obtained at different seeds.
    
    ### Args:
        graph_adjacency_matrix (pandas.DataFrame): A data matrix with rows as cells and columns as features.
        resolution (list): A list of resolutions to be used for clustering.
        ncores (int): The number of cores to be used for parallel processing.
        n_repetitions (int): The number of repetitions of applying the pipeline with different seeds; ignored if seed_sequence is provided by the user.
        seed_sequence (int array): A custom seed sequence; if the value is None, the sequence will be built starting from 1 with a step of 100.
        ecs_thresh (float): The ECS threshold used for merging similar clusterings.
        algorithms (str array): The community detection algorithm to be used: ['louvain'], or ['leiden'], or both ['louvain', 'leiden'].
        flavor (str): The implementation of the graph clustering algorithm to be used. Default is 'default'. You can also use 'igraph'.
    
    ### Raises:
        ValueError: If resolution is not a list or tuple of floats.
        ValueError: If n_repetitions is not an integer.
        ValueError: If ecs_thresh is not a numeric value.
        ValueError: If algorithm is not 'louvain' or 'leiden'.
        ValueError: If flavor is not 'default' or 'igraph'.
        
    ### Returns:
        # dict: A dictionary with the following structure:
        #     - split_by_resolution (dict): Stability results when evaluating the clustering stability by resolution. 
        #     - split_by_k (dict): Stability results when evaluating the clustering stability by the number of clusters (k). 
    
    Example:
    >>> import numpy as np
    >>> import pandas as pd
    >>> import ClustAssessPy as ca
    >>> # Create a PCA embedding.
    >>> pca_emb = np.random.uniform(low=0.0, high=1.0, size=(100, 30))
    >>> pca_emb_df = pd.DataFrame(pca_emb, index=map(str, range(1, 101)), columns=["PCA_" + str(i) for i in range(1, 31)])
    >>> # Get SNN matrix from pca using 20 neighbours
    >>> adj_mat = ca.get_adjacency_matrix_wrapper(pca_emb_df, n_neighs=20)['snn']
    >>> clust_stability = ca.assess_clustering_stability(adj_mat, resolution=[0.1, 0.3, 0.5, 0.7, 0.9], n_repetitions=10, algorithms=['louvain','leiden'])
    >>> # ------------------------------------------------
    >>> # ALTERNATIVELY WITH SCANPY - assuming you have an adata object
    >>> # ------------------------------------------------
    >>> import scanpy as sc
    >>> sc.pp.neighbors(adata, n_neighbors=20)
    >>> adj_mat = adata.obsp['connectivities']
    >>> clust_stability = ca.assess_clustering_stability(adj_mat, resolution=[0.1, 0.3, 0.5, 0.7, 0.9], n_repetitions=10, algorithms=['louvain','leiden']) 
    '''
    if not show_warnings:
        if 'leiden' in algorithms:
            print("FutureWarnings (caused by ScanPy for Leiden) will be suppressed. Set show_warnings=True to see them.")
         
    # Convert adjacency matrix to csr_matrix if it's not already
    if not isinstance(graph_adjacency_matrix, csr_matrix):
        graph_adjacency_matrix = csr_matrix(graph_adjacency_matrix)
    
    _validate_parameters_clustering_stability(resolution, n_repetitions, ecs_thresh, algorithms, flavor, ncores)
        
    if seed_sequence is None:
        seed_sequence = range(1, n_repetitions * 100, 100)
    else:
        if not isinstance(seed_sequence, (list, np.ndarray)):
            raise ValueError("seed_sequence parameter should be numeric")
        seed_sequence = np.array(seed_sequence).astype(int)
        
    result_object = {}

    adata = AnnData(graph_adjacency_matrix)
    
    for alg in algorithms: 
        result_object[alg] = {}
        
        with concurrent.futures.ProcessPoolExecutor(max_workers = ncores) as executor:
            futures = {executor.submit(_evaluate_clustering_resolution, graph_adjacency_matrix, adata, alg, res, seed_sequence, ncores, ecs_thresh, flavor, show_warnings):res for res in resolution}

        for future in concurrent.futures.as_completed(futures.keys()):
            res = futures[future]
            result_object[alg][str(res)] = future.result()
            
    algorithm_names = list(result_object.keys())
    split_by_k = {alg_name: merge_resolutions(result_object[alg_name]) for alg_name in algorithm_names if alg_name in result_object}

    return {
        'split_by_resolution': result_object,
        'split_by_k': split_by_k
    }