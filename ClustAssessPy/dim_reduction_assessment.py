from ClustAssessPy._common_imports import sc, multiprocessing, AnnData, np, pd
from ClustAssessPy.basic_functions import element_sim_elscore, weighted_element_consistency
from ClustAssessPy._private_helper_functions import assign_to_nested_dict, nn2_indices, process_neigh_matrix, compute_partition, merge_partitions
import warnings
import concurrent.futures

def _perform_feature_stability_step(data_matrix, feature_set, step, npcs, ncells, resolution, seed_sequence, alg, ecs_thresh, partitions_list, graph_reduction_type, pca_postprocessing, flavor, ncores, show_warnings, umap_arguments):
    if step <= len(feature_set):
        # Feature selection based on the step
        feature_set_subset = feature_set[:step]
        actual_npcs = min(npcs, step // 2)
        step = str(step)
            
        data_matrix_sub = data_matrix.loc[:, feature_set_subset]

        adata = AnnData(data_matrix_sub)
        
        # Compute PCA
        sc.tl.pca(adata, n_comps=actual_npcs, svd_solver="arpack")  
        pca_embeddings = adata.obsm['X_pca']
        
        # Perform post-processing such as harmony batch correction. Responsibility is to the user.
        if pca_postprocessing is not None:
            pca_embeddings = pca_postprocessing(adata)
        
        # Compute the neighborhood graph
        n_neighs = min(20, ncells - 1)
        
        if graph_reduction_type == "PCA":
            nn2_res = nn2_indices(pca_embeddings, k = n_neighs)
        else:
            if umap_arguments is None:
                umap_arguments = {}
            sc.pp.neighbors(adata, n_neighbors = n_neighs)
            
            # Run UMAP with the additional arguments
            sc.tl.umap(adata, **umap_arguments)
            umap_embeddings = adata.obsm['X_umap']
            nn2_res = nn2_indices(umap_embeddings, k = n_neighs)
        
        # Get snn graph
        adj_matrix = process_neigh_matrix(n_neighs, nn2_res, 1)['snn']
        
        # Initialize a dictionary to hold all resolution results for this step
        merged_resolutions_results = {}
        
        # Clustering for each resolution
        for res in resolution:
            cluster_results = []
                    
            # Prepare arguments for multiprocessing
            args = [(adj_matrix, adata, alg, res, seed, flavor, show_warnings) for seed in seed_sequence]
                
            # Run clustering in parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers = ncores) as executor:
                results = executor.map(compute_partition, args)
                
            for arg, result in zip(args, results):
                cluster_results.append({'mb': tuple(result), 'freq': 1, 'seed': arg[-1]})
                        
                # Analyze the cluster results
            merged_resolutions_results[str(res)] = merge_partitions(cluster_results, ecs_thresh = ecs_thresh)
            
        partitions_list[step] = merged_resolutions_results
            
        # Compute the steps_ecc_list
        temp_list = {}
            
        for res in map(str, resolution):
            temp_list[res] =  {
                    'ecc': weighted_element_consistency(
                        [x['mb'] for x in partitions_list[step][res]['partitions']],
                        [x['freq'] for x in partitions_list[step][res]['partitions']]
                    ),
                    'most_frequent_partition': partitions_list[step][res]['partitions'][0],
                    'n_different_partitions': len(partitions_list[step][res]['partitions'])
            }
        
        if graph_reduction_type == "UMAP":
            return pca_embeddings, umap_embeddings, temp_list
        else:
            return pca_embeddings, temp_list
    else:
        raise ValueError("The largest step should be smaller or equal to the number of features.") 
    

def _validate_parameters_feature_stability(data_matrix, feature_set, steps, feature_type, resolution, n_repetitions, graph_reduction_type, npcs, ecs_thresh, algorithm, flavor, ncores):
    if not isinstance(data_matrix, (np.ndarray, pd.DataFrame)):
        raise ValueError("The data matrix parameter should be a matrix")

    if not isinstance(feature_set, (list, tuple)) or not all(isinstance(item, str) for item in feature_set):
        raise ValueError("feature_set parameter should be a vector of strings")

    if not all(feature in data_matrix.columns for feature in feature_set):
        raise ValueError("All features from the feature_set should be found in the data_matrix")
    
    if not (isinstance(steps, int) or (isinstance(steps, (np.ndarray, list, tuple)) and all(isinstance(step, int) for step in steps))):
        raise ValueError("steps parameter should be numeric")

    if not isinstance(feature_type, str):
        raise ValueError("feature_type parameter should be a string")
    
    if not isinstance(resolution, (np.ndarray, list, tuple)) or not all(isinstance(r, (int, float)) for r in resolution):
        raise ValueError("resolution parameter should be a list of floats")

    if not isinstance(n_repetitions, int):
        raise ValueError("n_repetitions parameter should be numeric")

    if graph_reduction_type not in ["PCA", "UMAP"]:
        raise ValueError("graph_reduction_type parameter should take one of these values: 'PCA' or 'UMAP'")

    if not isinstance(npcs, int):
        raise ValueError("npcs parameter should be numeric")

    if not isinstance(ecs_thresh, (int, float)):
        raise ValueError("ecs_thresh parameter should be numeric")
    
    if algorithm not in ["louvain", "leiden"]:
        raise ValueError("algorithm should be either 'louvain' or 'leiden'")
    
    if flavor != "default" and flavor != "igraph":
        raise ValueError("flavor parameter should be 'default' or 'igraph'")
    
    cores_available = multiprocessing.cpu_count()
    if ncores > cores_available:
        raise ValueError(f"The number of cores should be less than or equal to {cores_available}.")
    
    


def assess_feature_stability(data_matrix, feature_set, steps, feature_type,
                             resolution, ncores, n_repetitions=100, seed_sequence=None,
                             graph_reduction_type="PCA", npcs=30, ecs_thresh=1,
                             algorithm='leiden', umap_arguments=None, show_warnings = False, parallel_steps = True, pca_postprocessing = None, flavor = "default"):
    """
    Assess the stability for configurations of feature types and size. Evaluate the stability of clusterings obtained based on incremental subsets of a given feature set.
    
    The algorithm assumes that the feature_set is already sorted when performing the subsetting. For example, if the user wants to analyze highly variable feature set, they should provide them sorted by their variability.
    
    ### Args:
        data_matrix (pandas.DataFrame): A data matrix with rows as cells and columns as features.
        feature_set (list): A list of features feature names that can be found on the rownames of the data matrix.
        steps (list): A list of integers containing the sizes of the subsets.
        feature_type (str): A name associated to the feature_set (e.g. HV - for highly variable genes).
        resolution (list): A list of resolutions to be used for clustering.
        ncores (int): The number of cores to be used for parallel processing.
        n_repetitions (int): The number of repetitions of applying the pipeline with different seeds; ignored if seed_sequence is provided by the user.
        seed_sequence (int array): A custom seed sequence; if the value is None, the sequence will be built starting from 1 with a step of 100.
        graph_reduction_type (str): The graph reduction type, denoting if the graph should be built on either the PCA or the UMAP embedding.
        npcs (int): The number of principal components.
        ecs_thresh (float): The ECS threshold used for merging similar clusterings.
        algorithm (str): The community detection algorithm to be used: 'louvain', or 'leiden'. Default and recommended is 'leiden'.
        umap_arguments (dict): A dictionary containing the arguments to be passed to the UMAP function.
        warnings_verbose (bool): Whether to print warnings or not.
        flavor (str): The implementation of the graph clustering algorithm to be used. Default is 'default'. You can also use 'igraph'.
    
    ### Raises:
        ValueError: If data_matrix is not a pandas DataFrame.
        ValueError: If feature_set is not a list or tuple of strings, or contains elements not in data_matrix.
        ValueError: If any feature in feature_set is not found in data_matrix columns.
        ValueError: If steps is not an integer or a list/tuple of integers.
        ValueError: If feature_type is not a string.
        ValueError: If resolution is not a list or tuple of floats.
        ValueError: If n_repetitions is not an integer.
        ValueError: If graph_reduction_type is not 'PCA' or 'UMAP'.
        ValueError: If npcs is not an integer.
        ValueError: If ecs_thresh is not a numeric value.
        ValueError: If algorithm is not 'louvain' or 'leiden'.
        ValueError: If flavor is not 'default' or 'igraph'.
        
    ### Returns:
        # dict: A dictionary with the following structure:
        #     - pca_list (dict): PCA embeddings for each step.
        #     - embedding_list (dict): UMAP embeddings for each step.
        #     - steps_ecc_list (dict): Contains the element consistency scores for each step and resolution, with each entry as follows:
        #             - ecc (float): The element consistency score for the corresponding step.
        #             - most_frequent_partition (dict): Information about the partition with the highest frequency:
        #                 - mb (list): The partition with the highest frequency.
        #                 - freq (int): The frequency of the most frequent partition.
        #                 - seed (int): The seed used for the most frequent partition.
        #             - n_different_partitions (int): The number of different partitions obtained for the corresponding step.
        #     - incremental_ecs_list (dict): Incremental element consistency scores for each step and resolution.

    ### Example:
    >>> import pandas as pd
    >>> import ClustAssessPy as ca
    >>> import numpy as np
    >>> # Generate a random data matrix
    >>> data_matrix = pd.DataFrame(np.random.rand(1000, 1000))
    >>> data_matrix.columns = [f'gene_{i}' for i in range(1000)] 
    >>> # Define the feature set
    >>> feature_set = data_matrix.columns.tolist()
    >>> # Define the steps
    >>> steps = [500, 1000, 1500]
    >>> # Define the resolutions
    >>> resolution = np.arange(0.1, 1.1, 0.3)
    >>> # Assess the feature stability
    >>> feature_stability_results = ca.assess_feature_stability(data_matrix, feature_set, steps, 'test', resolution, n_repetitions=10,  graph_reduction_type="PCA", npcs=30, algorithm='leiden')"""
   
    warnings.warn("Please ensure data matrix has rows as cells and columns as features.")
    
    if not show_warnings:
        if algorithm == "leiden":
            print("FutureWarnings (caused by ScanPy for Leiden) will be suppressed. Set show_warnings=True to see them.")
    
    # Parameter checks and initial processing
    _validate_parameters_feature_stability(data_matrix, feature_set, steps, feature_type, resolution, n_repetitions, graph_reduction_type, npcs, ecs_thresh, algorithm, flavor, ncores)
    
    ncells = data_matrix.shape[0]
    
    pca_list, embedding_list, partitions_list, steps_ecc_list, incremental_ecs_list = {}, {}, {}, {}, {}
    
    if seed_sequence is None:
        seed_sequence = range(1, n_repetitions * 100, 100) 
        
    if parallel_steps: 
        # Run steps in parallel
        with concurrent.futures.ProcessPoolExecutor(max_workers = ncores) as executor:
            futures = {}
            for step in steps:
                # Map each step to its corresponding future.
                future = executor.submit(_perform_feature_stability_step, data_matrix, feature_set, step, npcs, ncells, resolution, seed_sequence, algorithm, ecs_thresh, partitions_list, graph_reduction_type, pca_postprocessing, flavor, ncores, show_warnings, umap_arguments)
                futures[future] = step

            # Process the results as they complete
            for future in concurrent.futures.as_completed(futures):
                step = futures[future]
                result = future.result()
                
                if graph_reduction_type == "UMAP":
                    pca_embeddings, umap_embeddings, temp_list = result
                    assign_to_nested_dict(embedding_list, feature_type, step, umap_embeddings)
                else:
                    pca_embeddings, temp_list = result

                # Store pca_embeddings and steps_ecc (temp_list)
                assign_to_nested_dict(pca_list, feature_type, str(step), pca_embeddings)
                assign_to_nested_dict(steps_ecc_list, feature_type, str(step), temp_list)
    else:
        for step in steps:
            if graph_reduction_type == "UMAP":
                pca_embeddings, umap_embeddings, temp_list = _perform_feature_stability_step(data_matrix, feature_set, step, npcs, ncells, resolution, seed_sequence, algorithm, ecs_thresh, partitions_list, graph_reduction_type, pca_postprocessing, flavor, ncores, umap_arguments)
                assign_to_nested_dict(embedding_list, feature_type, step, umap_embeddings)
            else:
                pca_embeddings, temp_list = _perform_feature_stability_step(data_matrix, feature_set, step, npcs, ncells, resolution, seed_sequence, algorithm, ecs_thresh, partitions_list, graph_reduction_type, pca_postprocessing, flavor, ncores, umap_arguments)
                
            # Store pca_embeddings and steps_ecc (temp_list)
            assign_to_nested_dict(pca_list, feature_type, str(step), pca_embeddings)
            assign_to_nested_dict(steps_ecc_list, feature_type, str(step), temp_list)
        
    # Compute the incremental_ecs_list
    incremental_ecs_list = {}
    incremental_ecs_list[feature_type] = {}
    if len(steps) > 1:
        for i in range(1, len(steps)):
            temp_list = {}
            for res in map(str, resolution):
                temp_list[res] = element_sim_elscore(
                    steps_ecc_list[feature_type][str(steps[i - 1])][res]['most_frequent_partition']['mb'],
                    steps_ecc_list[feature_type][str(steps[i])][res]['most_frequent_partition']['mb']
                )

            incremental_ecs_list[feature_type][f'{steps[i - 1]}-{steps[i]}'] = temp_list
            
    return {
        'pca_list': pca_list,
        'embedding_list': embedding_list,
        'steps_ecc_list': steps_ecc_list,
        'incremental_ecs_list': incremental_ecs_list
    }