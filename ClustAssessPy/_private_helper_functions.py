from ClustAssessPy.basic_functions import weighted_element_consistency, element_sim_matrix
from ClustAssessPy._common_imports import np, sc, AnnData, cKDTree, nx
from snn_functions import computeSNN, pruneSNN, getNNmatrix
import concurrent.futures
import warnings
import contextlib

@contextlib.contextmanager
def suppressed_warnings(surpress_warnings):
    if not surpress_warnings:
        yield
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            yield

def get_highest_prune_param(nn_matrix, n_neigh):
    snn_matrix = computeSNN(nn_matrix, n_neigh, 0)

    # Create a graph from the adjacency matrix
    g = nx.from_scipy_sparse_array(snn_matrix, edge_attribute='weight')

    # Get the number of connected components in the original graph
    target_n_conn_comps = nx.number_connected_components(g)

    # Generate possible pruning values
    possible_values = [i / (2 * n_neigh - i) for i in range(n_neigh + 1)]

    start_n, stop_n = 1, len(possible_values)
    prev_g = g

    while start_n <= stop_n:
        middle = (stop_n + start_n) // 2
        # Prune edges
        edges_to_remove = [(u, v) for u, v, w in prev_g.edges(data='weight') if w <= possible_values[middle]]
        current_g = prev_g.copy()
        current_g.remove_edges_from(edges_to_remove)

        # Check the number of connected components
        current_n_conn_comps = nx.number_connected_components(current_g)

        if current_n_conn_comps > target_n_conn_comps:
            stop_n = middle
            if start_n == stop_n:
                start_n = middle - 1
        else:
            start_n = middle + 1
            prev_g = current_g
            if start_n == stop_n:
                break

    # Prune the snn matrix using the found prune value
    pruned_snn_matrix = pruneSNN(snn_matrix, possible_values[middle])

    return {
        'prune_value': possible_values[middle],
        'adj_matrix': pruned_snn_matrix
    }

def merge_partitions(partition_list, ecs_thresh=1, order=True, check_ties=False):
    # Parameter checks
    if not isinstance(ecs_thresh, (int, float)):
        raise ValueError("ecs_thresh parameter should be numeric")
    if not isinstance(order, bool):
        raise ValueError("order parameter should be logical")
    
    part_list = []
    if ecs_thresh == 1:
        part_list = merge_identical_partitions(partition_list, order)
    else:
        part_list = merge_partitions_ecs(partition_list, ecs_thresh, order)

    # Check for ties
    if check_ties and len(part_list) > 1:
        part_list = check_for_ties_and_calculate_ecc(part_list)
        return part_list
    else:
        return {'partitions': part_list, 'ecc': 1.0}

def check_for_ties_and_calculate_ecc(partition_list):
    weights = list(map(lambda partition: partition['freq'], partition_list))
    partitions_mb = list(map(lambda partition: partition['mb'], partition_list))
    
    consistency, ecs_matrix = weighted_element_consistency(partitions_mb, weights, True)
    
    # Check for ties in the highest frequency
    max_freq = max(partition['freq'] for partition in partition_list)

    # Identify partitions with max frequency
    max_freq_partitions = [partition for partition in partition_list if partition['freq'] == max_freq]

    # If there's more than one partition with max frequency, check for the highest similarity
    nmax = len(max_freq_partitions)
    
    if nmax > 1:
        highest_sim_index = find_highest_similarity_partition(ecs_matrix, nmax)

        # Swap the partitions if needed
        if highest_sim_index != 0:
            temp_mb = partition_list[0]['mb']
            partition_list[0]['mb'] = partition_list[highest_sim_index]['mb']
            partition_list[highest_sim_index]['mb'] = temp_mb


    result = {
        "partitions": partition_list,
        "ecc": consistency
    }
    
    return result

def find_highest_similarity_partition(ecs_matrix, nmax):
    row_sums = np.sum(ecs_matrix, axis=1)
    col_sums = np.sum(ecs_matrix, axis=0)
    
    n = ecs_matrix.shape[0]
    average_agreement = (row_sums + col_sums - 2) / (n - 1)

    # Limit to the first nmax elements
    limited_agreement = average_agreement[:nmax]

    # Find the index of the maximum value
    highest_sim_index = np.argmax(limited_agreement)

    return highest_sim_index

def merge_partitions_ecs(partition_list, ecs_thresh=0.99, order=True):
    nparts = len(partition_list)
    if nparts == 1:
        return partition_list

    # Calculate the pairwise ecs between the partitions
    sim_matrix = element_sim_matrix([x['mb'] for x in partition_list])
    np.fill_diagonal(sim_matrix, np.nan)

    partition_groups = {str(i): [i] for i in range(nparts)}

    while len(partition_groups) > 1:
        # Check if the maximum similarity is below the threshold
        if np.nanmax(sim_matrix) < ecs_thresh:
            break

        index = np.nanargmax(sim_matrix)
        first_cluster, second_cluster = index % nparts, index // nparts

        partition_groups[str(first_cluster)].extend(partition_groups.pop(str(second_cluster)))

        # Update similarities in a single-linkage fashion
        for i in map(int, partition_groups.keys()):
            if first_cluster < i:
                sim_matrix[first_cluster, i] = np.nanmin([sim_matrix[first_cluster, i], sim_matrix[second_cluster, i], np.nan], initial=np.inf)
            else:
                sim_matrix[i, first_cluster] = np.nanmin([sim_matrix[i, first_cluster], sim_matrix[i, second_cluster], np.nan], initial=np.inf)

        sim_matrix[second_cluster, :] = np.nan
        sim_matrix[:, second_cluster] = np.nan

    merged_partitions = []
    for kept_partition in partition_groups.values():
        partitions_to_merge = [partition_list[j] for j in kept_partition]
        merged_partition = {'mb': partitions_to_merge[0]['mb'], 'freq': sum(p['freq'] for p in partitions_to_merge)}
        merged_partitions.append(merged_partition)

    if order:
        merged_partitions.sort(key=lambda x: x['freq'], reverse=True)

    return merged_partitions

def merge_identical_partitions(clustering_list, order=True):
    if len(clustering_list) <= 1:
        return clustering_list

    # Merge identical partitions
    merged_partitions = []
    for partition in clustering_list:
        found = False
        for merged in merged_partitions:
            if are_identical_memberships(partition['mb'], merged['mb']):
                merged['freq'] += partition['freq']
                found = True
                break
        if not found:
            merged_partitions.append(partition)

    if order:
        merged_partitions.sort(key=lambda x: x['freq'], reverse=True)

    return merged_partitions

def are_identical_memberships(mb1, mb2):
    # Calculate the contingency table
    contingency_table = calculate_contigency_table(mb1, mb2) != 0

    if contingency_table.shape[0] != contingency_table.shape[1]:
        return False

    # Check for only singleton clusters
    if contingency_table.shape[0] == len(mb1):
        return True

    # Check if any column has two or more nonzero entries
    no_different_elements = np.sum(contingency_table, axis=0)
    if any(no_different_elements != 1):
        return False

    # Check if any row has two or more nonzero entries
    no_different_elements = np.sum(contingency_table, axis=1)
    return all(no_different_elements == 1)

def calculate_contigency_table(a, b):
    # Convert categories to numerical values
    category_to_num_a = {cat: i for i, cat in enumerate(sorted(set(a)))}
    category_to_num_b = {cat: i for i, cat in enumerate(sorted(set(b)))}

    a_num = [category_to_num_a[cat] for cat in a]
    b_num = [category_to_num_b[cat] for cat in b]

    minim_mb1, minim_mb2 = min(a_num), min(b_num)
    maxim_mb1, maxim_mb2 = max(a_num), max(b_num)
    nClustersA = maxim_mb1 - minim_mb1 + 1
    nClustersB = maxim_mb2 - minim_mb2 + 1

    result = np.zeros((nClustersA, nClustersB), dtype=int)

    for i in range(len(a_num)):
        result[a_num[i] - minim_mb1, b_num[i] - minim_mb2] += 1

    return result

def merge_resolutions(res_obj):
    clusters_obj = {}
    
    for res in res_obj.keys():
        for k in res_obj[res]['clusters'].keys():
            if k not in clusters_obj:
                clusters_obj[k] = res_obj[res]['clusters'][k]['partitions']
            else :
                clusters_obj[k] += res_obj[res]['clusters'][k]['partitions']
                
    for i, key in enumerate(clusters_obj.keys()):
        clusters_obj[key] = merge_partitions(clusters_obj[key], check_ties=True)

    return clusters_obj

def assign_to_nested_dict(outer_dict, outer_key, inner_key, value):
    # Assigns a value to a nested dictionary.
    if outer_key not in outer_dict:
        outer_dict[outer_key] = {}
    outer_dict[outer_key][inner_key] = value
    
def nn2_indices(data, k):
    tree = cKDTree(data)
    _, indices = tree.query(data, k=k)
    return indices + 1

def process_neigh_matrix(n_neigh, nn2_res, graph_type):
    # Compute the nearest neighbor matrix
    nn_matrix = getNNmatrix(nn2_res, n_neigh, 0, -1)

    if graph_type > 0:
        # Compute the SNN matrix
        snn_matrix_result = get_highest_prune_param(nn_matrix['nn'], n_neigh)
        snn_matrix = snn_matrix_result['adj_matrix']

        return {'nn': nn_matrix['nn'], 'snn': snn_matrix}

    return {'nn': nn_matrix['nn']}

def get_partitionings(neigh_matrix, algorithm, seed, graph_type, flavor, show_warnings):
    adata = AnnData(neigh_matrix[graph_type])

    if algorithm == 'louvain':
        sc.tl.louvain(
            adata,
            random_state = seed,
            key_added = "clusters",
            adjacency = neigh_matrix[graph_type],
            resolution = None if flavor == "igraph" else 0.8, # to match Seurat and ClustAssess in R
            directed = True, 
            use_weights = True,
            flavor = "igraph" if flavor == "igraph" else "vtraag"
        )
    elif algorithm == 'leiden':
        with suppressed_warnings(not show_warnings):
            sc.tl.leiden(
                adata,
                random_state = seed,
                key_added = "clusters",
                adjacency = neigh_matrix[graph_type],
                resolution = None if flavor == "igraph" else 0.8, # to match Seurat and ClustAssess in R
                n_iterations = 10,  
                directed = False if flavor == "igraph" else True, 
                use_weights = True,
                flavor = "igraph" if flavor == "igraph" else "leidenalg"
            )
    
    return {
        "mb": adata.obs["clusters"].astype(int),
        "freq": 1,
        "seed": seed,
    }

def get_graph_types_list(graph_type):
    if graph_type == 0:
        return ["nn"]
    elif graph_type == 1:
        return ["snn"]
    else:
        return ["nn", "snn"]

def compute_partitions(n_neigh_sequence, neigh_matrices, graph_type, algorithm, seed_sequence, ncores, ecs_thresh, flavor, show_warnings):
    graph_types_list = get_graph_types_list(graph_type)
        
    partitions_list = {"PCA_nn": {}, "PCA_snn": {}}
    
    for n_neigh in n_neigh_sequence:
        shared_neigh_matrix = neigh_matrices[n_neigh]
        
        for graph_type in graph_types_list:
            # Parallel
            with concurrent.futures.ProcessPoolExecutor(max_workers = ncores) as executor:
                # Submit tasks
                future_to_seed = {
                    executor.submit(get_partitionings, shared_neigh_matrix, algorithm, seed, graph_type, flavor, show_warnings): seed
                    for seed in seed_sequence
                }

                # Collect results as they complete
                partitions_list_temp = [future.result() for future in concurrent.futures.as_completed(future_to_seed)]
            
            partitions_list[f"PCA_{graph_type}"][str(n_neigh)] = merge_partitions(
                partitions_list_temp, ecs_thresh=ecs_thresh)
            
    return partitions_list

def compute_partition(args):
    graph_adjacency_matrix, adata, alg, res, seed, flavor, show_warnings = args
    
    if alg == 'louvain':
        sc.tl.louvain(
            adata, 
            adjacency=graph_adjacency_matrix, 
            resolution = None if flavor == "igraph" else res, 
            random_state=seed, 
            directed=True, 
            use_weights=True,
            flavor= "igraph" if flavor == "igraph" else "vtraag"
        )
    elif alg == 'leiden':
        with suppressed_warnings(not show_warnings):
            sc.tl.leiden(
                adata, 
                adjacency = graph_adjacency_matrix, 
                # resolution = None if flavor == "igraph" else res, 
                resolution = res, 
                random_state = seed, 
                directed = False if flavor == "igraph" else True, 
                n_iterations = 10, 
                use_weights = True,
                flavor= "igraph" if flavor == "igraph" else "leidenalg"
            )

    return adata.obs[alg]