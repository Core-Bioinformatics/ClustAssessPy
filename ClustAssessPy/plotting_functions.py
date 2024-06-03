from ClustAssessPy._common_imports import np, pd, ggplot, aes, geom_boxplot, geom_hline, geom_vline, geom_point, theme_classic, labs, scale_size_continuous, theme, guides, guide_legend, position_dodge, element_text, scale_y_continuous, scale_fill_cmap, ggtitle

def plot_feature_overall_stability_boxplot(feature_objects_assessment_list):
    '''
    Display EC consistency for each feature set and for each step.
    
    ### Args:
        feature_objects_assessment_list (dict): The object returned by the `assess_feature_stability` function.
        
    ### Example:
    >>> np.random.seed(42)
    >>> expr_matrix_values = np.vstack((np.random.rand(25, 10), np.random.uniform(5, 7, (75, 10))))
    >>> expr_matrix = pd.DataFrame(expr_matrix_values)
    >>> # Set the row names and column names
    >>> expr_matrix.index = map(str, range(1, 101))
    >>> expr_matrix.columns = ["feature " + str(i) for i in range(1, 11)]
    >>> feature_stability_result = assess_feature_stability(data_matrix = expr_matrix, feature_set = expr_matrix.columns.tolist(), steps = [5, 10], feature_type = 'type_name', resolution = [0.3, 0.5], n_repetitions=10, algorithm=4)
    >>> plot_feature_overall_stability_boxplot(feature_stability_result)
    '''
    # if it is not a list, make it a list
    if not isinstance(feature_objects_assessment_list, list):
        feature_objects_assessment_list = [feature_objects_assessment_list]
    
    # Get overall stability results in a dataframe. 
    data = []
    for fs_object in feature_objects_assessment_list:
        key = list(fs_object['steps_ecc_list'].keys())[0]
        feature_object = fs_object['steps_ecc_list'][key]
        for type_key, steps_dict in feature_object.items():
            for resolution, results_ecc in steps_dict.items():
                data.append({'#Features': type_key, 'ECC': np.median(results_ecc['ecc']), 'type': key})

    df = pd.DataFrame(data)
    
    # Sort by features
    df['#Features'] = df['#Features'].astype(int)  
    df = df.sort_values(by='#Features')
    df['#Features'] = pd.Categorical(df['#Features'], categories=sorted(df['#Features'].unique()), ordered=True)

    plot = (
        ggplot(df, aes(x='#Features', y='ECC', fill='type'))
        + geom_boxplot(outlier_shape=None, outlier_alpha=0)
        + theme_classic()
    )
    
    return plot.draw()
    

def plot_feature_overall_stability_incremental(feature_objects_assessment_list):
    '''
    Perform an incremental ECS between two consecutive feature steps.
    
    ### Args:
        feature_objects_assessment_list (dict): The object returned by the `assess_feature_stability` function.
        
    ### Example:
    >>> np.random.seed(42)
    >>> expr_matrix_values = np.vstack((np.random.rand(25, 10), np.random.uniform(5, 7, (75, 10))))
    >>> expr_matrix = pd.DataFrame(expr_matrix_values)
    >>> # Set the row names and column names
    >>> expr_matrix.index = map(str, range(1, 101))
    >>> expr_matrix.columns = ["feature " + str(i) for i in range(1, 11)]
    >>> feature_stability_result = assess_feature_stability(data_matrix = expr_matrix, feature_set = expr_matrix.columns.tolist(), steps = [5, 10], feature_type = 'type_name', resolution = [0.3, 0.5], n_repetitions=10, algorithm=4)
    >>> plot_feature_overall_stability_incremental(feature_stability_result)
    '''
    # if it is not a list, make it a list
    if not isinstance(feature_objects_assessment_list, list):
        feature_objects_assessment_list = [feature_objects_assessment_list]
    
    df = pd.DataFrame(columns=['#Features', 'ECS', 'type'])
    
    # Get incremental stability results in a dataframe.
    for fs_object in feature_objects_assessment_list:
        key = list(fs_object['incremental_ecs_list'].keys())[0]
        feature_object = fs_object['incremental_ecs_list'][key]
        for type_key, steps_dict in feature_object.items():
            for resolution, results_ecs in steps_dict.items():
                # Temporary df to hold the results for this resolution
                temp_df = pd.DataFrame({
                    '#Features': [type_key],
                    'ECS': [np.median(results_ecs)],
                    'type': [key]
                })
                # Append the temporary DataFrame to the main DataFrame
                df = pd.concat([df, temp_df], ignore_index=True)
    
    # Sort by features order (order that they appear in the incremental_ecs list).
    features_order = df['#Features'].drop_duplicates().tolist()
    df['#Features'] = pd.Categorical(df['#Features'], categories=features_order, ordered=True)

    plot = (
        ggplot(df, aes(x='#Features', y='ECS', fill='type'))
        + geom_boxplot(outlier_shape=None, outlier_alpha=0)
        + theme_classic()
    )
    
    return plot.draw()


    
def plot_clustering_overall_stability(clust_object, value_type="k", summary_function=np.median):
    '''
    Display EC consistency across clustering method.
    
    ### Args:
        clust_object (dict): The object returned by the `assess_clustering_stability` function.
        value_type (str): The type of value to be used for grouping the results. Can be either `k` or `resolution`.
        summary_function (function): The function to be used for summarizing the ECC values.
        
    ### Example:
    >>> pca_emb = np.random.rand(100, 30)
    >>> # Create an AnnData object from pca_emb
    >>> adata = sc.AnnData(X=pca_emb)
    >>> # Run scanpy.pp.neighbors to get the connectivities as an adjacency matrix
    >>> sc.pp.neighbors(adata, n_neighbors=10, use_rep='X')
    >>> adjacency_matrix = adata.obsp['connectivities']
    >>> clust_obj = assess_clustering_stability(adjacency_matrix, resolution=[0.1, 0.3, 0.5], n_repetitions=10, algorithm=[1,4])
    >>> plot_k_resolution_corresp(clust_obj)
    
    '''
    if value_type not in ["k", "resolution"]:
        raise ValueError("`value_type` should contain either `k` or `resolution`")

    # Extracting ECC values and applying the summary function
    ecc_vals = {
        method: {
            val_type: summary_function(data['ecc'])
            for val_type, data in alg.items()
        }
        for method, alg in clust_object[f"split_by_{value_type}"].items()
    }
    
    melted_df = pd.DataFrame([
        {'Method': method, value_type: val_type, 'ECC': ecc}
        for method, val_types in ecc_vals.items()
        for val_type, ecc in val_types.items()
    ])

    plot = (
        ggplot(melted_df, aes(x='Method', y='ECC', fill='Method')) +
        geom_boxplot() +
        theme_classic() +
        ggtitle(f"Overall clustering stability grouped by {value_type}")
    )

    return plot.draw()
    

def plot_k_n_partitions(clust_object, color_information="ecc", y_step=5, pt_size_range=(1.5, 4), dodge_width=0.3, summary_function=np.median):
    '''
    For each configuration provided in clust_object, display how many different partitions with the same number of clusters can be obtained by changing the seed.
    
    ### Args:
        clust_object (dict): The object returned by the `assess_clustering_stability` function.
        color_information (str): The information to be used for coloring the points. Can be either `ecc` or `freq_part`.
        y_step (int): The step for the y-axis.
        pt_size_range (tuple): The range of point sizes.
        dodge_width (float): The width of the dodging.
        summary_function (function): The function to be used for summarizing the ECC values.
        
    ### Example:
    >>> pca_emb = np.random.rand(100, 30)
    >>> # Create an AnnData object from pca_emb
    >>> adata = sc.AnnData(X=pca_emb)
    >>> # Run scanpy.pp.neighbors to get the connectivities as an adjacency matrix
    >>> sc.pp.neighbors(adata, n_neighbors=10, use_rep='X')
    >>> adjacency_matrix = adata.obsp['connectivities']
    >>> clust_obj = assess_clustering_stability(adjacency_matrix, resolution=[0.1, 0.3, 0.5], n_repetitions=10, algorithm=[1,4])
    >>> plot_k_n_partitions(clust_obj)
    '''
    if color_information not in ["ecc", "freq_part"]:
        raise ValueError("colour_information can be either `ecc` or `freq_part`")

    if 'split_by_k' in clust_object:
        clust_object = clust_object['split_by_k']

    data_list = []
    max_n_part = 0

    for configuration, partition_object in clust_object.items():
        for k, data in partition_object.items():
            n_partitions = len(data['partitions'])
            first_occ = max(partition['freq'] for partition in data['partitions'])
            total_occ = sum(partition['freq'] for partition in data['partitions'])
            freq_part = first_occ / total_occ
            ecc = summary_function(data['ecc'])

            data_list.append({
                # convert k to numeric
                'n.clusters': pd.to_numeric(k),
                'n.partitions': n_partitions,
                'configuration': configuration,
                'first.occ': first_occ,
                'total.occ': total_occ,
                'freq_part': freq_part,
                'ecc': ecc
            })

            max_n_part = max(max_n_part, n_partitions)

    overall_total_occ = sum(item['total.occ'] for item in data_list)
    for item in data_list:
        item['frequency_k'] = item['total.occ'] / overall_total_occ

    unique_parts = pd.DataFrame(data_list)
    y_breaks = list(range(0, max_n_part + y_step, y_step))
    unique_parts['n.clusters'] = pd.Categorical(unique_parts['n.clusters'], ordered=True)

    plot = (
        ggplot(unique_parts, aes(
            x='n.clusters',
            y='n.partitions',
            shape='configuration',
            size='frequency_k',
            fill=color_information,  # Replace with the correct column name
            group='configuration'
        )) +
        scale_y_continuous(breaks=y_breaks) +
        scale_size_continuous(range=pt_size_range, guide='legend') +
        geom_hline(yintercept=list(range(0, max_n_part + 1, y_step)), linetype="dashed", color="#e3e3e3") +
        geom_point(position=position_dodge(width=dodge_width)) +
        theme_classic() +
        labs(x="k", y="# partitions") +
        guides(shape=guide_legend(override_aes={'size': max(pt_size_range)}))
    )
    
    return plot.draw()

def plot_k_resolution_corresp(clust_object, color_information="ecc", dodge_width = 0.3, pt_size_range = [1.5, 4], summary_function=np.median):
    '''
    For each configuration provided in the clust_object, display what number of clusters appear for different values of the resolution parameters.
    
    ### Args:
        clust_object (dict): The object returned by the `assess_clustering_stability` function.
        color_information (str): The information to be used for coloring the points. Can be either `ecc` or `freq_part`.
        dodge_width (float): The width of the dodging.
        pt_size_range (float array): The range of point sizes.
        summary_function (function): The function to be used for summarizing the ECC values.
        
    ### Example:
    >>> pca_emb = np.random.rand(100, 30)
    >>> # Create an AnnData object from pca_emb
    >>> adata = sc.AnnData(X=pca_emb)
    >>> # Run scanpy.pp.neighbors to get the connectivities as an adjacency matrix
    >>> sc.pp.neighbors(adata, n_neighbors=10, use_rep='X')
    >>> adjacency_matrix = adata.obsp['connectivities']
    >>> clust_obj = assess_clustering_stability(adjacency_matrix, resolution=[0.1, 0.3, 0.5], n_repetitions=10, algorithm=[1,4])
    >>> plot_k_resolution_corresp(clust_obj)
    '''
    if color_information not in ["ecc", "freq_k"]:
        raise ValueError("color_information can be either `ecc` or `freq_k`")

    appearances_df = pd.DataFrame()

    # Iterate through the clustering results
    for method, resolutions in clust_object['split_by_resolution'].items():
        for resolution, res_object in resolutions.items():
            # Calculate total runs for normalization
            n_runs = sum(partition['freq'] for cluster in res_object['clusters'].values() for partition in cluster['partitions'])

            temp_data = {
                'method': [],
                'resolution_value': [],
                'number_clusters': [],
                'freq_partition': [],
                'ecc': [],
                'freq_k': []
            }

            for k, cluster_info in res_object['clusters'].items():
                # Calculate the sum of frequencies for this cluster size (k)
                freq_k = sum(partition['freq'] for partition in cluster_info['partitions'])
                
                temp_data['method'].append(method)
                temp_data['resolution_value'].append(resolution)
                temp_data['number_clusters'].append(k)
                temp_data['freq_partition'].append(cluster_info['partitions'][0]['freq']/sum(partition['freq'] for partition in cluster_info['partitions']))
                temp_data['ecc'].append(summary_function(cluster_info['ecc']))
                temp_data['freq_k'].append(freq_k)

            # Convert to DataFrame and normalize
            temp_df = pd.DataFrame(temp_data)
            temp_df['freq_k'] /= n_runs
            appearances_df = pd.concat([appearances_df, temp_df], ignore_index=True)

    appearances_df['number_clusters'] = pd.to_numeric(appearances_df['number_clusters'])
    appearances_df['resolution_value'] = pd.Categorical(appearances_df['resolution_value'])

    plot = (
        ggplot(appearances_df, aes(
            y='number_clusters',
            x='factor(resolution_value)',
            size='freq_partition',
            fill=color_information,
            shape='method',
            group='method'
        )) +
        geom_hline(yintercept=appearances_df['number_clusters'].unique(), linetype="dashed", color="#e3e3e3") +
        geom_point(position= position_dodge(width=dodge_width)) +
        theme_classic() +
        scale_fill_cmap(guide="colorbar") +
        labs(x="resolution", y="k") +
        scale_size_continuous(range=pt_size_range, guide="legend") +
        theme(axis_text_x=element_text(angle=90, hjust=1)) +
        guides(shape=guide_legend(override_aes={'size': max(pt_size_range)}))
    )
    
    return plot.draw()    

def plot_n_neigh_ecs(nn_ecs_object):
    '''
    Display, for all configurations consisting in different number of neighbors, graph types and base embeddings, 
    the EC Consistency of the partitions obtained over multiple runs.
    
    ### Args:
        nn_ecs_object (dict): The object returned by the `assess_nn_stability` function.
    
    ### Example:
    >>> pca_emb = np.random.rand(100, 30)
    >>> nn_stability_obj = assess_nn_stability(pca_emb, n_neigh_sequence= [10, 15, 20], n_repetitions = 10, algorithm = 4)
    >>> plot_n_neigh_ecs(nn_stability_obj)
    '''
    data = []
    for config_name, config in nn_ecs_object['n_neigh_ec_consistency'].items():
        for n_neigh, values in config.items():
            for ecc in values:
                data.append({'ECC': ecc, 'n_neigh': n_neigh, 'config_name': config_name})
    df = pd.DataFrame(data)

    # Convert 'n_neigh' to a categorical variable and sort its levels
    df['n_neigh'] = pd.Categorical(df['n_neigh'], ordered=True)
    df['n_neigh'] = df['n_neigh'].cat.reorder_categories(sorted(df['n_neigh'].unique(), key=lambda x: float(x)))

    
    plot = (
        ggplot(df, aes(x='n_neigh', y='ECC', fill='config_name'))
        + geom_boxplot()
        + theme_classic()
    )
    
    return plot.draw()