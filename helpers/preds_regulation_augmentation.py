def augment_pred_confs_with_regulation(configs, lp , program_halt, y_pred_confs):
    import numpy as np
    """
    Multi-label Notes:

    STEP(1): from confidence_scores to binary
              - For multi-label, there is NO dimension for the negative class in the decision layer.
              - In addition, we have used sigmoid activation in the decision layer.
              - Example output for one example pair:  [0.4, 0.6, 0.6]  ... this should be converted to --> [0,1,1]
              - We don't need to do this for y_true, because y_true in its original shape is like [0,1,1], so no need to transform anything really.  
      
              Example:
                   a = np.array([[0.6, 0.6, 0.2],
                                 [0.1, 0.8, 0.8],
                                 [0.2, 0.3, 0.4],
                                 [0.5, 0.5, 0.5]])
    
                   np.where(a >= 0.5 , 1 , 0) --> 
                           array([[1, 1, 0],
                                  [0, 1, 1],
                                  [0, 0, 0],
                                  [1, 1, 1]])
    
    STEP(2): Regulation Augmentation
        if
            Positive_regulation, Negative_regulation, Regulation_of_gene_expression, Regulation_of_proteolysis, Regulation_of_transcription:
        then:
            add Regulation 
    """
    if configs['classification_type'].lower() != 'multi-label':
        program_halt('only multi-label classification is supported for regulation augmentation!')

    #1-first convert y_pred_confs from floating_point confidence scores to binary
    y_pred_confs = np.where(y_pred_confs >= 0.5 , 1 , 0)

    #2-now, augment data
    to_the_left_indices = []
    to_the_left_indices.append(configs['RelationTypeEncoding'].relation_to_one_hot_index_mapping['Positive_regulation>'])
    to_the_left_indices.append(configs['RelationTypeEncoding'].relation_to_one_hot_index_mapping['Negative_regulation>'])
    to_the_left_indices.append(configs['RelationTypeEncoding'].relation_to_one_hot_index_mapping['Regulation_of_gene_expression>'])
    to_the_left_indices.append(configs['RelationTypeEncoding'].relation_to_one_hot_index_mapping['Regulation_of_proteolysis>'])
    to_the_left_indices.append(configs['RelationTypeEncoding'].relation_to_one_hot_index_mapping['Regulation_of_transcription>'])

    to_the_right_indices = []
    to_the_right_indices.append(configs['RelationTypeEncoding'].relation_to_one_hot_index_mapping['Positive_regulation<'])
    to_the_right_indices.append(configs['RelationTypeEncoding'].relation_to_one_hot_index_mapping['Negative_regulation<'])
    to_the_right_indices.append(configs['RelationTypeEncoding'].relation_to_one_hot_index_mapping['Regulation_of_gene_expression<'])
    to_the_right_indices.append(configs['RelationTypeEncoding'].relation_to_one_hot_index_mapping['Regulation_of_proteolysis<'])
    to_the_right_indices.append(configs['RelationTypeEncoding'].relation_to_one_hot_index_mapping['Regulation_of_transcription<'])

    to_the_left_regulation_index = configs['RelationTypeEncoding'].relation_to_one_hot_index_mapping['Regulation>']
    to_the_right_regulation_index = configs['RelationTypeEncoding'].relation_to_one_hot_index_mapping['Regulation<']

    number_of_rows , number_of_columns = y_pred_confs.shape

    #tmp = copy.deepcopy(y_pred_confs)

    for row_index in range(number_of_rows):
        if np.any(y_pred_confs[row_index, to_the_left_indices]==1):
            y_pred_confs[row_index,to_the_left_regulation_index] = 1

        if np.any(y_pred_confs[row_index,to_the_right_indices]==1):
            y_pred_confs[row_index,to_the_right_regulation_index] = 1
    """    
    print("-"*70)
    for row_index in range(number_of_rows):
        before_lbls = [configs['RelationTypeEncoding'].one_hot_index_to_relation_mapping[idx] for idx in list(np.where(tmp[row_index,:] == 1)[0])]
        after_lbls = [configs['RelationTypeEncoding'].one_hot_index_to_relation_mapping[idx] for idx in list(np.where(y_pred_confs[row_index,:] == 1)[0])]
        if len(before_lbls) > 0:
            print (gold_pair_tracking[row_index], before_lbls, after_lbls)

    """
    return y_pred_confs