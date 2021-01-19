import numpy as np
import pandas as pd
import json


def read_data(incorrect_type, dir_path='../data/', existing_company_only=False, sample=None):
    assert incorrect_type in ['inverse', 'boundary', 'correct']
    

    id_lists = get_id_lists(dir_path)
    slope_df = pd.read_csv(dir_path+'clean_data.csv')
    slope_df['y_boundary'] = [json.loads(bound) for bound in slope_df['y_boundary']]
    slope_df = slope_df.set_index('id')

    if existing_company_only:
        ticker_path = dir_path+'tickers_list.txt'
        tickers = []
        with open(ticker_path, 'r') as f:
            for line in f:
                tickers.append(line.strip())
        slope_df = slope_df[slope_df['ticker'].isin(tickers)].copy()

    if sample:
        print(f'Taking {sample} samples')
        slope_df = slope_df.iloc[:sample]

    ###########################################################################
    if incorrect_type == 'correct':
        id_lists = [list(set(l).intersection(set(slope_df.index))) for l in id_lists]

        train_df = slope_df.loc[id_lists[0]]
        val_df = slope_df.loc[id_lists[1]]
        test_df = slope_df.loc[id_lists[2]]

        y_train = train_df['label_from_score'].values.astype('int')
        y_val = val_df['label_from_score'].values.astype('int')
        y_test = test_df['label_from_score'].values.astype('int')

    ###########################################################################
    if incorrect_type == 'inverse':
        temp_slope_df = slope_df[~((slope_df['sentiment_correct']<1)&(slope_df['y_inverse']==2))]  # remove incorrect class 2
        id_lists = [list(set(l).intersection(set(temp_slope_df.index))) for l in id_lists]

        train_df = temp_slope_df.loc[id_lists[0]+id_lists[3]]
        val_df = temp_slope_df.loc[id_lists[1]+id_lists[4]]
        test_df = temp_slope_df.loc[id_lists[2]+id_lists[5]]

        y_train = train_df['y_inverse'].values.astype('int')
        y_val = val_df['y_inverse'].values.astype('int')
        y_test = test_df['y_inverse'].values.astype('int')

    ###########################################################################
    if incorrect_type == 'boundary':
        slope_df['possible_classes'] = slope_df['y_boundary']

        id_lists = [list(set(l).intersection(set(slope_df.index))) for l in id_lists]

        train_df = slope_df.loc[id_lists[0]+id_lists[3]]
        val_df = slope_df.loc[id_lists[1]+id_lists[4]]
        test_df = slope_df.loc[id_lists[2]+id_lists[5]]

        train_df = train_df.explode('y_boundary')
        val_df = val_df.explode('y_boundary')
        # test_df = test_df.explode('y_boundary')
        
        y_train = train_df['y_boundary'].values.astype('int')
        y_val = val_df['y_boundary'].values.astype('int')
        y_test = test_df['label_from_score'].values.astype('int')

        # z_train = train_df['possible_classes'].values
        # z_val = val_df['possible_classes'].values
        # z_test = test_df['possible_classes'].values
    
    ###########################################################################
    X_train = train_df['slop_raw'].values
    X_val = val_df['slop_raw'].values
    X_test = test_df['slop_raw'].values

    train_ids = train_df['slope_id'].values.astype('int')
    val_ids = val_df['slope_id'].values.astype('int')
    test_ids = test_df['slope_id'].values.astype('int')

    return (train_ids,X_train,y_train), (val_ids,X_val,y_val), (test_ids,X_test,y_test), slope_df


def get_id_lists(dir):
    file_name_list = ['train_correct_ids', 'val_correct_ids', 'test_correct_ids', 'train_incorrect_ids', 'val_incorrect_ids', 'test_incorrect_ids']

    train_correct_ids = []
    val_correct_ids = []
    test_correct_ids = []

    train_incorrect_ids = []
    val_incorrect_ids = []
    test_incorrect_ids = []

    for i, id_list in enumerate((train_correct_ids, val_correct_ids, test_correct_ids, train_incorrect_ids, val_incorrect_ids, test_incorrect_ids)):
        with open(dir+file_name_list[i]+'.txt', 'r') as f:
            for line in f:
                id_list.append(line.strip())
    
    return train_correct_ids, val_correct_ids, test_correct_ids, train_incorrect_ids, val_incorrect_ids, test_incorrect_ids


def remix(val_ids, X_val, y_val, test_ids, X_test, y_test, seed = 2020):
    # remix val and test for the performance comparison
    np.random.seed(seed)
    idx = np.random.permutation(np.arange(len(y_val) + len(y_test)))

    X = np.concatenate([X_val, X_test])
    y = np.concatenate([y_val, y_test])
    ids = np.concatenate([val_ids, test_ids])

    X_val = X[idx[ : len(idx)//2]]
    y_val = y[idx[ : len(idx)//2]]
    val_ids = ids[idx[ : len(idx)//2]]

    X_test = X[idx[(len(idx)//2) : ]]
    y_test = y[idx[(len(idx)//2) : ]]
    test_ids = ids[idx[(len(idx)//2) : ]]

    return val_ids, X_val, y_val, test_ids, X_test, y_test


def read_test_data(incorrect_type, dir_path='data/', existing_company_only=False, sample=None):
    assert incorrect_type in ['inverse', 'boundary', 'correct']
    
    slope_df = pd.read_csv(dir_path+'final_test_set_ticker.csv')
    slope_df['y_boundary'] = [json.loads(bound) for bound in slope_df['y_boundary']]
    slope_df = slope_df.set_index('id')

    if existing_company_only:
        ticker_path = dir_path+'tickers_list.txt'
        tickers = []
        with open(ticker_path, 'r') as f:
            for line in f:
                tickers.append(line.strip())
        slope_df = slope_df[slope_df['ticker'].isin(tickers)].copy()
        print('Reading Existing Company Only', slope_df.shape)

    if sample:
        print(f'Taking {sample} samples')
        slope_df = slope_df.iloc[:sample]

    ###########################################################################
    if incorrect_type == 'inverse':
        temp_slope_df = slope_df[~((slope_df['sentiment_correct']<1)&(slope_df['y_inverse']==2))]  # remove incorrect class 2
        X = temp_slope_df['slop_raw'].values
        y = temp_slope_df['y_inverse'].values.astype('int')
        ids = temp_slope_df['slope_id'].values.astype('int')
    elif incorrect_type == 'correct':
        temp_slope_df = slope_df[slope_df['sentiment_correct']== 1]
        X = temp_slope_df['slop_raw'].values
        y = temp_slope_df['label_from_score'].values.astype('int')
        ids = temp_slope_df['slope_id'].values.astype('int')
    else:
        X = slope_df['slop_raw'].values
        y = slope_df['label_from_score'].values.astype('int')
        ids = slope_df['slope_id'].values.astype('int')

    return ids, X, y, slope_df