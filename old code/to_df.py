import pandas as pd
from gap_statistic import OptimalK
# import numpy as np
import pickle
import concurrent.futures
import tqdm
import joblib

# Options
pd.options.display.float_format = '{:,.0f}'.format
optimalK = OptimalK(parallel_backend='rust')


# To dataframe
def to_df(itt, vectors, vocab, labels):
    train_cl_dict = {'cluster': labels, 'vector': vectors, 'word': vocab}
    df = pd.DataFrame(train_cl_dict)
    df = zip_df(df)
    df.insert(0, 'cv_iteration', itt)
    return df


# # TEST SET
# def df_test(itt, vectors, vocab, labels, cl_method, c_data):
#     # Create rows and add to dataframe
#     test_cl_dict = {'cluster': labels, 'vector': vectors, 'word': vocab}
#     df = pd.DataFrame(test_cl_dict)
#     f_df = pd.DataFrame()
#     for i in range(0, int(df['file'].max() + 1)):
#         mini_df = df.loc[df['file'] == i]
#         x = zip_df(mini_df)
#         x.insert(0, 'file', i)
#         x.insert(1, 'dbcv_score', dbcv_scores[i])
#         f_df = f_df.append(x, ignore_index=True)
#     f_df.to_pickle('pkl/' + cl_method + '_' + c_data + '_t.pkl')


def zip_df(df):
    cl_df = pd.DataFrame()
    for i in range(0, int(df['cluster'].max()+1)):
        mini_df = df.loc[df['cluster'] == i]
        vectors = mini_df['vector'].tolist()
        words = mini_df['word'].tolist()
        cl_dict = {'cluster': i, 'vectors': vectors, 'words': words}
        cl_df = cl_df.append(cl_dict, ignore_index=True)
    return cl_df


def main():
    # optics_trl, optics_ttl = joblib.load('pkl/optics_labels_5.pkl')
    # train_vocab, test_vocab = joblib.load('pkl/vocabs.pkl')
    train_vectors, test_vectors = joblib.load('pkl/vectors.pkl')
    index = list(range(len(train_vectors)))
    print(train_vectors)
    print('Inputs loaded!')
    # train_vocab, train_vectors, test_vocab, test_vectors = [], [], [], []
    # for i in range(0, len(inputs)):
    #     tr_voc, tr_vec, tt_voc, tt_vec = inputs[i]
    #     train_vocab.append(tr_voc), train_vectors.append(tr_vec)
    #     test_vocab.append(tt_voc), test_vectors.append(tt_vec)
    # optics_trl, optics_ttl = optics_results
    # hdbscan_trl, hdbscan_ttl = hdbscan_results
    # with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
    #     optics_tr_df = list(
    #         tqdm.tqdm(executor.map(to_df, index, train_vectors, train_vocab, optics_trl), total=len(index)))
    #     optics_tt_df = list(
    #         tqdm.tqdm(executor.map(to_df, index, test_vectors, test_vocab, optics_ttl), total=len(index)))
    #     optics_dfs = [optics_tr_df, optics_tt_df]
    # joblib.dump(optics_dfs, 'pkl/optics_dfs.pkl')


if __name__ == '__main__':
    main()
