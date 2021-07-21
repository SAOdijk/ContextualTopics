import pandas as pd
# from scipy.spatial import minkowski_distance
# import itertools
import pickle
import concurrent.futures
import tqdm
from gensim.models import KeyedVectors, Word2Vec
import joblib


# Centers
# def cl_center_distances(df_train, df_test):
#     # Distance cluster_centers
#     euc_dist = []
#     tr_clu = []
#     for i in df_test['cl_center']:
#         ccl = 1000
#         index = 0
#         for j in df_train['cl_center']:
#             dist = minkowski_distance(i, j, 2)
#             if dist < ccl:
#                 ccl = dist
#                 tr_cl = df_train['cluster'][index]
#             index += 1
#         tr_clu.append(tr_cl)
#         euc_dist.append(ccl)
#     return euc_dist, tr_clu


def purity_score(y_true, y_pred, df):
    f = 0
    for i in y_true:
        for j in range(0, len(df.words)):
            pred_words = df.words[j]
            if i in pred_words and j != y_pred:
                f += 1

    cl = df.words[y_pred]
    match = [x for x in y_true if x in cl]
    n = len(match)
    purity = 1 / len(y_true) * (n - f)
    return purity


# Precision
def classification(true, pre):
    tp, fp = 0, 0
    for i in true:
        if i in pre:
            tp += 1
        else:
            fp += 1
    precision = tp / (tp + fp)
    return precision


def stats_to_df(index, df_train, df_test):
    if df_test.empty:
        return None
    else:
        # Precision score inputs
        model = KeyedVectors.load('models/w2v.model')
        f_cl, wmd_distance, pre_score, pur_score = [], [], [], []
        test_words, fitted_words = [], []
        for i in df_test['words']:
            distance = 10
            wmd_cluster = 0
            cl_index = 0
            for j in df_train['words']:
                wdm_dist = model.wmdistance(i, j)
                if wdm_dist < distance:
                    distance = wdm_dist
                    wmd_cluster = cl_index
                cl_index += 1
            # Precision & purity

            precision = classification(i, df_train.loc[df_train['cluster'] == wmd_cluster, 'words'].iloc[0])
            purity = purity_score(i, wmd_cluster, df_train)
            wmd_distance.append(distance), f_cl.append(wmd_cluster), pur_score.append(purity),
            pre_score.append(precision),
            test_words.append(i), fitted_words.append(df_train.loc[df_train['cluster'] == wmd_cluster, 'words'].iloc[0])

        cluster = df_test.cluster
        data_dict = {'index': index, 'cluster': cluster, 'fitted_cl': f_cl, 'wmd': wmd_distance,
                     'precision': pre_score, 'purity': pur_score, 'words': test_words, 'fitted_words': fitted_words}
        df = pd.DataFrame(data_dict)
        return df


def main():
    df_optics_train_sets, df_optics_test_sets = joblib.load('pkl/optics_dfs.pkl')
    # with open('pkl/df_hdbscan.pkl', 'rb') as pickle_file:
    #     df_sets_hdbscan = pickle.load(pickle_file)
    # df_hdbscan_train_sets, df_hdbscan_test_sets = df_sets_hdbscan
    index = list(range(len(df_optics_train_sets)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:
        optics_stats = list(tqdm.tqdm(executor.map(stats_to_df, index, df_optics_train_sets, df_optics_test_sets),
                                      total=len(df_optics_train_sets)))
        # hdbscan_stats = list(tqdm.tqdm(executor.map(stats_to_df, index, df_hdbscan_train_sets, df_hdbscan_test_sets),
        #                                total=len(df_hdbscan_train_sets)))
    optics_stats = [x for x in optics_stats if x is not None]
    # hdbscan_stats = [x for x in hdbscan_stats if x is not None]
    optics_stats = pd.concat(optics_stats)
    joblib.dump(optics_stats, 'pkl/optics_stats3')
    # hdbscan_stats = pd.concat(hdbscan_stats)
    # # with open('pkl/hdbscan_stats.pkl', 'wb') as pickle_file:
    # #     pickle.dump(hdbscan_stats, pickle_file)
    optics_stats.to_excel('xlsx/optics_stats3.xlsx')
    # hdbscan_stats.to_excel('xlsx/hdbscan_stats.xlsx')


if __name__ == '__main__':
    main()
