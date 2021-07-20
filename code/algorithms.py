import pandas as pd
from gap_statistic import OptimalK
from sklearn.cluster import KMeans, AgglomerativeClustering, OPTICS
from hdbscan import HDBSCAN
import numpy as np
import joblib
from gensim.models import Word2Vec
import umap


def kmeans(data, n_clusters):
    model = KMeans(n_clusters=n_clusters, init='k-means++')
    result = model.fit(data)
    labels = result.labels_
    return labels


def agglomerative(data, n_clusters):
    model = AgglomerativeClustering(n_clusters=n_clusters, affinity='cosine', linkage='single')
    result = model.fit(data)
    labels = result.labels_
    return labels


def optics(data, n_clusters):
    model = OPTICS(min_samples=n_clusters)
    result = model.fit(data)
    labels = result.labels_
    return labels


def hdbscan(data, n_clusters):
    model = HDBSCAN(min_cluster_size=n_clusters)
    model.fit(data)
    labels = model.labels_
    return labels


def dim_reduc(n_neighbors, min_dist, metric, n_components):
    embedding = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, n_components=n_components)
    return embedding


def to_df(words, vectors, visual_embedding, visual_embedding2, embedding, lbl1, lbl2, lbl3, lbl4):
    d = {'word': words, 'vector': vectors, 'visual_embedding': visual_embedding, 'visual_embedding2': visual_embedding2,
         'embedding': embedding, 'kmeans_label': lbl1, 'optics_label': lbl2, 'hdbscan_label': lbl3, 'ag_label': lbl4}
    df = pd.DataFrame(d)
    return df


def main(cluster_size, mod):
    model = mod
    word_dic = model.wv.vocab
    w = list(word_dic.keys())
    wv = model.wv.vectors
    wv = [v for v in wv]
    optimal_k = OptimalK(n_jobs=8, parallel_backend='multiprocessing')

    # Umap
    visual_embedding = dim_reduc(15, 0.1, 'cosine', 3).fit_transform(wv)
    visual_embedding = [v for v in visual_embedding]
    visual_embedding2 = dim_reduc(15, 0.1, 'cosine', 2).fit_transform(wv)
    visual_embedding2 = [v for v in visual_embedding2]
    input_embedding = dim_reduc(30, 0, 'cosine', 30).fit_transform(wv)
    input_embedding = [v for v in input_embedding]

    # Clustering
    n_clusters = optimal_k(np.array(input_embedding), cluster_array=np.arange(1, 60))
    print('The optimal number of clusters is:', n_clusters)
    km_labels = kmeans(input_embedding, n_clusters)
    ag_labels = agglomerative(input_embedding, n_clusters)
    op_labels = optics(input_embedding, cluster_size)
    print('The number of Optics clusters is:', max(op_labels))
    hd_labels = hdbscan(input_embedding, cluster_size)
    print('The number of Hdbscan clusters is:', max(hd_labels))

    # Dataframe
    df = to_df(w, wv, visual_embedding, visual_embedding2, input_embedding, km_labels, op_labels, hd_labels, ag_labels)
    return df


if __name__ == '__main__':
    mod1 = Word2Vec.load('models/w2v_hf.model')
    mod2 = Word2Vec.load('models/w2v_pf.model')
    hf = main(10, mod1)
    joblib.dump(hf, 'dataframes/results10hf2.pkl')
    pf = main(10, mod2)
    joblib.dump(pf, 'dataframes/results10pf2.pkl')
    # li = [5, 10, 15, 20]
    # mods = [mod1, mod2]
    # for m in range(len(mods)):
    #     print('Mod:', m)
    #     for item in li:
    #         print('Cluster size:', item)
    #         main(item, mods[m])
