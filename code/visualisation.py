import concurrent.futures
import tqdm
from wordcloud import WordCloud
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import random
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from DBCV import DBCV
from pyclustertend import hopkins
import numpy
from scipy.spatial.distance import cosine


def wordcloud_gen(text, topic):
    text = ' '.join(text)
    wordcloud = WordCloud().generate(text)

    # Save the generated image:
    wordcloud.to_file("wordcloud/" + str(topic) + ".png")


def standard_plotting(embedding, title, labels=None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    if labels is None:
        ax.scatter(embedding[:, 0], embedding[:, 1], embedding[:, 2], c='blue', s=5)
    else:
        clustered = (labels >= 0)
        ax.scatter(embedding[~clustered, 0], embedding[~clustered, 1], embedding[~clustered, 2], c='grey', s=5)
        ax.scatter(embedding[clustered, 0], embedding[clustered, 1], embedding[clustered, 2], c=labels[clustered], s=5,
                   cmap='Spectral')
    plt.title(title)
    plt.show()


def layered_plotting(data1, data2, labels, title):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(data2[:, 0], data2[:, 1], data2[:, 2], c='grey', s=5)
    ax.scatter(data1[:, 0], data1[:, 1], data1[:, 2], c=labels, s=5, cmap='Spectral')
    plt.title(title)
    plt.show()


def table(df, labels):
    tb = pd.DataFrame()
    for i in range(0, int(df[labels].max()+1)):
        mini_tb = df.loc[df[labels] == i]
        words = mini_tb['word'].tolist()
        n_words = len(words)
        tb_dict = {'topic': i, 'words': words, 'length': n_words}
        tb = tb.append(tb_dict, ignore_index=True)
    return tb


def tp_presence(doc, tbl):
    topics_count = []
    for i in range(len(tbl)):
        c = len([w for w in doc if w in tbl.iloc[i]['words']])
        topics_count.append(c)
    noise_points = len(doc) - sum(topics_count)
    tp_points = len(doc) - noise_points
    # topics_count.append(noise)
    topics_percentages = [n / tp_points * 100 for n in topics_count]
    topic_dict = {}
    for i in range(len(topics_percentages)):
        if topics_count[i] >= 0.04 * tp_points:
            topic_dict['Topic ' + str(i)] = topics_percentages[i]
    return topic_dict


def tp_count(tbl, name):
    sent_docs = joblib.load('data/data_hf.pkl')
    docs = []
    for i in sent_docs:
        doc = [item for sublist in i for item in sublist]
        docs.append(doc)
    topics_count = []
    for doc in docs:
        inter_count = []
        for i in range(len(tbl)):
            c = len([w for w in set(doc) if w in tbl.iloc[i]['words']])
            if c >= 5:
                inter_count.append(1)
            else:
                inter_count.append(0)
        topics_count.append(inter_count)
    tp_counts = [sum(i) for i in zip(*topics_count)]
    tbl['topic_count'] = tp_counts
    tbl.to_excel('tables/' + name + '_hf2.xlsx')
    return tp_counts


def method_choice(method, labels):
    df = joblib.load('dataframes/results10hf.pkl')
    ve = numpy.array(list(df.visual_embedding))
    nn_optics = df[df['optics_label'] >= 0]
    nn_hdbscan = df[df['hdbscan_label'] >= 0]
    if method == 'optics':
        sil = silhouette_score(numpy.array(list(nn_optics.embedding)),
                               numpy.array(list(nn_optics.optics_label)), metric='euclidean')
        dbs = davies_bouldin_score(numpy.array(list(nn_optics.embedding)),
                                   numpy.array(list(nn_optics.optics_label)))
        chs = calinski_harabasz_score(numpy.array(list(nn_optics.embedding)),
                                      numpy.array(list(nn_optics.optics_label)))
        dbcv = DBCV(numpy.array(list(nn_optics.embedding)),
                    numpy.array(list(nn_optics.optics_label)), dist_function=cosine)
    elif method == 'hdbscan':
        sil = silhouette_score(numpy.array(list(nn_hdbscan.embedding)),
                               numpy.array(list(nn_hdbscan.hdbscan_label)), metric='euclidean')
        dbs = davies_bouldin_score(numpy.array(list(nn_hdbscan.embedding)),
                                   numpy.array(list(nn_hdbscan.hdbscan_label)))
        chs = calinski_harabasz_score(numpy.array(list(nn_hdbscan.embedding)),
                                      numpy.array(list(nn_hdbscan.hdbscan_label)))
        dbcv = DBCV(numpy.array(list(nn_hdbscan.embedding)),
                    numpy.array(list(nn_hdbscan.hdbscan_label)), dist_function=cosine)
    elif method == 'kmeans':
        sil = silhouette_score(ve, numpy.array(list(labels)), metric='euclidean')
        dbs = davies_bouldin_score(ve, numpy.array(list(labels)))
        chs = calinski_harabasz_score(ve, numpy.array(list(labels)))
        dbcv = DBCV(ve, numpy.array(list(labels)), dist_function=cosine)
    else:
        sil = silhouette_score(ve, numpy.array(list(labels)), metric='euclidean')
        dbs = davies_bouldin_score(ve, numpy.array(list(labels)))
        chs = calinski_harabasz_score(ve, numpy.array(list(labels)))
        dbcv = DBCV(ve, numpy.array(list(labels)), dist_function=cosine)

    scores = [method, sil, dbs, chs, dbcv]
    return scores


def pie_chart(data):
    x = list(data.values())
    my_labels = list(data.keys())
    plt.pie(x, labels=my_labels, startangle=90)
    plt.show()


def main():
    # Data, Docs & Dataframe
    df = joblib.load('dataframes/results10hf.pkl')
    sent_docs = joblib.load('data/data_hf.pkl')
    docs = []
    for i in sent_docs:
        doc = [item for sublist in i for item in sublist]
        docs.append(doc)
    method = ['kmeans', 'optics', 'hdbscan', 'agglomerative']
    labels = [df.kmeans_label, df.optics_label, df.hdbscan_label, df.ag_label]
    # subset = df[['visual_embedding', 'optics_label', 'hdbscan_label']]
    nn_optics = df[df['optics_label'] >= 0]
    nn_hdbscan = df[df['hdbscan_label'] >= 0]
    ve = numpy.array(list(df.visual_embedding))
    ve2 = numpy.array(list(df.visual_embedding2))
    emb = numpy.array(list(df.embedding))

    # Create Table
    k_table = table(df, 'kmeans_label')
    o_table = table(df, 'optics_label')
    h_table = table(df, 'hdbscan_label')
    a_table = table(df, 'ag_label')
    tables = [k_table, o_table, h_table, a_table]
    tables_names = ['k_table', 'o_table', 'h_table', 'a_table']

    # Cluster tendency
    print('Hopkins score:', hopkins(numpy.array(list(df.embedding)), 3000))

    # Determine Method
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        scores = list(tqdm.tqdm(executor.map(method_choice, method, labels), total=len(method)))
    print(scores)

    # Topic count
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        tqdm.tqdm(executor.map(tp_count, tables, tables_names), total=len(tables))

    # # Scatter plots
    standard_plotting(ve, 'Umap Overview')
    standard_plotting(ve, 'K_means Clustering', numpy.array(list(df.kmeans_label)))
    standard_plotting(ve, 'Project Proposals - OPTICS Clustering with Noise', numpy.array(list(df.optics_label)))
    standard_plotting(numpy.array(list(nn_optics.visual_embedding)), 'Project Proposals - OPTICS Clustering Without Noise',
                      numpy.array(list(nn_optics.optics_label)))
    standard_plotting(ve, 'HDBSCAN Clustering with Noise', numpy.array(list(df.hdbscan_label)))
    standard_plotting(numpy.array(list(nn_hdbscan.visual_embedding)), 'HDBSCAN Clustering without Noise',
                      numpy.array(list(nn_hdbscan.hdbscan_label)))
    standard_plotting(ve, 'Agglomerative Clustering', numpy.array(list(df.ag_label)))

    # PF pie Chart
    random.seed(7)
    sample_pf = random.sample(docs, 1)
    sample_pf = tp_presence(sample_pf[0], o_table)
    # HF pie chart
    pie_chart(sample_pf)
    sample_hf = docs[13]
    sample_hf = tp_presence(sample_hf, o_table)
    pie_chart(sample_hf)


if __name__ == '__main__':
    main()
