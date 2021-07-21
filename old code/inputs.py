from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import KeyedVectors
import numpy as np
import joblib
from sklearn.model_selection import LeaveOneOut
# import itertools
import spacy
import pickle
import concurrent.futures
import tqdm
import multiprocessing
from nltk.tokenize import word_tokenize


def split_data():
    with open('pkl/Data0_new.pkl', 'rb') as pickle_file:
        data = pickle.load(pickle_file)
    cv = LeaveOneOut()
    train_sets = []
    test_sets = []
    data = np.array(data)
    for train_ix, test_ix in cv.split(data):
        x_train, x_test = data[train_ix], data[test_ix]
        train_sets.append(x_train)
        test_sets.append(x_test)
    sets = train_sets, test_sets
    print('Data-split complete!')
    filename = 'pkl/splitted_data.pkl'
    joblib.dump(sets, filename)


# def tfidf_features(tfidf, n_feat):
#     indices = np.argsort(tfidf.idf_)[::-1]
#     features = tfidf.get_feature_names()
#     top_n = [features[i] for i in indices[:n_feat]]
#     return top_n
#
#
# def to_tfidf(data, n_feat):
#     tfidf = TfidfVectorizer(max_df=0.99, max_features=20000, min_df=0.05, use_idf=True)
#     x = tfidf.fit_transform(data)
#     top_feat = tfidf_features(tfidf, n_feat)
#     # print(top_feat)
#     return top_feat


def create_vocab(train_data, test_data):
    train_text = ' '.join(train_data)
    train_tokens = word_tokenize(train_text)
    train_tokens = [w for w in train_tokens if w in model.wv.vocab]
    train_vocab = list(dict.fromkeys(train_tokens))
    test_tokens = word_tokenize(str(test_data[0]))
    test_tokens = [w for w in test_tokens if w in model.wv.vocab]
    test_vocab = list(dict.fromkeys(test_tokens))
    if len(test_vocab) >= 20:
        vocabulary = [train_vocab, test_vocab]
        return vocabulary
    else:
        return None


def vectorize(vocab):
    vectors = [model.wv[w] for w in vocab]
    return vectors


# def d2v(train_data, test_data, l_model):
#     model = KeyedVectors.load('models/' + l_model + '.model')
#     train_vocab = ' '.join(train_data)
#     train_vocab = [i.split('.') for i in train_vocab]
#     train_vectors = [model[i] for i in train_vocab]
#     test_vocab = [i.split('.') for i in test_data]
#     test_vectors = []
#     for i in test_vocab:
#         vectors = []
#         for j in i:
#             v = model[j]
#             vectors.append(v)
#         test_vectors.append(vectors)
#     c_results = [train_vocab, train_vectors, test_vocab, test_vectors]
#     with open('pkl/d2v_results.pkl', "wb") as f:
#         pickle.dump(c_results, f)


def main():
    # train_sets, test_sets = joblib.load('pkl/splitted_data.pkl')
    # with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    #     vocabs = list(tqdm.tqdm(executor.map(create_vocab, train_sets, test_sets), total=len(train_sets)))
    # vocabs = [x for x in vocabs if x]
    # train_vocabs, test_vocabs = [], []
    # for i in vocabs:
    #     train_vocabs.append(i[0])
    #     test_vocabs.append(i[1])
    train_vocabs, test_vocabs = joblib.load('pkl/vocabs.pkl')
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        train_vectors = list(tqdm.tqdm(executor.map(vectorize, train_vocabs), total=len(train_vocabs)))
        test_vectors = list(tqdm.tqdm(executor.map(vectorize, test_vocabs), total=len(test_vocabs)))
    vector_data = [train_vectors, test_vectors]
    # vocabs = [train_vocabs, test_vocabs]
    # joblib.dump(vocabs, 'pkl/vocabs.pkl')
    joblib.dump(vector_data, 'pkl/vectors.pkl')


if __name__ == '__main__':
    model = KeyedVectors.load('models/w2v2.model')
    main()
