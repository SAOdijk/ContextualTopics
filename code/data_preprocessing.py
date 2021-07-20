from tika import parser
import glob
import string
from nltk.tokenize import word_tokenize, sent_tokenize
import concurrent.futures
import tqdm
import pickle
from sklearn.feature_extraction.text import CountVectorizer


def create_data(file):
    raw = parser.from_file(file)
    text = raw['content']
    if type(text) == str:
        doc = preprocess_file(text)
    else:
        doc = ''
    return doc


def preprocess_file(doc):
    doc = sent_tokenize(doc)
    doc = [word_tokenize(s) for s in doc]
    doc = [[w.lower() for w in s] for s in doc]
    table = str.maketrans('', '', string.punctuation)
    doc = [[w.translate(table) for w in s] for s in doc]
    doc = [[w for w in s if w.isalpha()] for s in doc]
    doc = [[w.strip() for w in s] for s in doc]
    doc = [[w for w in s if len(w) >= 2] for s in doc]
    doc = [s for s in doc if s]
    return doc


def uncom(docs, min_df, max_df):
    cv = CountVectorizer(min_df=min_df, max_df=max_df)
    flat_docs = []
    for i in docs:
        merge_inner = [item for sublist in i for item in sublist]
        string_inner = ' '.join(merge_inner)
        flat_docs.append(string_inner)
    x = cv.fit_transform(flat_docs)
    valid = cv.get_feature_names()
    docs = [[[w for w in s if w in valid] for s in d] for d in docs]
    return docs


def set_stats(data):
    avg_sent = 0
    avg_words = 0
    avg_vocab = 0
    flat_docs = []
    for i in data:
        avg_sent += len(i)
        merge_inner = [item for sublist in i for item in sublist]
        avg_words += len(merge_inner)
        avg_vocab += len(set(merge_inner))
        flat_docs.append(merge_inner)
    print('Total no. of sentences:', avg_sent)
    print('Avg. no. of sentences:', avg_sent / len(data))
    print('Total no. of words:', avg_words)
    print('Avg. no. of words:', avg_words / len(data))
    merge_outer = [item for sublist in flat_docs for item in sublist]
    print('Vocab size:', avg_vocab / len(data))
    return merge_outer


def main():
    p1 = 'C:/Users/Steven/Dropbox/2.Documents/UU/IBM/Thesis/Data1/*.*'
    p2 = 'C:/Users/Steven/Dropbox/2.Documents/UU/IBM/Thesis/Data2/*.*'
    f1 = glob.glob(p1)
    f2 = glob.glob(p2)
    print('Total number of highlight reports: ' + str(len(f1)))
    print('Total number of project files: ' + str(len(f2)))
    print('Total number of files: ' + str(len(f2 + f1)))
    with concurrent.futures.ProcessPoolExecutor(max_workers=14) as executor:
        hf = list(tqdm.tqdm(executor.map(create_data, f1), total=len(f1)))
        pf = list(tqdm.tqdm(executor.map(create_data, f2), total=len(f2)))
    # print('Stats Highlights')
    # flat1 = set_stats(hf)
    # print('Stats Proposals')
    # flat2 = set_stats(pf)
    # flats = flat1 + flat2
    # t_vocab = len(set(flats))
    # print(t_vocab)
    hf = uncom(hf, 2, 0.8)
    print(hf[13])
    pf = uncom(pf, 5, 0.9)
    with open('data/data_hf.pkl', 'wb') as pickle_file:
        pickle.dump(hf, pickle_file)
    with open('data/data_pf.pkl', 'wb') as pickle_file:
        pickle.dump(pf, pickle_file)


if __name__ == '__main__':
    main()
