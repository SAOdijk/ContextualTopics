from gensim.models import Word2Vec, Doc2Vec, TfidfModel, CoherenceModel, LdaModel
from gensim.models.doc2vec import TaggedDocument
from gensim.corpora import Dictionary
import timeit
import multiprocessing
import pickle


def tagged_document(list_of_list_of_words):
    for i, list_of_words in enumerate(list_of_list_of_words):
        yield TaggedDocument(list_of_words, [i])


def train_w2v(data, name):
    cores = multiprocessing.cpu_count()
    # Build the model
    nlp_input = [item for sublist in data for item in sublist]
    model = Word2Vec(min_count=1, size=300, negative=10, workers=cores - 1, sg=1)
    start_t = timeit.default_timer()
    model.build_vocab(nlp_input, progress_per=1000)
    stop_t = timeit.default_timer()
    print('Time to build vocab of Word2vec in sec: ', stop_t - start_t)
    # Train model model
    start_t = timeit.default_timer()
    model.train(nlp_input, total_examples=model.corpus_count, epochs=100, report_delay=1)
    stop_t = timeit.default_timer()
    print('Time to train the Word2vec model in sec: ', stop_t - start_t)
    model.init_sims(replace=True)
    model.save(name)


def train_lda(data):
    nlp_input = [w.split(' ') for w in data]
    d = Dictionary(nlp_input)
    d.filter_extremes(no_below=10, no_above=0.5, keep_n=100000)
    bow_corpus = [d.doc2bow(doc) for doc in nlp_input]
    tf_idf = TfidfModel(bow_corpus)
    corpus_tfidf = tf_idf[bow_corpus]
    model = LdaModel(corpus=corpus_tfidf, num_topics=50, id2word=d)
    model.save('models/lda.model')


if __name__ == '__main__':
    with open('data/data_hf.pkl', 'rb') as pickle_file:
        d0 = pickle.load(pickle_file)
    with open('data/data_pf.pkl', 'rb') as pickle_file:
        d1 = pickle.load(pickle_file)
    train_w2v(d0, 'models/w2v_hf.model')
    train_w2v(d1, 'models/w2v_pf.model')
