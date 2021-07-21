import pickle
# import fasttext
# import spacy
# import numpy as np
# import itertools
import pandas as pd
# from wiki_dump_reader import Cleaner, iterate
# import io
# from spellchecker import SpellChecker
# import spacy
# import glob
# import re
# from collections import Counter
from gensim.models import KeyedVectors, Word2Vec
from gensim.models.coherencemodel import CoherenceModel
import joblib
from nltk.tokenize import word_tokenize, sent_tokenize
import string
import random
from tika import parser
from gensim.corpora.dictionary import Dictionary
#
#
# def table(df, labels):
#     tb = pd.DataFrame()
#     for i in range(0, int(df[labels].max()+1)):
#         mini_tb = df.loc[df[labels] == i]
#         words = mini_tb['word'].tolist()
#         n_words = len(words)
#         tb_dict = {'topic': i, 'words': words, 'length': n_words}
#         tb = tb.append(tb_dict, ignore_index=True)
#     return tb
#
#
# dataframe = joblib.load('dataframes/results10hf.pkl')
# data = joblib.load('data/data_hf.pkl')
# # doc = data[13]
# # doc = [item for sublist in doc for item in sublist]
# o_table = table(dataframe, 'optics_label')
# # topics = [75, 43, 41, 29, 26, 13]
# # for i in topics:
# #     common = [w for w in doc if w in o_table.iloc[i]['words']]
# #     print(i)
# #     print(common)
# words = o_table['words']
#
# # nn_optics = df[df['optics_label'] >= 0]
# # # model = Word2Vec.load("models/w2v_pf.model")
# model = Word2Vec.load("models/w2v_hf.model")
# data = [item for sublist in data for item in sublist]
# dictionary = Dictionary(data)
# cm = CoherenceModel(model=model, topics=words, texts=data, dictionary=dictionary, coherence='c_v')
# coherence = cm.get_coherence()
# print(coherence)

# print(len(model.wv.vocab))
# print(len(nn_optics))

# df = joblib.load('dataframes/results.pkl')
#
# with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
#     print(df.head())

# topics_count = [[1,6,6],[1,6,6],[1,6,6],[1,6,6],[1,6,6]]
# print(len(topics_count))
# tp_counts = [sum(i) for i in zip(*topics_count)]
# print(len(topics_count))

# def create_data(file):
#     raw = parser.from_file(file)
#     text = raw['content']
#     # if type(text) == str:
#     if type(text) == str:
#         doc = preprocess_file(text)
#     else:
#         doc = ''
#     return doc
#
#
# def preprocess_file(doc):
#     doc = sent_tokenize(doc)
#     doc = [word_tokenize(s) for s in doc]
#     doc = [[w.lower() for w in s] for s in doc]
#     table = str.maketrans('', '', string.punctuation)
#     doc = [[w.translate(table) for w in s] for s in doc]
#     doc = [[w for w in s if w.isalpha()] for s in doc]
#     doc = [[w.strip() for w in s] for s in doc]
#     doc = [[w for w in s if len(w) >= 2] for s in doc]
#     doc = [s for s in doc if s]

    # Cleaning
    # # Remove all the special characters
    # doc = re.sub(r'\W', ' ', str(doc))

    # remove all single characters
    # doc = re.sub(r'\s+[a-zA-Z]\s+', ' ', doc)
    #
    # # Remove single characters from the start
    # doc = re.sub(r'\^[a-zA-Z]\s+', ' ', doc)

    # Substituting multiple spaces with single space
    # doc = re.sub(r'\s+', ' ', doc, flags=re.I)
    #
    # # Removing prefixed 'b'
    # doc = re.sub(r'^b\s+', '', doc)
    #
    # # Remove random characters
    # doc = doc.replace('\n', ' ').replace('\uf04a', '').replace('\uf04b', '').replace('\uf04c', '') \
    #     .replace('\uf020', '').replace('xxx', '').replace('\t', '').replace('\xa0', '')

    # Transform
    # doc = nlp(doc)
    # norms = []
    # for token in doc:
    #     if token.text != '' and token.text != ' ' and len(token.text) > 1:
    #         norm = token.norm_.strip()
    #         norms.append(norm)
    # norms = nlp(' '.join(norms))
    # lemmas = []
    # for token in norms:
    #     if not token.is_stop:
    #         lemma = token.lemma_
    #         lemmas.append(lemma)
    # doc = ' '.join(lemmas)
    # return doc


# def preprocess_spacy_sent(file):
#     doc = nlp(file)
#     sentences = [sent.text for sent in doc.sents]
#     n_doc = []
#     # Cleaning
#     for i in sentences:
#         # Remove all the special characters
#         i = re.sub(r'\W', ' ', str(i))
#
#         # remove all single characters
#         i = re.sub(r'\s+[a-zA-Z]\s+', ' ', i)
#
#         # Remove single characters from the start
#         i = re.sub(r'\^[a-zA-Z]\s+', ' ', i)
#
#         # Substituting multiple spaces with single space
#         i = re.sub(r'\s+', ' ', i, flags=re.I)
#
#         # Removing prefixed 'b'
#         i = re.sub(r'^b\s+', '', i)
#
#         # Remove random characters
#         i = i.replace('\n', ' ').replace('\uf04a', '').replace('\uf04b', '').replace('\uf04c', '') \
#             .replace('\uf020', '').replace('xxx', '').replace('\t', '').replace('\xa0', '')
#
#         # Remove punctuation
#         i = i.translate(str.maketrans('', '', string.punctuation))
#
#         # Convert to lowercase
#         i = i.lower()
#
#         # Remove unnecessary spaces
#         i = i.strip()
#
#         # Transform
#         i = nlp(i)
#         norms = []
#         for j in i:
#             norm = j.norm_.strip()
#             norms.append(norm)
#         norms = nlp(' '.join(norms))
#         lemmas = []
#         for j in norms:
#             if j.is_alpha and not j.is_stop:
#                 lemma = j.lemma_
#                 lemmas.append(lemma)
#         i = ' '.join(lemmas)
#         n_doc.append(i)
#     return n_doc

#
# def uncom(docs):
#     cv = CountVectorizer(min_df=0.01, max_df=0.90)
#     flat_docs = []
#     for i in docs:
#         merge_inner = [item for sublist in i for item in sublist]
#         string_inner = ' '.join(merge_inner)
#         flat_docs.append(string_inner)
#     x = cv.fit_transform(flat_docs)
#     valid = cv.get_feature_names()
#     docs = [[[w for w in s if w in valid] for s in d] for d in docs]
#     return docs
#
#
# def main():
#     p1 = 'C:/Users/Steven/Dropbox/2.Documents/UU/IBM/Thesis/Data1/*.*'
#     p2 = 'C:/Users/Steven/Dropbox/2.Documents/UU/IBM/Thesis/Data2/*.*'
#     f1 = glob.glob(p1)
#     f2 = glob.glob(p2)
#     print('Total number of highlight reports: ' + str(len(f1)))
#     print('Total number of project files: ' + str(len(f2)))
#     with concurrent.futures.ProcessPoolExecutor(max_workers=14) as executor:
#         hf = list(tqdm.tqdm(executor.map(create_data, f1), total=len(f1)))
#         pf = list(tqdm.tqdm(executor.map(create_data, f2), total=len(f2)))
#     hf = uncom(hf)
#     pf = uncom(pf)
#     results = hf + pf
#     with open('data/data_v5.0.pkl', 'wb') as pickle_file:
#         pickle.dump(results, pickle_file)
#
#
# if __name__ == '__main__':
#     main()





# print(len([2.5948103792415167, 1.1976047904191618, 3.592814371257485, 0.39920159680638717, 2.5948103792415167, 3.792415169660679, 0.39920159680638717, 0.7984031936127743, 2.3952095808383236, 0.7984031936127743, 1.3972055888223553, 2.5948103792415167, 5.189620758483033, 1.5968063872255487, 7.385229540918163, 1.7964071856287425, 1.996007984031936, 3.592814371257485, 1.3972055888223553, 0.998003992015968, 3.3932135728542914, 7.584830339321358, 2.9940119760479043, 3.1936127744510974, 3.992015968063872, 1.3972055888223553, 1.3972055888223553, 1.996007984031936, 3.592814371257485, 0.5988023952095809, 0.5988023952095809, 2.3952095808383236, 2.3952095808383236, 1.1976047904191618, 9.580838323353294, 1.5968063872255487, 0.39920159680638717, 1.3972055888223553, 0.7984031936127743, 0.998003992015968, 1.996007984031936]))

# docs = joblib.load('data/data_v4.0.pkl')
# for i in docs:
#     merge_inner = [item for sublist in i for item in sublist]
#     print(merge_inner)


# text = 'God is Great! I won a lottery.'
# text2 = 'God is Great! I won a lottery.'
#
#
# def process(doc):
#     sent_tokens = sent_tokenize(doc)
#     sent_tokens = [word_tokenize(s) for s in sent_tokens]
#     sent_tokens = [[w.lower() for w in s] for s in sent_tokens]
#     table = str.maketrans('', '', string.punctuation)
#     sent_tokens = [[w.translate(table) for w in s] for s in sent_tokens]
#     sent_tokens = [[w for w in s if w.isalpha()] for s in sent_tokens]
#     sent_tokens = [[w.strip() for w in s] for s in sent_tokens]
#     sent_tokens = [[w for w in s if len(w) > 2] for s in sent_tokens]
#     return sent_tokens
#
#
# c = []
# a = process(text)
# b = process(text2)
# c.append(a)
# c.append(b)
#
#
# d = c + c
# e = [item for sublist in d for item in sublist]
# print(d)
# print(e)

# d = []
# for i in c:
#     flat_list = [item for sublist in i for item in sublist]
#     flat = ' '.join(flat_list)
#     d.append(flat)
#
# print(d)



# w2v_model = KeyedVectors.load('models/w2v.model')
# df_optics_train_sets, df_optics_test_sets = joblib.load('pkl/optics_dfs.pkl')
# df_stats = joblib.load('pkl/optics_stats3')
# pd.set_option("display.max_rows", None, "display.max_columns", None)
#
# new_test = []
# for df in df_optics_test_sets:
#     if len(df.words[0]) < 50:
#         new_test.append(df)


#
# def print_stats(dataframe):
#     min_cl = 1000
#     max_cl = 0
#     avg_cl = 0
#     min_w = 1000
#     max_w = 0
#     avg_w = 0
#     for df in dataframe:
#         cl = len(df)
#         total_w = 0
#         avg_cl += cl
#         if cl < min_cl:
#             min_cl = cl
#         elif cl > max_cl:
#             max_cl = cl
#         for w in df.words:
#             words = len(w)
#             total_w += words
#             if words < min_w:
#                 min_w = words
#             elif words > max_w:
#                 max_w = words
#         total_w = total_w / cl
#         avg_w += total_w
#     avg_cl = avg_cl / len(dataframe)
#     avg_w = avg_w / len(dataframe)
#     print(max_cl, min_cl, avg_cl)
#     print(max_w, min_w, avg_w)
#
#
# print_stats(df_optics_train_sets)
# print_stats(new_test)



# for df in range(0, len(test_vectors)):
#     if df == 73:
#         print(test_vectors[df])





# print(w2v_model.wv['cloudapplicatie'])




# with open('pkl/Data0_new.pkl', 'rb') as pickle_file:
#     data = pickle.load(pickle_file)
#
# print(len(data))



# with open('pkl/df_optics.pkl', 'rb') as pickle_file:
#     df_sets_optics = pickle.load(pickle_file)
#
#
# df_optics_train_sets, df_optics_test_sets = df_sets_optics
#
# df_train, df_test = df_optics_train_sets[0], df_optics_test_sets[0]
#
# for index, row in df_train.iterrows():
#     print(' '.join(row['words']))



# with open('pkl/w2v_inputs.pkl', 'rb') as pickle_file:
#     inputs = pickle.load(pickle_file)
#
# index = list(range(len(inputs)))
# print(index)




# test = ['miniconnector project implementatie cursusonderwijs doorontwikkeltraject ng laat kennen gep land activiteit aankomen kennen green plannen yellow schema klein impact red schema groot impact afronden verwijderen koppeling inrichting acceptatie omgeving integratieplat form afronden ophalen osiri iam koppeling cursusonderwijs integratieplatform bouwen werking cursusonderwijs testen cursus ophalen docent word ophalen iam ap mits aanwe zig iam applicatiebeheer budgetpost cursusonderwijs doorzetten aanpassing acceptatie omgeving testen acceptatie project fb fsw doornemen sprint bevinding leverancier akkoord fsw project bouwen aanpassing studentenverzoek gebruiker komen aanpassen sing plan release productieomgeving overdracht behe restpunt bespreken overdracht do cumentenset akko ord sec faculteitscontroller hiervoor volgen test koppeling lann kostenregistratie cursusonderwijs acc iam ap beschikbaar zien krijgen faculteit specifiek product ownership applicatiebeheer budgetpost cursusonderwijs uitvoeren security test uu cursusonderwijs doorzetten release productie geving lanning weekend mei overdracht beheer dienst via cab offic ieel beheer emen openstaan punt doornemen mijlpaal datum status opleveren testomgeving vaststellen projectplan opleveren acceptatieomgeving opleveren communicatieplan opleveren productieomgeving acceptatie inrichting afnemen evaluatie productieomgeving terugkoppeling docent coördinator rapportage pilotfase blok oplevering draaiboek implementatie planning aansluiting faculteit cursusonderwijs rapportage pilotfase blok teamoverleg conform planning aansluiting faculteit ev projectscope plan aanpak fsw oplevering product budget investeringskost businesskost mrt mei juni juni juli juli ec reviewcommentaar importbestand uitgave verpl aandachtspunt oorzaak afwijking actie datum naam koppeling productiesysteem werken iam iam patch plaatsen uitleveren leverancier ophalen gegeven iam integratieplatform datasystem krijgen productieomgeving wachten patch iam leverancier patch beschikbaar tq testen testen acc beooren eld acc api zetten pr od andré andré']
# nlp = spacy.load('nl_core_news_lg')
#
#
# test_tokenize = [token.text for token in nlp(test)]
# test_vocab = list(dict.fromkeys(test_tokenize))
# print(test_vocab)



# with open('pkl/Data0.pkl', 'rb') as pickle_file:
#     d1 = pickle.load(pickle_file)
#
# print(d1)



# path = 'C:/Users/Steven/Dropbox/2.Documents/UU/IBM/Thesis/Data0/*.*'
# file_list = []
#
# for file in glob.glob(path):
#     file_list.append(file)
#
# print(file_list)
# with open('pkl/optics_cl_w2v_tr.pkl', 'rb') as pickle_file:
#     d1 = pickle.load(pickle_file)
#
# print(d1.head())

# spell = SpellChecker()
# spell.word_frequency.load_dictionary('custom_dict.gz')
#
# text = 'Wim moet een lijst ophangen Zijn vrouw heeft een grote foto van hun kleinkind' \
#        ' Met de boor willl hij een gat boren Dat lukt niet de muur is te hard Hij pakt de hamer ' \
#        'Nu moet hij hard slaana De spijkerr springt weg De hamer valt uit zijn hand De ' \
#        'hamer valt op zijn teen Wim brulty van pijnnn Hij geeft zijn vrouw de schuld  '
#
# text = text.lower()
# text = text.split(' ')
# text = [t for t in text if t != '' and t != ' ' and len(t) > 1]
# print(len(text))
# text = [spell.correction(w) for w in text]
# print(text)


# for w, i in enumerate(spell_check):
#     word = spell.correction(w)
#     spell_check[i] = word

# print(spell_check)
# all_txt = []
# cleaner = Cleaner()
# for title, text in iterate('nlwiki-latest-pages-articles-multistream.xml'):
#     text = cleaner.clean_text(text)
#     cleaned_text, _ = cleaner.build_links(text)
#     all_txt.append(cleaned_text)
#
# spell_text = ' '.join(all_txt)
# with open('spell.pkl', 'wb') as pickle_file:
#     pickle.dump(spell_text, pickle_file)
# with open('spell.pkl', 'rb') as pickle_file:
#     text = pickle.load(pickle_file)
# with io.open('spell.txt', 'w', encoding='utf-8') as f:
#     f.write(text)


# nlp = spacy.load('nl_core_news_lg')
# nlp.max_length = 2000000
#
#
# def df_printer(df):
#     with pd.option_context('display.max_rows', None, 'display.max_columns', None):
#         print(df)
#
# # with open('pkl/Data0.pkl', 'rb') as pickle_file:
# #     data = pickle.load(pickle_file)
#
#
# with open('pkl/Data0.pkl', 'rb') as pickle_file:
#     data = pickle.load(pickle_file)
# data = ' '.join(data)
# tokenize = [token.text for token in nlp(data)]
# print(tokenize)
# new_data = []
# for i in data:
#     li = []
#     for j in i:
#         if j != '':
#             li.append(j)
#     new_data.append(li)
#
# data = new_data
# mp = 'Data0/*.*'
# with open('pkl/' + mp[0:5] + '_fs.pkl', 'wb') as pickle_file:
#     pickle.dump(data, pickle_file)

# print(d)

# with open('pkl/dbscan_cl_w2v0_tr.pkl', 'rb') as pickle_file:
#     d1 = pickle.load(pickle_file)
# with open('pkl/dbscan_cl_w2v0_t.pkl', 'rb') as pickle_file:
#     d2 = pickle.load(pickle_file)

# token_words = [w.split(' ') for w in data]
# df_printer(d1)
#
# print(token_words)

# train_data, test_data = train_test_split(data)
#
# temp_vocab = [i.split(' ') for i in test_data]
# test_vocab = []
# for i in temp_vocab:
#     i2 = list(dict.fromkeys(i))
#     print(len(i), len(i2))
#     # test_vocab.append(i)

# words = [['dog sentecne of a dog','cat rule the world','horse carry humans','bird fly through the sky'],
#          ['house is were you live','roof is on the house','door is an gate','window is an oppertunity']]
#
# words = list(itertools.chain.from_iterable(words))
# x = [i.split(' ') for i in words]
# words = list(itertools.chain.from_iterable(x))
# print(words)


# test_data = train_test_split(data)[1]
#
#
# model = fasttext.load_model('cc.nl.300.bin')
#
# test_vocab = [i.split(' ') for i in test_data]
# new_vocab = []
# for i in test_vocab:
#     i = list(dict.fromkeys(i))
#     new_vocab.append(i)
# print(len(new_vocab))
# test_vectors = []
# for i in new_vocab:
#     vectors = []
#     for j in i:
#         v = model[j]
#         vectors.append(v)
#     test_vectors.append(vectors)
# print(len(test_vectors))
# vectors = data.vectors
# words = data.words
# for i in vectors:
#     print(i)
#
# a = data.words
# b = data2.words[2]
#
# nlp = spacy.load('nl_core_news_lg')
# train = [['dog','cat','horse','bird'], ['house','roof','door','window'], ['lion', 'race'], ['duck', 'car']]
# test = ['dog','cat','horse','house','car','lion','duck']
# pred = [0, 1]
#
# vector = nlp
#
# a = np.asarray([vector])
# most_similar = model.vocab.vectors.most_similar(a, n=10)
#
# def purity_score(y_true=test, y_pred=pred, train_list=train):
#     false = [0, 0]
#     for k in range(0, len(false)):
#         f = 0
#         for i in y_true:
#             for j in range(0, len(train_list)):
#                 pred_words = train_list[j]
#                 if i in pred_words and j != y_pred[k]:
#                     f += 1
#         false[k] = f
#
#     purity = []
#     for i in range(0, len(y_pred)):
#         cl = train_list[y_pred[i]]
#         match = [x for x in y_true if x in cl]
#         n = len(match)
#         p = 1 / len(y_true) * (n - false[i])
#         purity.append(p)
#     return purity
#
#
# purity_score()
