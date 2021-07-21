from inputs import w2v, split_data, d2v
from stats import stats_to_df
from model import train_w2v, train_lda
import pickle
from pyclustertend import hopkins, vat
from sklearn.preprocessing import scale
from visualisation import w2visual
import timeit
import datetime
from PIL import Image
import itertools

train_model = False
create_inputs = True
model_info = False
create_stats = False
m = 'optics_cl'
m1, m2, m3 = ['w2v', 'd2v', 'lda']
c1, c2, c3 = ['w2v0_optics_cl', 'd2v0_optics_cl', 'lda0']


# def create_pkl_data():
#     path = 'C:/Users/Steven/Dropbox/2.Documents/UU/IBM/Thesis/'
#     mp = 'Data0/*.*'
#     processed_data = cleanup(path + mp)
#     # processed_data2 = cleanup(path + mp, True)
#     with open('pkl/' + mp[0:5] + '_spell.pkl', 'wb') as pickle_file:
#         pickle.dump(processed_data, pickle_file)
#     # with open('pkl/' + mp[0:5] + '_fs.pkl', 'wb') as pickle_file:
#     #     pickle.dump(processed_data2, pickle_file)
#
#
# # Create data
# if create_data:
#     create_pkl_data()


# Combined data, highlight reports, project files
def open_data_pkl():
    with open('pkl/data_corrected.pkl', 'rb') as pickle_file:
        g_data = pickle.load(pickle_file)
    # with open('pkl/Data0_fs.pkl', 'rb') as pickle_file:
    #     d_data = pickle.load(pickle_file)
    return g_data


gen_data = open_data_pkl()
if train_model:
    train_w2v(gen_data, m1)
    # train_w2v(d2v_data, m2)
    train_lda(gen_data)


def exec_inputs(data, l_model):
    train_sets, test_sets = split_data(data)[0], split_data(data)[1]
    inputs = []
    for i in range(0, len(train_sets)):
        r = w2v(train_sets[i], test_sets[i], l_model)
        print('Inputs: ' + str(i + 1) + '/' + str(len(train_sets)))
        inputs.append(r)
    with open('pkl/inputs.pkl', "wb") as f:
        pickle.dump(inputs, f)
    # elif l_model == 'd2v':
    #     d2v(train_data, test_data, l_model)


if create_inputs:
    exec_inputs(gen_data, m1)
    # d2v0 = exec_inputs(d2v_data, m2)
if model_info:
    with open('pkl/inputs.pkl', 'rb') as pf:
        results = pickle.load(pf)
    data_sample = results[0]
    train_vocab, train_vectors, test_vocab, test_vectors = data_sample
    vo = list(itertools.chain.from_iterable(test_vectors))
    x = scale(train_vectors + vo)
    y = scale(train_vectors)
    # print('Hopkins score:', hopkins(x, 1000))
    start_t = timeit.default_timer()
    v = vat(y)
    print(v)
    stop_t = timeit.default_timer()
    seconds = stop_t - start_t
    print('Time of vat in: ' + str(datetime.timedelta(seconds=seconds)))
    # w2visual(train_vocab)



def open_df_pkl():
    with open('pkl/' + m + '_' + m1 + '_tr.pkl', 'rb') as pickle_file:
        w2v_tr_df = pickle.load(pickle_file)
    with open('pkl/' + m + '_' + m1 + '_t.pkl', 'rb') as pickle_file:
        w2v_t_df = pickle.load(pickle_file)
    w2v_list = [w2v_tr_df, w2v_t_df]
    # with open('pkl/' + m + '_' + m2 + '_tr.pkl', 'rb') as pickle_file:
    #     d2v_tr_df = pickle.load(pickle_file)
    # with open('pkl/' + m + '_' + m2 + '_t.pkl', 'rb') as pickle_file:
    #     d2v_t_df = pickle.load(pickle_file)
    d2v_list = [0, 0]
    return w2v_list, d2v_list


def df_stats(dfs, syntax):
    train_df, test_df = dfs
    stats_to_df(train_df, test_df, syntax, False)


if create_stats:
    w2v_df, d2v_df = open_df_pkl()
    df_stats(w2v_df, c1)
    # df_stats(d2v_df, c2)
