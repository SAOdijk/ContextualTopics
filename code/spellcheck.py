from spellchecker import SpellChecker
from wiki_dump_reader import Cleaner, iterate
import io
import itertools
from iw_list import iw_list, ap_list
import pickle

all_txt = []
cleaner = Cleaner()
for title, text in iterate('nlwiki-latest-pages-articles-multistream.xml'):
    text = cleaner.clean_text(text)
    cleaned_text, _ = cleaner.build_links(text)
    all_txt.append(cleaned_text)

spell_text = ' '.join(all_txt)
with open('spell.pkl', 'wb') as pickle_file:
    pickle.dump(spell_text, pickle_file)
with open('spell.pkl', 'rb') as pickle_file:
    text = pickle.load(pickle_file)
with io.open('spell.txt', 'w', encoding='utf-8') as f:
    f.write(text)

spell = SpellChecker(language=None, case_sensitive=False)
spell.word_frequency.load_text_file('spell.txt')
spell.export('custom_dict.gz', gzipped=True)

spell = SpellChecker()
spell.word_frequency.load_dictionary('dictionaries/add_dict.gz')
w = []
with open('pkl/Data0_new.pkl', 'rb') as pickle_file:
    s_data = pickle.load(pickle_file)


def splitted_data(data):
    all_words = []
    for i in data:
        i = i.split(' ')
        all_words.append(i)
    unique_words = len(set(list(itertools.chain.from_iterable(all_words))))
    print('Total number of unique words: ' + str(unique_words))
    return all_words


def unknown_spell(data):
    # Check Spelling
    unknown = []
    for i in data:
        misspelled = spell.unknown(i)
        unknown.append(misspelled)
    unknown = list(set(itertools.chain.from_iterable(unknown)))
    print('Number of unknown words:', len(unknown))
    print(unknown)
    if len(unknown) == 0:
        data = [' '.join(i) for i in data]
        print(data)
        with open('pkl/data_corrected.pkl', 'wb') as p_file:
            pickle.dump(data, p_file)
    return unknown


def replace_values(data, unknown, improved):
    for i, x in enumerate(data):
        for j, a in enumerate(x):
            if a in unknown:
                index = unknown.index(a)
                data[i][j] = a.replace(a, improved[index])
    m = []
    for i in data:
        g = ' '.join(i)
        m.append(g)
    return m


def add_to_dict(words):
    spell.word_frequency.load_words(words)
    spell.export('dictionaries/add_dict.gz', gzipped=True)


def delete_values(data, r_list):
    for i, x in enumerate(data):
        for j, a in enumerate(x):
            if a in r_list:
                del data[i][j]
    return data


if __name__ == '__main__':
    s = splitted_data(s_data)
    u1 = unknown_spell(s)
    r = replace_values(s, u1, iw_list)
    s2 = splitted_data(r)
    u2 = unknown_spell(s2)
    d = delete_values(s2, u2)
    u3 = unknown_spell(d)
    add_to_dict(ap_list)
    print(u2)
