from gensim.models.keyedvectors import KeyedVectors
import pandas as pd
import re

def vector_distance():
    gensim_model = KeyedVectors.load_word2vec_format(
            'GoogleNews-vectors-negative300.bin', binary=True, limit=300000)

    type_table = pd.read_csv('type.csv')
    dic = {}
    for word in range(len(type_table)):
        dic[word] = type_table['Type'][word]

    word_list = []
    new_dic = {}
    for word in dic:
        word = re.split('[-_]', dic[word])
        try:
            new_dic[dic[word]] = np.mean(gensim_model[word], axis = 0)
        except:
            word_list.append(dic[word])

    vec = np.zeros((340,300))
    row = 0
    for word in new_dic:
        vec[row] = new_dic[word]
        row += 1

    np.save('./vector.npy', vec)

if __name__ == '__main__':
     vector_distance()