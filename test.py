import numpy as np
from scipy.sparse import csr_matrix
import sframe
from sklearn.metrics.pairwise import euclidean_distances


def load_csr(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']
    return csr_matrix((data, indices, indptr), shape)


def unpack_dict(matrix, map_index_to_word):
    table = list(map_index_to_word.sort('index')['category'])
    data = matrix.data
    indices = matrix.indices
    indptr = matrix.indptr
    num_doc = matrix.shape[0]
    return [{k: v for k, v in zip([table[word_id] for word_id in indices[indptr[i]: indptr[i + 1]]],
                                  data[indptr[i]: indptr[i + 1]].tolist())} for i in xrange(num_doc)]


def top_word(name):
    row = wiki[wiki['name'] == name]
    word_count_table = row[['word_count']].stack('word_count', new_column_name=['word', 'count'])
    return word_count_table.sort('count', ascending=False)


# question_2
# def getrow(name):
#     id = wiki[wiki['name'] == name][0]['id']
#     return word_count[id]

# question_1
# def has_top_words(word_count_vector):
#     unique_words = set(word_count_vector.keys())
#     # print len(unique_words)
#     if set(common_words).issubset(unique_words): return True
#     return False

wiki = sframe.SFrame('people_wiki.gl/')
# print wiki.shape
wiki = wiki.add_row_number()

word_count = load_csr('people_wiki_word_count.npz')
# print word_count.shape
map_index_to_word = sframe.SFrame('people_wiki_map_index_to_word.gl/')
wiki['word_count'] = unpack_dict(word_count, map_index_to_word)


# question_1

obama_words = top_word('Barack Obama')
barrio_words = top_word('George W. Bush')
combined_words = obama_words.join(barrio_words, on='word').rename({'count':'Obama_count', 'count.1': 'bush_count'})
combined_words.sort('Obama_count', ascending=False).print_rows(30)

common_words = combined_words.head(5)['word'].to_numpy()
print common_words, type(common_words)

wiki['has_top'] = wiki['word_count'].apply(has_top_words)
num = 0
for v in wiki['has_top']:
    if v: num += 1

print num

# question_2
# print euclidean_distances(getrow('Barack Obama'), getrow('George W. Bush'))
# print euclidean_distances(getrow('Barack Obama'), getrow('Joe Biden'))
# print euclidean_distances(getrow('Joe Biden'), getrow('George W. Bush'))

# question_3
# sf = top_word('Barack Obama').join(top_word('George W. Bush'), on='word')
# wrd_count = sf.sort('count', ascending=False)
# wrd_count.print_rows(10)

# question_4
