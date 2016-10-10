import numpy as np
import sframe
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors


def load_sparse_matrix(filename):
    loader = np.load(filename)
    data = loader['data']
    indices = loader['indices']
    indptr = loader['indptr']
    shape = loader['shape']
    return csr_matrix((data, indices, indptr), shape)


def unpack_dict(map_index_to_word, matrix):
    table = list(map_index_to_word.sort('index')['category'])
    num_doc = matrix.shape[0]
    indices = matrix.indices
    indptr = matrix.indptr
    data = matrix.data
    return [{k: v for k, v in zip([table[word_id] for word_id in indices[indptr[i]: indptr[i + 1]]] \
                                  , data[indptr[i]: indptr[i + 1]])}for i in xrange(num_doc)]


def top_word_tf_itf(name):
    row = wiki[wiki['name'] == name]
    return row[['tf_idf']].stack('tf_idf', new_column_name=['word', 'tf_idf']).sort('tf_idf', ascending=False)


# question_2
def has_top_word(tf_idf_vector):
    common_wrds = set(common_words)
    unique_wrds = set(tf_idf_vector.keys())
    return common_wrds.issubset(unique_wrds)

tf_idf = load_sparse_matrix('people_wiki_tf_idf.npz')
wiki = sframe.SFrame('people_wiki.gl/')
wiki.add_row_number('id')
wiki['id'] = range(wiki.shape[0])
map_index_to_word = sframe.SFrame('people_wiki_map_index_to_word.gl/')
wiki['tf_idf'] = unpack_dict(map_index_to_word, tf_idf)

print wiki

# question_1
# model = NearestNeighbors(metric='euclidean', algorithm='brute')
# model.fit(tf_idf)
# distance, indices = model.kneighbors(tf_idf[35817], n_neighbors=10)
# neighbor = sframe.SFrame({'id': indices.flatten(), 'distance': distance.flatten()})
# wiki.add_row_number()
# wiki['id'] = range(wiki.shape[0])
# neighbors = neighbor.join(wiki, on='id').sort('distance')[['id', 'name', 'distance']]
# neighbors.print_rows(10, 3, 150)

# question_2
obama_tf_idf = top_word_tf_itf('Barack Obama')
schiliro_tf_idf = top_word_tf_itf('Phil Schiliro')
common_words = obama_tf_idf.join(schiliro_tf_idf, on='word').sort('tf_idf', ascending=False).head(5)['word'].to_numpy()
print common_words
wiki['has_top'] = wiki['tf_idf'].apply(has_top_word)

num = 0
for v in wiki['has_top']:
    if v: num += 1

print num

# question_3
# obama_id = wiki[wiki['name'] == 'Barack Obama'][0]['id']
# biden_id = wiki[wiki['name'] == 'Joe Biden'][0]['id']
# print euclidean_distances(tf_idf[obama_id], tf_idf[biden_id])
