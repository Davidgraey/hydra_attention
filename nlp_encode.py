import numpy as np
#import pickle
import layers
import matplotlib.pyplot as plt
import itertools
import time

import loss_functions as loss

import scipy.sparse as sp
from collections import defaultdict

def cosine_similarity(u, v):
    ''' also Euclidian Distance
    u, v = word vectors for indiviudal words, output from our word2vec w2v.word_vec('input')
    if highly similar, return is close to 1
    if highly dissimilar, return is close to -1
    '''
    #dist = 0.0
    dot = u @ v
    # L2 norm is the length of the vector
    norm_u = np.linalg.norm(u)
    norm_v = np.linalg.norm(v)

    cosine_theta = dot / (norm_u * norm_v)

    return cosine_theta


def corpus_list_to_array(corpus):
    # assert len(corpus[1]) == len(corpus[5])
    token_sentence = len(corpus)
    word = len(corpus[0])
    corpus_array = np.zeros(shape=(token_sentence, word), dtype='int')
    # print(corpus_array.shape, token_sentence, word)
    for s in range(token_sentence):
        corpus_array[s, :] = np.array(corpus[s], dtype='int')

    return corpus_array


class Glove:
    def __init__(self, dimensionality, window_size):
        self.dim = dimensionality
        self.window_size = window_size

        self.word_to_index = {}
        self.index_to_word = {}

        self.init_weights = False

    def generate_training_data(self, corpus, vocab, minimum_gate=None):
        '''

        :param corpus:
        :param vocab:
        :param minimum_gate:
        :return:
        '''
        #####################################################
        word_counts = defaultdict(int)
        for sentence in corpus:
            for token in sentence:
                word_counts[token] += 1
        self.vocab_size = len(word_counts.keys()) + 1

        # Put together LookUpTables
        self.words_list = list(word_counts.keys())
        for list_i, word in enumerate(self.words_list):
            self.word_to_index.update((word, int(list_i)))
            self.index_to_word.update((int(list_i), word))
        #####################################################

        training_data = []

        # self.cooccurrences = np.zeros((self.vocab_size, self.vocab_size), dtype=np.float64)
        self.cooccurrences = sp.lil_matrix((self.vocab_size, self.vocab_size), dtype=np.float64)

        for i, line in enumerate(corpus):
            token_ids = [self.word_to_index[word] for word in line]

            for center_i, center_id in enumerate(token_ids):
                # Collect all word IDs in left window of center word
                context_ids = token_ids[max(0, center_i - self.window_size): center_i]
                contexts_len = len(context_ids)

                for left_i, left_id in enumerate(context_ids):
                    # weight by inverse distance between words
                    distance = contexts_len - left_i
                    dist_w = 1.0 / float(distance)

                    self.cooccurrences[center_id, left_id] += dist_w
                    self.cooccurrences[left_id, center_id] += dist_w

        # this version from
        # https://github.com/hans/glove.py/blob/582549ddeeeb445cc676615f64e318aba1f46295/glove.py#L171-182
        # this is using sparse from scipy for self.coocurrences - will have to figure out...
        for i, (row, data) in enumerate(itertools.izip(self.cooccurrences.rows, self.cooccurrences.data)):

            if minimum_gate is not None and vocab[self.index_to_word[i]][0] < minimum_gate:
                continue

            for data_i, j in enumerate(row):
                if minimum_gate is not None and vocab[self.index_to_word[j]][0] < minimum_gate:
                    continue

                yield i, j, data[data_i]


    def train(self, vocab, epochs=10, learning_rate=0.002, x_max=100, alpha=0.75):
        self.error_trace = []
        if not self.init_weights:
            self.weights = (np.random.rand(self.vocab_size * 2, self.dim) - 0.5) / float(self.dim + 1)
            self.biases = (np.random.rand(self.vocab_size * 2) - 0.5) / float(self.dim + 1)

            self.grad_squares = np.ones((self.vocab_size * 2, self.dim), dtype=np.float64)
            self.grad_bias = np.ones(self.vocab_size * 2, dtype=np.float64)

            data = self.get_views()

        for i in range(epochs):
            full_cost = 0
            np.shuffle(data)
            for (weights, weights_context, bias, bias_context, grad_squares, grad_squares_context,
                 grad_bias, grad_bias_context, cooccurrence) in data:
                weight = (cooccurrence / x_max) ** alpha if cooccurrence < x_max else 1

                cost_inner = (weights @ weights_context + bias[0] + bias_context[0] - np.log(cooccurrence))
                cost = weight * (cost_inner ** 2)
                full_cost += 0.5 * cost

                grad_main = cost_inner * weights_context
                grad_context = cost_inner * weights

                grad_bias_main = cost_inner
                grad_bias_context = cost_inner

                # weight updates
                weights -= (learning_rate * grad_main / np.sqrt(grad_squares))
                weights_context -= (learning_rate * grad_context / np.sqrt(grad_squares_context))

                bias -= (learning_rate * grad_bias_main / np.sqrt(grad_bias))
                bias_context -= (learning_rate * grad_bias_context / np.sqrt(grad_bias_context))

                # Update squared gradient sums
                grad_squares += np.square(grad_main)
                grad_squares_context += np.square(grad_context)
                grad_bias += grad_bias_main ** 2
                grad_bias_context += grad_bias_context ** 2

            self.error_trace.append(full_cost)

    def get_views(self):
        return [(self.weights[i_main],
                 self.weights[i_context + self.vocab_size],
                 self.biases[i_main: i_main + 1],
                 self.biases[i_context + self.vocab_size: i_context + self.vocab_size + 1],
                 self.grad_squares[i_main],
                 self.grad_squares[i_context + self.vocab_size],
                 self.grad_bias[i_main: i_main + 1],
                 self.grad_bias[i_context + self.vocab_size: i_context + self.vocab_size + 1],
                 cooccurrence) for
                i_main, i_context, cooccurrence in self.cooccurrences]


# --------------------------------------------------------------------------------------------------------------------

class BlindEmbed:
    '''create like a layer-like weight structure to embed words without context or position'''

    def __init__(self, dimensionality, vocab_size, w2i, i2w):
        '''
        :param dimensionality: number of output dimensions - size of vector representation of encoded words
        '''
        self.dim = dimensionality
        self.vocab_size = vocab_size

        self.weights = np.random.normal(size=(self.vocab_size, self.dim))

        self.word_to_index = w2i
        self.index_to_word = i2w

    def forward(self, corpus):
        '''
        :param corpus: incoming corpus must be a integer representation or indexed corpus, not words.  It should be a
        :return:
        '''
        return np.take(self.weights, corpus, axis=0)  # output shape = (n_sentences, sentence_length, dimensionality)

    def convert_w2i(self, word):
        '''
        :param word: single word
        :return:
        '''
        return self.word_to_index[word]

    def convert_w2v(self, word):
        w_i = self.convert_w2i(word)
        w_v = self.weights[w_i]
        return w_v

    def convert_i2w(self, index):
        return self.index_to_word(index)

    def convert_i2v(self, index):
        return self.weights[index]

    def convert_v2i(self, vector):
        
        pass

    def convert_v2w(self, vector):
        pass

    def index_to_onehot(self, word_i):
        '''
        call only after we've generated training data
        :param word:
        :return:
        '''
        # one_hot = [0 for i in range(0, self.vocab_size)]
        one_hot = np.zeros((self.vocab_size))
        one_hot[word_i] = 1
        return one_hot.tolist()

# --------------------------------------------------------------------------------------------------------------------

class WordEmbed:
    '''create like a layer-like Word2Vec object to build a vector representation of a token using CBOW (predicting
    surrounding contexts from a single word'''

    def __init__(self, dimensionality, window_size):
        '''
        :param dimensionality: number of output dimensions - size of vector representation of encoded words
        :param window_size: steps before and after to create window of relevant words
        '''
        self.dim = dimensionality
        self.window_size = window_size

        self.activation = 'linear'
        self.output_activation = 'softmax'
        self.loss_func = 'cwd'  # context window difference

        self.layers = []
        self.vocab_size = 0
        self.words_list = []
        self.word_to_index = {}
        self.index_to_word = {}

    def generate_training_data(self, corpus, w2i, i2w, vocab, vocab_size):
        '''

        :param corpus:
        :param w2i:
        :param i2w:
        :param vocab:
        :param vocab_size:
        :return:
        '''
        #####
        print(f'loaded corpus of {vocab_size}')
        self.vocab_size = vocab_size
        self.vocab = vocab
        self.word_to_index = w2i
        self.index_to_word = i2w
        """"######################################################
        # Find unique word counts using dictonary
        word_counts = defaultdict(int)
        for sentence in corpus:
            for token in sentence:
                word_counts[token] += 1
        self.vocab_size = len(word_counts.keys())

        # assemble Look Up Tables
        self.words_list = list(word_counts.keys())
        for list_i, word in enumerate(self.words_list):

            self.word_to_index[word] = list_i
            self.index_to_word[list_i] = word
            #self.word_to_index.update((word, list_i))
            #self.index_to_word.update((list_i, word))
        """  #####################################################
        training_data = []

        for sentence in corpus:
            for i, word in enumerate(sentence):
                w_hot = self.index_to_onehot(sentence[i])
                w_context = []
                for j in range(i - self.window_size, i + self.window_size + 1):
                    if j != i and j <= (len(sentence) - 1) and j >= 0:
                        w_context.append(self.index_to_onehot(sentence[j]))

                training_data.append([w_hot, w_context])
        training_data = np.array(training_data)
        print(f'generated context training for {self.vocab_size} unique tokens \n Training data is shaped '
              f'{training_data.shape}')
        return training_data

    def train(self, corpus, w2i, i2w, vocab, vocab_size, epochs=10, learning_rate=0.02):
        assert type(corpus[0][0]) == int

        training_data = self.generate_training_data(corpus, w2i, i2w, vocab, vocab_size)

        if self.layers == []:
            self.add_layer(self.vocab_size, self.dim, self.activation)
            self.add_layer(self.dim, self.vocab_size, self.output_activation, is_output=True)
            print(f'added layers {self.layers}')
        start = time.time()
        self.error_trace = []

        for i in range(epochs):
            batch_error = []
            for incoming_word, contexts in training_data:
                vec, output, value_output = self.forward(incoming_word)
                error = loss.loss_dictionary[self.loss_func](output, contexts)
                gradients = self.backward(error)
                self.update_weights(-learning_rate * gradients)

                value_output = np.squeeze(value_output)
                batch_error.append(-np.sum([value_output[word.index(1)] for word in contexts]) + len(
                    contexts) * np.log(np.sum(np.exp(value_output))))

            self.error_trace.append(np.mean(batch_error))
            print(f'epoch {i}, has sumLoss of {self.error_trace[-1]}')
        print('time in training for full net', time.time() - start)
        return self.error_trace

    def forward(self, incoming_x):
        '''

        :param incoming_x: incoming data to pass forward, shapes must match.
        :return: output of full pass, value output (non-softmax output)
        '''
        outs = np.expand_dims(incoming_x, 0)
        # value_output = None
        vec = layers[0].forward(outs)
        outputs = layers[1].forward(vec)
        value_outputs = layers[1].forward(vec, forced_activation='linear')
        return vec, value_outputs, outputs

    def backward(self, delta):
        '''

        :param delta:
        :return:
        '''
        for li in reversed(range(len(self.layers))):
            # print('inputs', self.layers[li].input.shape)
            # print('layer_shape', self.layers[li])
            # print('delta', delta.shape)
            delta = self.layers[li].backward(delta)
        return self.gradients

    def update_weights(self, values):
        '''**********ARGUMENTS**********
        :param values: values created by optimizer using gradients, amount by which to change layer weight values
        '''
        for layer, gradient in zip(self.layers, values):
            # print(f'layer is {layer.shape}, grad is {gradient.shape}')
            try:  # try inserting a new axis if shapes don't match
                layer.update_weight(gradient[..., np.newaxis])
            except ValueError:
                layer.update_weight(gradient)
            except TypeError:
                layer.update_weight(gradient)

    # -------------------
    def word_to_onehot(self, word):
        '''
        call only after we've generated training data
        :param word:
        :return:
        '''
        one_hot = np.zeros((self.vocab_size))
        w_i = self.word_to_index[word]
        one_hot[w_i] = 1
        return one_hot.tolist()

    def index_to_onehot(self, word_i):
        '''
        call only after we've generated training data
        :param word_i: word index
        :return:
        '''
        one_hot = np.zeros((self.vocab_size))
        one_hot[word_i] = 1
        return one_hot.tolist()

    def get_vector(self, word):
        w_i = self.word_to_index[word]
        vec = self.layers[0].weights[w_i + 1]
        return vec

    def get_similar_words(self, word, n_similar):
        w_vec = self.get_vector(word)
        similar = {}

        for i in range(self.vocab_size):
            next_w_vec = self.layers[0].weights[i + 1]
            theta_dot = w_vec @ next_w_vec
            theta_norm = np.linalg.norm(w_vec) * np.linalg.norm(next_w_vec)
            theta = theta_dot / theta_norm

            word = self.index_to_word[i]
            similar[word] = theta

        words_sorted = sorted(similar.items(), key=lambda kv: kv[1], reverse=True)

        for word, sim in words_sorted[:n_similar]:
            print(word, sim)

    def get_word_map(self):
        w2v_map = {}
        for word in self.words_list:
            vec = self.get_vector(word)
            w2v_map[word] = vec
        return w2v_map

    def complete_analogy(self, word_a, word_b, word_c):
        e_a = self.get_vector(word_a)
        e_b = self.get_vector(word_b)
        e_c = self.get_vector(word_c)

        max_cosine_sim = -100
        best_word = None
        input_words_set = set([word_a, word_b, word_c])

        for w in self.words_list:
            # to avoid best_word being one of the input words, skip the input words
            if w in input_words_set:
                continue

            cosine_sim = cosine_similarity(e_b - e_a, self.get_vector(w) - e_c)
            if cosine_sim > max_cosine_sim:
                max_cosine_sim = cosine_sim
                best_word = w

        return best_word

    def purge(self):
        pass

    def load_pretrained(self, filepath):
        # loads a fully pretrained word32vec embedding layers
        #
        # assert that dimensions match before continuing.
        pass

    # ------------------------------PROPERTIES---------------------------------------
    @property
    def gradients(self):
        g = []
        for l in self.layers:
            g.append(l.gradient)
        try:
            g = np.squeeze(g)
        except ValueError:
            pass
        return g

    @property
    def outputs(self):
        o = []
        for l in self.layers:
            o.append(l.output)
        try:
            o = np.squeeze(o)
        except ValueError:
            try:
                o = np.array(o)
            except ValueError:
                pass
        return o

    @property
    def value_outputs(self):
        '''
        Value Outputs is a straight linear output from each cascade, rather than a softmax or sigmoid or other
        funciton when doing classification - these value outputs should be summed from all cascades, then have the
        activation applied to get the full network's classification.
        :param:
        ---
        :return:
        '''
        return self.value_output

    @property
    def inputs(self):
        i = []
        for l in self.layers:
            i.append(l.input)
        try:
            i = np.squeeze(i)
        except ValueError:
            pass
        return i

    @property
    def shapes(self):
        s = []
        for l in self.layers:
            s.append(l.shape)
        try:
            s = np.squeeze(s)
        except ValueError:
            pass
        return s

    @property
    def weights(self):
        w = []
        for l in self.layers:
            w.append(l.weights)
        try:
            w = np.squeeze(w)
        except ValueError:
            pass
        return w

# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

class PositionalEncode:
    '''create a layer-like object to add positionality data to our existing embeded representation of a token  -
    embed_layer = PositionEncode(token_length, dimensionality) - dimensionality must match the dimension used in
    Embedding layers'''
    def __init__(self, token_length, dimensionality):
        '''
        :param token_length: length of sentence or token block
        :param dimensionality: Dimensionality of the model (must match attention layer's output dimension
        '''
        self.length = token_length
        self.dim = dimensionality

        self.output = None
        self.positional_modifier = np.array([])
        self.positional_encoding()


    def get_angles(self, pos, i):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(self.dim))
        return pos * angle_rates


    def positional_encoding(self):
        angle_rads = self.get_angles(np.arange(self.length)[:, np.newaxis], np.arange(self.dim)[np.newaxis, :])

        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2]) #apply sine to even indices 2i
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2]) #apply cosine to odd indices 2i+1

        self.positional_modifier = angle_rads[np.newaxis, ...]


    def forward(self, incoming_x):
        return incoming_x + self.positional_modifier

    def backward(self, incoming_delta):
        return incoming_delta - self.modifier


    def visualize_encoding(self, output_encoding):
        plt.figure(figsize=(12, 8))
        plt.pcolormesh(output_encoding[0], cmap='viridis')
        plt.xlabel('Embedding Dimensions')
        plt.xlim((0, self.dim))
        plt.ylim((self.length, 0))
        plt.ylabel('Token Position')
        plt.colorbar()
        plt.show()

    # position_encoding = positional_encoding(n_tokens_per_review, embed_dim)

    # plt.imshow(position_encoding[0, :, :]);

"""
if __name__ == '__main__':

   with open('whale_corpus.pickle', 'rb') as handle:
        corpus = pickle.load(handle)

    tokenizer = Tokenizer()
    corpus, indexed_corpus = tokenizer.tokenize(filename='doraingray.txt', pad_to_length=256, build_luts=True)
    w2i, i2w = tokenizer.get_luts()
    vocab, vocab_size = tokenizer.get_vocab()
    print(vocab_size, len(i2w))

    array_corp = corpus_list_to_array(indexed_corpus)

    simple_embed = BlindEmbed(dimensionality=25, vocab_size=vocab_size, w2i=w2i, i2w=i2w)
    vectorized = simple_embed.forward(array_corp)
    pos_embed = PositionalEncode(token_length=256, dimensionality=25)

    for v_i, sent in enumerate(vectorized):
        sent = pos_embed.forward(sent)
        vectorized[v_i] = sent

    print(vectorized.shape)

    # with open('vectorized_corpus.pkl', 'wb') as handle:
    #    pickle.dump(vectorized, handle)


    for i in range(vocab_size):
        print(corpus[i])
        print(indexed_corpus[i])



    xs = indexed_corpus[:10]
    start_time = time.time()
    model = WordEmbed(dimensionality=25, window_size=10)
    error_trace = model.train(xs, w2i, i2w, vocab, vocab_size, epochs = 2, learning_rate = 0.002)
    print('network code took', time.time() - start_time)
    plt.plot(error_trace)
    plt.show()



    start_time = time.time()
    settings = {
        'window_size': 10,  # context window +- center word
        'n': 50,  # dimensions of word embeddings, also refer to size of hidden layer
        'epochs': 50,  # number of training epochs
        'learning_rate': 0.002  # learning rate
    }

    w2v = word2vec()
    training_data = w2v.generate_training_data(settings, xs)
    w2v.train(training_data)
    print('raw code took', time.time() - start_time)


    #------
    print(model.get_similar_words('whale', 5))
    #finish = model.complete_analogy('whale', 'captain', 'ship')
    #print(f'whale is to captain as ishamel is to {finish}')

    word = "whale"
    vec = w2v.word_vec(word)
    w2v.vec_sim(word, 5)

    # Find similar words



    # TODO: Now that tokenizer is done and working, find a way to get the LUTs into GLOVE and word2vec so we don't
    # rebuild vocab.
   """