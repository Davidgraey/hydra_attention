import dill as pickle

import numpy as np

import layers
import activation_functions as activation
import loss_functions as loss


class AttentionHead():
    '''
    Our attention block assumes that words have already been tokenized
    For Attention Encoding:
        Query - Current Word (encoded as a vector AND including positional data) in the INPUT sentence
        Key - All words (encoded as before) in the INPUT sentence
        Value - All words (encoded as before) in the INPUT sentence
    For Attention Decoding:
        Query - Current Word (encoded as a vector AND including positional data) in the OUTPUT sentence
        Key - All words (encoded as before) in the OUTPUT sentence
        Value - All words (encoded as before) in the OUTPUT sentence
    For Encoder-Decoder :
        Query - the output of the decoder's attention
        Key - All of the encoder's hidden state vectors
        Value - All of the encoder's hidden state vectors

    For Generation -
        Next token prediction :
            Masked

    '''
    def __init__(self, vector_dimension, projected_dimension, position, masked = False, dropout_percent = 0.1):
        '''
        :param ni: Vector Dimension of embed layer - this will be the vector dimension from our embeding / number of
            heads used in our multi-headed attention
        :param no: Output Dimensionality of the model - this could be any reduction of vector dimension, make it % 0
        for simplicity's sake
        '''
        self.ni = vector_dimension
        self.no = projected_dimension
        self.dropout_percent = dropout_percent

        self.head_id = position

        self.activation = 'linear'

        #TODO: change from this layer strucutre to keep the persisent input X, and not copy inputs multiple times
        # Refer to the drawing for residual connections

        self.query_layer = layers.Layer(self.ni, self.no, self.activation)
        self.key_layer = layers.Layer(self.ni, self.no, self.activation)
        self.value_layer = layers.Layer(self.ni, self.no, self.activation)

        self.dropout = layers.DropoutLayer(self.ni, self.no, self.dropout_percent)

        # TODO: add masking
        if masked:
            self.mask = np.ma.masked_array()

        self.training_now = False


    def forward(self, incoming_x):
        # incoming data should be (sentence length, projected_dimension)
        print('x shape into head is', incoming_x.shape)
        n_sents, vector_dimension, projected_dimension = incoming_x.shape
        assert vector_dimension == self.ni

        Q = self.query_layer.forward(incoming_x)
        K = self.key_layer.forward(incoming_x)
        V = self.value_layer.forward(incoming_x)

        # scaled dot product
        score = (Q @ K.T) / np.sqrt(self.no)

        #MASK would occur here,
        #self.mask[:, :, :T, :T] == 0, -np.inf)

        score = activation.softmax(score)
        self.score = self.dropout.forward(score, self.training_now)
        self.output = self.score @ V

        return self.output


    def backward(self, incoming_delta):

        v_delta = incoming_delta @ self.score.T

        score_delta = incoming_delta @ self.value_layer.output.T

        score_delta = self.dropout.backward(score_delta) # if we're in backward we should always be training.
        score_delta *= activation.derivative_dictionary['softmax'](self.score)

        # decomposing (Q @ K.T) / np.sqrt(self.no)
        q_score_delta = score_delta @ (self.key_layer.output.T / np.sqrt(self.no)).T #may not need .Ts?
        k_score_delta = score_delta @ (self.query_layer.output / np.sqrt(self.no)).T

        q_delta = self.query_layer.backward(q_score_delta)
        k_delta = self.query_layer.backward(k_score_delta)
        v_delta = self.value_layer.backward(v_delta)

        return self.outputs, self.gradients, [q_delta, k_delta, v_delta]


    def train(self, x):
        self.training_now = True
        pass

    def use(self, x):
        self.training_now = False
        pass

    def decay_dropout(self):
        self.dropout_percent *= 0.8

    def visualize_attention(self):
        # build attention plot
        pass

    @property
    def gradients(self):
        g = []
        g.append(self.query_layer.gradient)
        g.append(self.key_layer.gradient)
        g.append(self.value_layer.gradient)
        try:
            g = np.squeeze(g)
        except ValueError:
            pass
        return g


#------------------------------------------------------------------------------------------------------------

class AttentionNeck:
    '''multi-headed attention
    AttentionNeck is the container for the multi-headed attention and the concatination of all their outputs in
    self.bottleneck.
    '''
    def __init__(self, num_heads, vector_dimension, projected_dimension, dropout=0.01):
        '''
        The attention "neck" is the connection structure around all multi-heads.
        '''
        self.num_heads = num_heads  # 8
        self.vector_dimension = vector_dimension
        self.projected_dimension = projected_dimension

        self.activation = 'linear'

        # each attention head is shape (vectordim, projecteddim)
        self.heads = []
        pos = 0
        # TODO: look into explicitly distributing the head processing?

        while len(self.heads) < self.num_heads:
            self.heads.append(AttentionHead(vector_dimension=self.vector_dimension,
                                            projected_dimension=self.projected_dimension,
                                            position=pos,
                                            masked=False,
                                            dropout_percent=dropout))
            pos += 1

        # neck - single projection of all concatenated attention head outputs
        #TODO: figure out where the issue is with layer size in bottleneck - could need -1???
        self.bottleneck = layers.Layer(ni=projected_dimension * num_heads,
                                       no=projected_dimension,
                                       activation_type=self.activation)


    def forward(self, incoming_x):
        '''
        :param incoming_x:
        :return:
        '''
        head_outs = []
        # self.input = copy.copy(incoming_x
        for head in self.heads:
            head_outs.append(head.forward(incoming_x))
        print(f'One Hydra head outputs {head_outs[0].shape}')
        head_outs = np.concatenate(head_outs, axis=0)
        print(f'All Hydra Heads stacked outputs {head_outs.shape}')
        neck_outs = self.bottleneck.forward(head_outs.T)
        print(f'swallowed through the bottleneck {neck_outs.shape}')
        self.output = neck_outs + incoming_x  # Layer Normalization step if on batches?

        return self.output


    def backward(self, incoming_delta):
        incoming_delta = self.bottleneck.backward(incoming_delta)
        head_deltas = []
        start = 0
        stop = self.projected_dimension
        # select only each head's
        for head_i, head in enumerate(self.heads): #may have to reverse? or reverse our indexing...
            assert (head.head_id) == head_i  # just to be safe, make sure we're matching the correct head.
            head_deltas.append(head.backward(incoming_delta[start:stop, ...]))
            start = stop
            stop += self.projected_dimension

        return head_deltas


    def train(self, incoming_x, incoming_t, learning_rate_eta, momentum_alpha, epochs):
        n_samples = incoming_x.shape[0]
        selector = np.array(np.arange(0, n_samples))
        # shuffle?
        for sent_i in selector:
            _x = incoming_x[sent_i, ...]
            _t = incoming_t[sent_i, ...]
            output = self.forward(_x)

            error = loss.loss_dictionary[self.loss_func](output, _t)


#-----------------------------------------------------------------------------------------------------------------


class HydraAttention:
    ''''
    multi-headed attention
    Hydra Attention is the top-level class that holds the AttentionNeck and brain_layers (dense nnet layers at the
    end of the attention mechanism.
    AttentionNeck is the wrapper that holds the multiheaded attention.
    #TODO: Encoder / Decoder arcchitecture would take the place of the brain_layers
    #TODO: Hold a persisting RESIDUAL_X within this class and use for residual connections within the attention heads and other residual connections (check drawing)
    '''
    def __init__(self, num_heads, num_hiddens, vector_dimension, projected_dimension, dropout = 0.01):

        self.attention_block = AttentionNeck(num_heads, vector_dimension, projected_dimension, dropout)

        self.projected_dimension = projected_dimension
        #self.vector_dimension = vector_dimension

        self.loss_func = 'mse'
        self.activation = 'tanh'

        self.nhs = num_hiddens

        self.brain_1 = layers.Layer(self.projected_dimension, self.nhs, self.activation)
        self.brain_2 = layers.Layer(self.nhs, self.nhs, self.activation)
        self.brain_3 = layers.Layer(self.nhs, self.projected_dimension, 'linear', is_output = True)
        self.brain_layers = [self.brain_1, self.brain_2, self.brain_3]

        self.RESIDUAL_X = np.array()


    def forward(self, incoming_x):
        '''
        :param incoming_x:
        :return:
        '''
        #n_sentences, n_tokens, vector_dimension = incoming_x.shape

        attn_out = self.attention_block.forward(incoming_x)
        #print('attention_block output', attn_out.shape)
        for brain_layer in self.brain_layers:
            attn_out = brain_layer.forward(attn_out)
        self.output = attn_out

        return self.output


    def backward(self, incoming_delta):
        delta = incoming_delta
        for layer in reversed(self.brain_layers):
            delta = layer.backward(delta)

        delta = self.attention_block.backward(delta)

        return delta


    def train(self, incoming_x, incoming_t, learning_rate_eta, momentum_alpha, epochs):
        n_samples = incoming_x.shape[0]
        selector = np.array(np.arange(0, n_samples))
        #shuffle?
        for sent_i in selector:
            _x = incoming_x[sent_i, ...]
            _t = incoming_t[sent_i, ...]
            output = self.forward(_x)

            error = loss.loss_dictionary[self.loss_func](output, _t)

            # update using self..gradient
            #attention ws, neck w, self.dense ws


    def make_standalone(self, n_classes, num_hidden ):
        self.standalone = True

        self.brain_layers.append(layers.Layer(self.projected_dimension, num_hidden, self.activation))
        self.brain_layers.append(layers.Layer(num_hidden, n_classes, activation_type = 'softmax'))

        self.loss_func = 'cross-entropy'
        print('added final nonlinear and softmax output layer to model')

    @property
    def gradients(self):
        g = []
        #head_g = []
        for head_l in self.attention_block.heads:
            g.append(head_l.gradients)  # each entry is Query, Key, Value - [3, NI, NO]
        g.append(self.attention_block.bottleneck.gradient)
        for l in self.brain_layers:
            g.append(l.gradient)
        #g is a list, num_heads + 1(bottleneck) + num_brain_layers
        try: #remove the try/except from squeeze
            g = np.squeeze(g)
        except ValueError:
            pass
        return g



if __name__ == '__main__':

    #with open('vectorized_corpus.pkl', 'rb') as handle:
    #    corpus = pickle.load(handle)


    x = np.random.uniform(-1, 1, (32, 32, 32))

    hydra = HydraAttention( num_heads = 8,
                            num_hiddens = 32,
                            vector_dimension = 32,
                            projected_dimension = 32,
                            dropout = 0.01)


    outs = hydra.forward(x)
    #hydra.train(x, x,learning_rate_eta = 0.002, momentum_alpha = 0.002, epochs = 1 )
    #x.shape