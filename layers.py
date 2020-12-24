import numpy as np
import activation_functions as func
import copy

#-------------    weight initilization functions    ---------------
# ------------------------------------------------------------------
def xavier(ni, no):
    return np.random.normal(loc = 0.0, scale = 1 / np.sqrt(ni), size = (ni+1, no))


def kaiming(ni, no):
    '''weight init function for linear / relu functions with zero bias'''
    w = np.random.normal(loc = 0.0, scale = np.sqrt(2 / ni), size = (ni+1, no))
    w[0] = 0
    return w


weight_init = {'linear': kaiming,
               'relu': kaiming,
               'relu_leaky': kaiming,
               'sigmoid': xavier,
               'tanh': xavier,
               'softmax': xavier}

# ------------------------------------------------------------------

class Layer:
    def __init__(self, ni, no, activation_type, is_output=False):
        '''**********ARGUMENTS**********
        :param ni: number of input units
        :param no: number of output units
        :param activation_type: string identifying activation type, 'linear', 'sigmoid', 'tanh', etc.
        :param is_output: boolean flag designating if this is an output layer or hidden layer
        '''
        self.activation = activation_type
        self.is_output = is_output
        self.weights = weight_init[activation_type](ni, no)
        self.shape = self.weights.shape

        # these values will be rewritten or updated on each pass
        self.output = 0.0
        self.input = 0.0
        self.gradient = 0.0


    def forward(self, incoming_x, forced_activation = False):
        '''**********ARGUMENTS**********
        :param incoming_x: input data that is already standardized, if called for
        **********RETURNS**********
        :return: product of forward pass of incoming values
        '''
        self.input = copy.copy(incoming_x)
        # bias units are self.weights[0:1, :]
        if forced_activation != False:
            outs = func.activation_dictionary[forced_activation](incoming_x @ self.weights[1:, :] + self.weights[0:1,
                                                                                                    :])
        else:
            outs = func.activation_dictionary[self.activation](incoming_x @ self.weights[1:, :] + self.weights[0:1, :])
            self.output = outs
        return outs


    def backward(self, incoming_delta):
        '''**********ARGUMENTS**********
        :param incoming_gradient: delta from previous step of backprop
        **********RETURNS**********
        :return: returns this layer's gradient contribution
        '''
        activated_delta = incoming_delta * func.derivative_dictionary[self.activation](self.output)
        this_delta = self.input.T @ activated_delta
        bias_delta = np.sum(activated_delta, 0)
        self.gradient = np.vstack((bias_delta, this_delta)) #make gradient persistent in layer

        grad_contribution = incoming_delta @ self.weights[1:, ...].T
        return grad_contribution


    def update_weight(self, value):
        '''**********ARGUMENTS**********
        :param value: variable to update this layer's weights by with - this will already have Learning rate,
        depreication or momentum / other calculations addressed in the upper level.
        '''
        self.weights += value


    def purge(self):
        '''
        resets values tracked during training to zero
        '''
        self.input = 0.0
        self.output = 0.0
        self.gradient = 0.0

    def __str__(self):
        return f'Layer of {self.activation}, shaped {self.shape} -- output is {self.is_output}'


    def __repr__(self):
        return f'{self.shape}, using {self.activation}'


#TODO: FUTURE - add below layer types : As we add more layer types, we will have to change our cascade assembly
# construction slightly to better differentiate type.


class DropoutLayer(Layer):
    '''Hinton style Dropout layer with scaling outputs'''
    def __init__(self, ni, no, dropout_prob = 0.5):
        self.ni = ni
        self.no = no
        self.shape = [self.ni, self.no]
        self.dropout_prob = dropout_prob

        self.generate_dropout()

        self.input = [0]
        self.output = [0]


    def generate_dropout(self):
        self.weights = np.random.binomial([np.ones((self.ni, self.no))],
                                          1 - self.dropout_prob)[0] * (1.0 / (1 - self.dropout_prob))
        return

    def set_dropout_value(self, value):
        self.dropout_prob = value
        return


    def forward(self, incoming_x, training_now, forced_activation = False):
        self.input = copy.copy(incoming_x)
        if training_now:
            self.output = self.weights * incoming_x

        else:
            self.output = self.input

        return self.output


    def backward(self, incoming_delta):
        incoming_delta *= self.weights
        incoming_delta = self.input.T * incoming_delta

        self.gradient = incoming_delta

        return self.gradient

    def __str__(self):
        return f'Layer of Hinton Dropout currently at {self.dropout_prob}, shaped {self.shape}'

    def __repr__(self):
        return f'{self.shape}, using Hinton Dropout'


#TODO: Create Normalize Layer that can be Undone in backpass (hold 'sqhashing_factor')
class NormalizeLayer(Layer):
    def __init__(self):
        #super().__init__(ni = , no = , activation_type = , is_output = )
        self.squashing_factor = 0

    def forward(self, incoming_x, training_now):

    #normalize layer will normalize all values between 0 and 1; !!!record that squashing factor


#TODO: Create attention layer (KQV layer within a single head) that does NOT track inputs - these will be the same
# across all heads - this layer will use a RESIDUAL_X from container / parent.  Complicated to figure out exactily
# how to propigate through.
class AttentionLayer():
    '''
    Attention layer does not track inputs; since they are shared across heads, Keys Query and Value
    '''
    def __init__(self, ni, no, activation_type, is_output=False):
        '''**********ARGUMENTS**********
        :param ni: number of input units
        :param no: number of output units
        :param activation_type: string identifying activation type, 'linear', 'sigmoid', 'tanh', etc.
        :param is_output: boolean flag designating if this is an output layer or hidden layer
        '''
        self.activation = activation_type
        self.is_output = is_output
        self.weights = weight_init[activation_type](ni, no)
        self.shape = self.weights.shape

        # these values will be rewritten or updated on each pass
        self.output = 0.0
        self.gradient = 0.0