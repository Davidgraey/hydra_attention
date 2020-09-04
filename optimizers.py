import numpy as np
import loss_functions as loss

# ----------------------------------CONSTANTS------------------------------------------------------
lr_activation_scaler = {'linear': 1.0, #.09
                        'relu': 1.0,
                        'relu_leaky': 1.0,
                        'sigmoid': 1.0,
                        'tanh': 1.0}


EPSILON = 1e-15


def find_sqrt(n_dimensional_array):
    finished = []
    for this_array in n_dimensional_array:
        try:
            these_sqrts = np.sqrt(this_array)
        except AttributeError:
            deeper_finished = []
            for deeper_array in this_array:
                sub_squares = np.sqrt(deeper_array)
                deeper_finished.append(sub_squares)
            these_sqrts = np.array(deeper_finished)
        finished.append(these_sqrts)
    return np.array(finished)


# ------------------------------------------------------------------------------------------------------------
# ----------------------------------CASCADE OPTIMIZERS-----------------------------------------------------
""" Beginning work on optimizer as a seperate oop rather than a complex function call - this will let us do things 
like second order / double-step SCG
class BasalCascade:
    def __init__(self):
        pass

    def train(self, incoming_x, incoming_t, epochs...):
        #current epoch loop in here...
        pass

    def step(self):
        #takes a single step
        pass

    def get_error(self):
        return self.error_trace

"""

# ------------------------------------------------------------------------------------------------------------
# ----------------------------------CASCADE OPTIMIZERS------------------------------------------------------
#TODO: make momentum / mt&vt for adam pulled out into cascades and network to make persistent across all epochs and
# epochs. stored in each cascade and network
def cascade_sgd_momentum(incoming_x, incoming_t, epochs, this_cascade, error_trace, learning_rate_eta=0.02,
                         momentum_alpha=0.96, batch_size=4):
    '''**********ARGUMENTS**********
    :param incoming_x: input values - incoming data should be standardized
    :param incoming_t: target values - targets should be standardized
    :param epochs: number of updates
    :param this_cascade: passes in cascade assembly object
    :param error_trace: error trace object from network object
    :param learning_rate_eta: learning rate value for nnet weight updates
    :param momentum_alpha: Momentum to roll down gradient
    :param batch_size: split batch as 1/nth of the full dataset at a time
    '''
    n_samples = incoming_x.shape[0]
    # n_outputs = incoming_t.shape[-1]

    # make indexing array, shuffle it and use it to index X and T to create batches
    selector = np.array(np.arange(0, n_samples))
    #batch_size = int((1 / batch_size) * n_samples)

    #velocity = this_cascade.momentum

    for i in range(0, epochs):
        np.random.shuffle(selector)
        batch_error = []

        for batch_i in range(0, n_samples, batch_size):
            this_batch = selector[batch_i: batch_i + batch_size]
            # print(f'training on {batch_i} : {batch_i + batch_size}, {this_batch.shape}')
            _x = incoming_x[this_batch, ...]
            _t = incoming_t[this_batch, ...]

            output = this_cascade.forward(_x)

            error = loss.loss_dictionary[this_cascade.loss_func](output, _t)

            d_error = loss.derivative_dictionary[this_cascade.loss_func + '_derivative'](output, _t)
            #TODO: add L2 / weight decay: seperate method / function here -
            #TODO: L2 = costF + gamma / n_samples * sum (abs(weights))
            #add gamma/n * w to der of all weights - leave biases intact.
            delta = d_error / batch_size

            output, gradients, delta = this_cascade.backward(delta)

            activation_modifier = lr_activation_scaler[this_cascade.hidden_activation]
            adjusted_lr = learning_rate_eta * activation_modifier

            # update V factors in this batch
            this_cascade.momentum = (momentum_alpha * this_cascade.momentum) - (adjusted_lr * gradients)

            #Update weights
            this_cascade.update_weights(this_cascade.momentum)

            batch_error.append(error)

        error_trace.append(np.mean(batch_error))


def cascade_adam(incoming_x, incoming_t,
                 epochs, this_cascade,
                 error_trace, learning_rate_eta=0.02,
                 batch_size=4):
    # pseudo code from Dr Chuck Anderson - Colorado State University

    n_samples = incoming_x.shape[0]
    selector = np.array(np.arange(0, n_samples))
    #batch_size = int((1 / batch_size) * n_samples)

    beta1 = 0.9
    beta2 = 0.9
    beta1t = 1
    beta2t = 1

    for i in range(0, epochs):
        np.random.shuffle(selector)
        batch_error = []

        for batch_i in range(0, n_samples, batch_size):
            this_batch = selector[batch_i: batch_i + batch_size]

            _x = incoming_x[this_batch, ...]
            _t = incoming_t[this_batch, ...]

            output = this_cascade.forward(_x)

            error = loss.loss_dictionary[this_cascade.loss_func](output, _t)

            d_error = loss.derivative_dictionary[this_cascade.loss_func + '_derivative'](output, _t)
            delta = d_error / batch_size

            _outs, gradients, delta = this_cascade.backward(delta)

            this_cascade.mt = beta1 * this_cascade.mt + (1 - beta1) * gradients
            this_cascade.vt = beta2 * this_cascade.vt + (1 - beta2) * gradients * gradients

            beta1t *= beta1
            beta2t *= beta2

            m_hat = this_cascade.mt / (1 - beta1t)
            v_hat = this_cascade.vt / (1 - beta2t)

            activation_modifier = lr_activation_scaler[this_cascade.hidden_activation]
            adjusted_lr = (learning_rate_eta) * activation_modifier


            value = -adjusted_lr * m_hat / (find_sqrt(v_hat) + EPSILON)
            this_cascade.update_weights(value)


            batch_error.append(error)

        error_trace.append(np.mean(batch_error))

"""
def cascade_scg(incoming_x, incoming_t, epochs, this_cascade, error_trace, batch_size=4):
    '''Adapted from Dr Charles Anderson - CSU'''
    
    sigma0 = 1.0e-6
    complete = True  # Force calculation of directional derivs.
    beta = 1.0e-6  # Initial scale parameter. Lambda in Moeller.
    betamin = 1.0e-15  # Lower bound on scale.
    betamax = 1.0e20  # Upper bound on scale.

    shape = this_cascade.weights
    n_layers = len(shape)

    #w_new = np.zeros_like(shape)
    w_temp = np.zeros(shape)
    g_new = np.zeros(shape)
    g_old = np.zeros(shape)
    g_smallstep = np.zeros(shape)
    search_dir = np.zeros(shape)


    n_n_complete = 0  # n_n_complete counts number of n_complete.
    iteration = 1 #counts iterations



    n_samples = incoming_x.shape[0]
    selector = np.array(np.arange(0, n_samples))

    for i in range(0, epochs):
        np.random.shuffle(selector)
        batch_error = []

        for batch_i in range(0, n_samples, batch_size):
            this_batch = selector[batch_i: batch_i + batch_size]

            _x = incoming_x[this_batch, ...]
            _t = incoming_t[this_batch, ...]

            if complete:
                mu = search_dir @ g_new
                if mu >= 0:
                    search_dir[:] = - g_new
                    mu = search_dir.T @ g_new

                #kappa for
                kappa = search_dir.T @ search_dir
                if math.isnan(kappa):
                    print('kappa', kappa)
                    
                if kappa < floatPrecision:
                    return error_trace

                sigma = sigma0 / np.sqrt(kappa)
                w_temp[:] = all_weights
                all_weights += sigma * search_dir
                #forward and gradient calculation
                output = this_cascade.forward(_x)
                error = loss.loss_dictionary[this_cascade.loss_func](output, _t)
                d_error = loss.derivative_dictionary[this_cascade.loss_func + '_derivative'](output,_t)
                delta = d_error / batch_size
                _outs, gradients, _delta = this_cascade.backward(delta)
                g_smallstep[:] = gradients
                #all_weights[:] = w_temp
                
                
                theta = search_dir @ (g_smallstep - g_new) / sigma
            
            #increase effective curvature and evaluate step size alpha.

            delta = theta + beta * kappa
            if math.isnan(delta):
                print('delta is NaN', 'theta', theta, 'beta', beta, 'kappa', kappa)
            elif delta <= 0:
                delta = beta * kappa
                beta = beta - theta / kappa

            if delta == 0:
                complete = False
                f_now = f_old
            else:
                alpha = -mu / delta
                ## Calculate the comparison ratio Delta
                w_temp[:] = all_weights
                all_weights += alpha * search_dir #weight update
                #f_new = error_f(*fargs) #forward pass and loss func returns f_new
                        output = this_cascade.forward(_x)
                        error = loss.loss_dictionary[this_cascade.loss_func](output, _t)
                        d_error = loss.derivative_dictionary[this_cascade.loss_func + '_derivative'](output,_t)
                        delta = d_error / batch_size
                        fnew = delta
                    
                Delta = 2 * (f_new - f_old) / (alpha * mu)
                if not math.isnan(Delta) and Delta >= 0:
                    complete = True
                    n_complete += 1
                    # w[:] = wnew
                    f_now = f_new
                else:
                    complete = False
                    f_now = f_old
                    all_weights[:] = w_temp

            error_trace.append(error_convert_f(f_now))

            if complete:

                f_old = f_new
                g_old[:] = g_new
                g_new[:] = gradient_f(*fargs)

                # If the gradient is zero then we are done.
                gg = g_new @ g_new  # dot(gradnew, gradnew)
                if gg == 0:
                    return error_trace

            if math.isnan(Delta) or Delta < 0.25:
                beta = min(4.0 * beta, betamax)
            elif Delta > 0.75:
                beta = max(0.5 * beta, betamin)

            # Update search direction using Polak-Ribiere formula, or re-start
            # in direction of negative gradient after nparams steps.
            if n_complete == n_layers: #if we have a complete for each layer:
                search_dir[:] = -g_new
                n_complete = 0
            elif complete:
                gamma = (g_old - g_new) @ (g_new / mu)
                # search_dir[:] = gamma * search_dir - g_new
                search_dir *= gamma
                search_dir -= g_new


            batch_error.append(error)
                # print(batch_error)
        error_trace.append(np.mean(batch_error))
"""





# ------------------------------------------------------------------------------------------------------------
# ----------------------------------NETWORK OPTIMIZERS------------------------------------------------------

def network_sgd_momentum(incoming_x, incoming_t,
                         network_object, error_trace, epochs,
                         learning_rate_eta=0.02, momentum_alpha=0.5, batch_size=4):
    '''**********ARGUMENTS**********
    :param incoming_x: input values - incoming data should be standardized
    :param incoming_t: target values - targets should be standardized
    :param epochs: number of updates
    :param network_object: passes in as self from Adaptive_Cascade Object
    :param error_trace: error trace object from network object
    :param learning_rate_eta: learning rate value for nnet weight updates
    :param momentum_alpha: Momentum to roll down gradient
    :param batch_size: split batch as 1/nth of the full dataset at a time
    '''
    n_samples = incoming_x.shape[0]
    #n_inputs = incoming_x.shape[-1]
    #n_outputs = incoming_t.shape[-1]

    selector = np.array(np.arange(0, n_samples))
    #batch_size = int((1 / batch_size) * n_samples)

    #velocity = np.zeros_like(network_object.weights)

    for i in range(0, epochs):
        np.random.shuffle(selector)
        batch_error = []

        for batch_i in range(0, n_samples, batch_size):
            this_batch = selector[batch_i: batch_i + batch_size]
            # print(f'training on {batch_i} : {batch_i + batch_size}, {this_batch.shape}')

            _x = incoming_x[this_batch, ...]
            _t = incoming_t[this_batch, ...]

            full_output = network_object.forward(_x)
            error = loss.loss_dictionary[network_object.network[-1].loss_func](full_output, _t)
            d_error = loss.derivative_dictionary[network_object.network[-1].loss_func + '_derivative'](full_output, _t)
            delta = d_error / batch_size

            for c_i, this_cascade in enumerate(reversed(network_object.network)):

                _out, gradients, delta = this_cascade.backward(delta)

                #adjusted_lr = (learning_rate_eta / batch_size)
                this_cascade.momentum = -(momentum_alpha * this_cascade.momentum) - (learning_rate_eta * gradients)

                this_cascade.update_weights(this_cascade.momentum)
                delta = d_error / batch_size

            #full_output = network_object.forward(_x)
            batch_error.append(error)
            # print(batch_error)
        error_trace.append(np.mean(batch_error))


def network_adam(incoming_x, incoming_t,
                 epochs, network_object,
                 error_trace, learning_rate_eta=0.02,
                 batch_size=4):
    #pseudo code from Dr Chuck Anderson - Colorado State University

    n_samples = incoming_x.shape[0]

    selector = np.array(np.arange(0, n_samples))
    #batch_size = int((1 / batch_size) * n_samples)

    beta1 = 0.9
    beta2 = 0.98
    beta1t = 1
    beta2t = 1

    for i in range(0, epochs):
        np.random.shuffle(selector)
        batch_error = []

        for batch_i in range(0, n_samples, batch_size):
            this_batch = selector[batch_i: batch_i + batch_size]
            # print(f'training on {batch_i} : {batch_i + batch_size}, {this_batch.shape}')

            _x = incoming_x[this_batch, ...]
            _t = incoming_t[this_batch, ...]

            output = network_object.forward(_x)

            error = loss.loss_dictionary[network_object.network[-1].loss_func](output, _t)
            d_error = loss.derivative_dictionary[network_object.network[-1].loss_func + '_derivative'](output, _t)
            delta = d_error / batch_size

            for c_i, this_cascade in enumerate(reversed(network_object.network)):
                _outs, gradients, delta = this_cascade.backward(delta)

                this_cascade.mt = beta1 * this_cascade.mt + (1 - beta1) * gradients
                this_cascade.vt = beta2 * this_cascade.vt + (1 - beta2) * gradients * gradients

                beta1t *= beta1
                beta2t *= beta2

                m_hat = this_cascade.mt / (1 - beta1t)
                v_hat = this_cascade.vt / (1 - beta2t)

                value = -learning_rate_eta * m_hat / (find_sqrt(v_hat) + EPSILON)
                this_cascade.update_weights(value)

                delta = d_error / batch_size


            this_error = error
            #if this_error.shape != batch_error[0].shape:
            #    np.pad(this_error, (0, (batch_error[0].shape[0], this_error.shape[0])))
            batch_error.append(this_error)
        #print(batch_error)
        error_trace.append(np.mean(batch_error))

'''
def network_swat(incoming_x, incoming_t,
                 epochs, network_object,
                 error_trace, learning_rate_eta=0.02,
                 momentum_alpha=0.5, batch_size=4):

    n_samples = incoming_x.shape[0]
    n_inputs = incoming_x.shape[-1]
    n_outputs = incoming_t.shape[-1]

    reshaping_matrix = -1 * np.ones((n_inputs, n_outputs))

    selector = np.array(np.arange(0, n_samples))
    batch_size = int((1 / batch_size) * n_samples)

    mt = np.zeros_like(network_object.weights)  # mean moment
    vt = np.zeros_like(network_object.weights)  # variance moment
    velocity = np.zeros_like(network_object.weights) # velocity for SGD momentum

    beta1 = 0.9
    beta2 = 0.999
    beta1t = 1
    beta2t = 1

    #for e_i in range(0, epochs):

    for i in range(0, epochs):
        np.random.shuffle(selector)
        batch_error = []

        # calc SWATS projection and find switchover point
        # Pseudocode
        # if e_i > 1 and abs((lambda_k / (1-beta2)) - gamma_k

        # pk = adam_step
        # pk_T = perpandicular (T) line to adam_step
        # gk = gradient step

        # gamma_k = (pk_T*pk) / (-pk_t * gk)
        # lambda_k = (beta2 * current_lr) + (1-beta2)*gamma_k

        # if e_i > 1 and abs((lambda_k / (1 - beta2)) - gamma_k) < 1e-9
        # new_lr = lamba_k / (1 - beta2)

        for batch_i in range(0, n_samples, batch_size):
            this_batch = selector[batch_i: batch_i + batch_size]
            # print(f'training on {batch_i} : {batch_i + batch_size}, {this_batch.shape}')

            _x = incoming_x[this_batch, ...]
            _t = incoming_t[this_batch, ...]

            output = network_object.forward(_x)

            delta = _t - output

            for c_i, this_cascade in enumerate(network_object.network):

                _outs, gradients, delta = this_cascade.backward(delta)

                #Select which training?

                #if ADAM - standard update
                mt[c_i] = beta1 * mt[c_i] + (1 - beta1) * gradients
                vt[c_i] = beta2 * vt[c_i] + (1 - beta2) * gradients * gradients
                beta1t *= beta1
                beta2t *= beta2

                m_hat = mt[c_i] / (1 - beta1t)
                v_hat = vt[c_i] / (1 - beta2t)

                adjusted_lr = (learning_rate_eta / batch_size)

                value = adjusted_lr * m_hat / (find_sqrt(v_hat) + EPSILON)
                this_cascade.update_weights(value)

                #SGD
                #calculate LR
                adjusted_lr = (learning_rate_eta / batch_size)
                velocity[c_i] = (momentum_alpha * velocity[c_i]) - (adjusted_lr * gradients)

                this_cascade.update_weights(velocity[c_i])


    pass
'''