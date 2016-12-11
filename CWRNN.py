from keras.layers import SimpleRNN
import numpy as np
import keras.backend as K

class ClockworkRNN(SimpleRNN):
    '''
        Clockwork Recurrent Unit - Koutnik et al. 2014
        Clockwork RNN splits simple RNN neurons into groups of equal sizes.
        Each group is activated every specified period. As a result, fast
        groups capture short-term input features while slow groups capture
        long-term input features.
        References:
            A Clockwork RNN
                http://arxiv.org/abs/1402.3511
    '''
    def __init__(self, output_dim, period_spec=[1],**kwargs):
        self.output_dim = output_dim
        assert output_dim % len(period_spec) == 0, ("ClockworkRNN requires the output_dim to be " +
                                                "a multiple of the number of periods; " +
                                                "output_dim %% len(period_spec) failed.")
        self.period_spec = np.asarray(sorted(period_spec, reverse=True))
       
        super(ClockworkRNN, self).__init__(output_dim, **kwargs)

    def build(self, input_shape):

        ### construct the clockwork structures
        ### basically: every n units the period changes;
        ### `period` is for flaggin this; `mask` is for enforcing it
        n = self.output_dim // len(self.period_spec)
        mask = np.zeros((self.output_dim, self.output_dim), K.floatx())
        period = np.zeros((self.output_dim,), np.int16)
        for i, t in enumerate(self.period_spec):
            mask[i*n:(i+1)*n, i*n:] = 1
            period[i*n:(i+1)*n] = t
        self.mask = K.variable(mask, name='clockword_mask')
        self.period = K.variable(period, dtype='int16', name='clockwork_period')

        super(ClockworkRNN, self).build(input_shape)

        self.U = self.U * self.mask  ### old implementation did this at run time... 

        ### simple rnn initializes the wrong size self.states
        ### we want to also keep the time step in the state. 
        if self.stateful:
            self.reset_states()
        else:
            self.states = [None, None]



    def get_initial_states(self, x):
        initial_states = super(ClockworkRNN, self).get_initial_states(x)
        if self.go_backwards:
            input_length = self.input_spec[0].shape[1]
            initial_states[-1] = float(input_length)
        else:
            initial_states[-1] = K.variable(0.)
        return initial_states


    def reset_states(self):
        assert self.stateful, 'Layer must be stateful.'
        input_shape = self.input_spec[0].shape
        if not input_shape[0]:
            raise Exception('If a RNN is stateful, a complete ' +
                            'input_shape must be provided (including batch size).')

        if self.go_backwards:
            initial_time = self.input_spec[0].shape[1]
        else:
            initial_time = 0.

        if hasattr(self, 'states'):
            K.set_value(self.states[0],
                        np.zeros((input_shape[0], self.output_dim)))
            K.set_value(self.states[1], initial_time)
        else:
            self.states = [K.zeros((input_shape[0], self.output_dim)), K.variable(initial_time)]

    def get_constants(self, x):
        consts = super(ClockworkRNN, self).get_constants(x)
        consts.append(self.period)
        return consts

    def step(self, x, states):
        prev_output = states[0]
        time_step = states[1]
        B_U = states[2]
        B_W = states[3]
        period = states[4]

        if self.consume_less == 'cpu':
            h = x
        else:
            h = K.dot(x * B_W, self.W) + self.b

        output = self.activation(h + K.dot(prev_output * B_U, self.U))
        output = K.switch(K.equal(time_step % period, 0.), output, prev_output)
        return output, [output, time_step+1]


    def get_config(self):
        config = {"period_spec": self.period_spec}
        base_config = super(ClockworkRNN, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
