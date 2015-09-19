# TODO: implement this, still just normal RNN cell.
from __future__ import print_function
import cgt
import numpy as np
from cgt.nn import parameter, init_array, Constant, HeUniform

# Gate setup:
#
# * = matrix mult, . = elementwise mult
#
# hidden(t) = (1 - update(t)) * hidden(t-1) + update(t) * ~hidden(t)
# ~hidden(t) = tanh(matrix * input + matrix * (reset(t) . hidden(t-1)))
# update(t) =  sigmoid(matrix * input + matrix * hidden(t-1))
# reset(t) = sigmoid(matrix * input + matrix * hidden(t-1))
#
# Chung, Junyoung, et al.
# "Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling."
# arXiv preprint arXiv:1412.3555 (2014).
#
# In the above paper:
# z is used as notation for the update gate
# r as notation for the reset gate
class GRUCell(object):
    def __init__(self, input_size, hidden_size, output_size,
            name="", weight_init=HeUniform(1.0)):
        """
        Initialize an RNN cell
        """
        # TODO: bias
        # The paper makes no mention of bias in equations or text.
        # Sooo I'm not sure we need it.

        # reset gate
        self.W_xr = parameter(init_array(weight_init, (input_size, hidden_size)), name=name+".W_input_to_reset")
        self.W_hr = parameter(init_array(weight_init, (hidden_size, hidden_size)), name=name+"W_hidden_to_reset")

        # update gate
        self.W_xz = parameter(init_array(weight_init, (input_size, hidden_size)), name=name+".W_input_to_update")
        self.W_hz = parameter(init_array(weight_init, (hidden_size, hidden_size)), name=name+"W_hidden_to_update")

        # ~hidden is the candidate activation, so we'll denote it as c
        self.W_xc = parameter(init_array(weight_init, (input_size, hidden_size)), name=name+".W_input_to_candidate")
        self.W_hc = parameter(init_array(weight_init, (hidden_size, hidden_size)), name=name+"W_hidden_to_candidate")


    def __call__(self, x, h):
        """
        x is the input
        h is the input from the previous timestep

        Returns (out, next_h). Feed out into the next layer and
        next_h to the next timestep.
        """

        reset_gate = cgt.sigmoid(x.dot(self.W_xr) + h.dot(self.W_hr))
        update_gate = cgt.sigmoid(x.dot(self.W_xz) + h.dot(self.W_hz))

        # the elementwise multiplication here tells what of the previous
        # input we should forget.
        forget_gate = reset_gate * h

        # this part is very similar to vanilla RNN
        next_h = cgt.tanh(x.dot(self.W_xc) + h.dot(forget_gate))
        out = (1 - update_gate) * h + update_gate * next_h

        return out, next_h

# Make sure it compiles!

x = cgt.matrix() # (batch_size, n_features)
h = cgt.matrix() # this will later be the identity matrix

o, next_h = GRUCell(5, 10, 5)(x, h)
print("Output:", o, cgt.infer_shape(o))
print("Next Hidden:", next_h, cgt.infer_shape(next_h))
