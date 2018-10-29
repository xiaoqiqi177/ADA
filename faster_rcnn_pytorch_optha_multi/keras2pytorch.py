from __future__ import print_function

from nn_transfer import util, transfer
import numpy as np
import h5py
import keras
import torch

VAR_AFFIX = ':0' if keras.backend.backend() == 'tensorflow' else ''

KERAS_GAMMA_KEY = 'gamma' + VAR_AFFIX
KERAS_KERNEL_KEY = 'kernel' + VAR_AFFIX
KERAS_ALPHA_KEY = 'alpha' + VAR_AFFIX
KERAS_BIAS_KEY = 'bias' + VAR_AFFIX
KERAS_BETA_KEY = 'beta' + VAR_AFFIX
KERAS_MOVING_MEAN_KEY = 'moving_mean' + VAR_AFFIX
KERAS_MOVING_VARIANCE_KEY = 'moving_variance' + VAR_AFFIX
KERAS_EPSILON = 1e-3
PYTORCH_EPSILON = 1e-5
    
def transfer_keras2pytorch(keras_mdl, own_state):
    flip_filters = not keras.backend.backend() == 'tensorflow'

    with h5py.File(keras_mdl, 'r') as f:
        model_weights = f['model_weights']
        layer_names = list(map(str, model_weights.keys()))

        own_state_keys = list(own_state.keys())
        for layer in layer_names:
            if 'block' in layer and 'pool' not in layer:
                try:
                    params = util.dig_to_params(model_weights[layer])
                except:
                    import IPython
                    IPython.embed()
                block_no = layer[5]
                layer_no = str(int(layer[11])-1)
                weight_key = 'rpn.features.conv'+block_no+'.'+layer_no+'.conv.weight'
                bias_key = 'rpn.features.conv'+block_no+'.'+layer_no+'.conv.bias'
                running_mean_key = layer + '.running_mean'
                running_var_key = layer + '.running_var'
            else:
                continue
            
            #load weights
            if weight_key in own_state_keys:
                if KERAS_GAMMA_KEY in params:
                    weights = params[KERAS_GAMMA_KEY][:]
                elif KERAS_KERNEL_KEY in params:
                    weights = params[KERAS_KERNEL_KEY][:]
                else:
                    weights = np.squeeze(params[KERAS_ALPHA_KEY][:])
                weights = transfer.convert_weights(weights,
                        to_keras=True,
                        flip_filters=flip_filters)
                own_state[weight_key].copy_(torch.from_numpy(weights))
                print('weight: ', weight_key)
            #load bias
            if bias_key in own_state_keys:
                if running_var_key in own_state_keys:
                    bias = params[KERAS_BETA_KEY][:]
                else:
                    bias = params[KERAS_BIAS_KEY][:]
                own_state[bias_key].copy_(torch.from_numpy(bias.transpose()))
                print('bias: ', bias_key)

