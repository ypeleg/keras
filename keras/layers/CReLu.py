from __future__ import print_function
from keras.layers import Dense, Dropout, Layer, Activation
from keras import backend as K
from keras.utils import np_utils


class CReLU(Layer):
    '''
	Based on: https://arxiv.org/pdf/1603.05201v2.pdf
    '''
    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2  
        shape[-1] *= 2
        return tuple(shape)

    def call(self, x, mask=None):
        pos = K.relu(x)
        neg = K.relu(-x)
        return K.concatenate([pos, neg], axis=1)

		
class CReLUReLU(Layer):
    '''
	Not Based on: https://arxiv.org/pdf/1603.05201v2.pdf :)
    '''
    def get_output_shape_for(self, input_shape):
        shape = list(input_shape)
        assert len(shape) == 2  
        shape[-1] *= 2
        return tuple(shape)

    def call(self, x, mask=None):
        pos = K.relu(x)
        neg = K.relu(-x)
		con = K.concatenate([pos, neg], axis=1)
        return K.relu(con)
