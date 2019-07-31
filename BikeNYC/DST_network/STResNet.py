'''
    ST-ResNet: Deep Spatio-temporal Residual Networks
'''

from __future__ import print_function
from tensorflow.python.keras.layers import (
    Input,
    Activation,
    Add,
    merge,
    Dense,
    Reshape
)
from tensorflow.python.keras.layers.convolutional import Convolution2D
from tensorflow.python.keras.layers.normalization import BatchNormalization
from tensorflow.python.keras.models import Model


# from tensorflow.python.keras.utils.visualize_util import plot


def _shortcut(input, residual):
    # return merge([input, residual], mode='sum')
    return Add()([input, residual])


def _bn_relu_conv(nb_filter, nb_row, nb_col, subsample=(1, 1), bn=False):
    def f(input):
        if bn:
            # input = BatchNormalization(mode=0, axis=1)(input)
            input = BatchNormalization(mode=0, axis=1)(input)
        activation = Activation('relu')(input)
        # return Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample, border_mode="same")(activation)
        return Convolution2D(nb_filter, (nb_row, nb_col),
                             strides=subsample, padding='same')(activation)

    return f


def _residual_unit(nb_filter, init_subsample=(1, 1)):
    def f(input):
        residual = _bn_relu_conv(nb_filter, 3, 3, subsample=init_subsample)(input)
        residual = _bn_relu_conv(nb_filter, 3, 3, subsample=init_subsample)(residual)
        return _shortcut(input, residual)

    return f


def ResUnits(residual_unit, nb_filter, repetations=1):
    def f(input):
        for i in range(repetations):
            init_subsample = (1, 1)
            input = residual_unit(nb_filter=nb_filter,
                                  init_subsample=init_subsample)(input)
        return input

    return f


def stresnet(c_conf=(3, 2, 32, 32), p_conf=(3, 2, 32, 32), t_conf=(3, 2, 32, 32),
             external_dim=8, nb_residual_unit=3, CF=64):
    '''
    C - Temporal Closeness
    P - Period
    T - Trend
    conf = (len_seq, nb_flow, map_height, map_width)
    external_dim
    '''

    # main input
    main_inputs = []
    outputs = []
    for conf in [c_conf, p_conf, t_conf]:
        if conf is not None:
            len_seq, nb_flow, map_height, map_width = conf
            input = Input(shape=(nb_flow * len_seq, map_height, map_width))
            main_inputs.append(input)
            # Conv1
            conv1 = Convolution2D(
                filters=CF, kernel_size=(3, 3), padding="same")(input)
            # [nb_residual_unit] Residual Units
            residual_output = ResUnits(_residual_unit, nb_filter=CF,
                                       repetations=nb_residual_unit)(conv1)
            # Conv2
            activation = Activation('relu')(residual_output)
            conv2 = Convolution2D(
                filters=nb_flow, kernel_size=(3, 3), padding="same")(activation)
            outputs.append(conv2)

    # parameter-matrix-based fusion
    if len(outputs) == 1:
        main_output = outputs[0]
    else:
        from BikeNYC.DST_network.ilayer import iLayer
        new_outputs = []
        for output in outputs:
            new_outputs.append(iLayer()(output))
        main_output = Add()(new_outputs)

    # fusing with external component
    if external_dim != None and external_dim > 0:
        # external input
        external_input = Input(shape=(external_dim,))
        main_inputs.append(external_input)
        embedding = Dense(units=10)(external_input)
        embedding = Activation('relu')(embedding)
        h1 = Dense(units=nb_flow * map_height * map_width)(embedding)
        activation = Activation('relu')(h1)
        external_output = Reshape((nb_flow, map_height, map_width))(activation)
        main_output = Add()([main_output, external_output])
    else:
        print('external_dim:', external_dim)

    main_output = Activation('tanh')(main_output)
    model = Model(input=main_inputs, output=main_output)

    return model


if __name__ == '__main__':
    model = stresnet(external_dim=28, nb_residual_unit=12)
    # plot(model, to_file='ST-ResNet.png', show_shapes=True)
    model.summary()
