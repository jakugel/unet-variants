from tensorflow.keras.layers import Conv2D, MaxPooling2D, concatenate, Conv2DTranspose
from tensorflow.keras.layers import BatchNormalization, Input, Activation, Add, GlobalAveragePooling2D, Reshape, Dense, multiply, Permute, maximum, Concatenate, Multiply
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model


def attention_unet(filters, output_channels, width=None, height=None, input_channels=1, conv_layers=2):
    def conv2d(layer_input, filters, conv_layers=2):
        d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(layer_input)
        d = BatchNormalization()(d)
        d = Activation('relu')(d)

        for i in range(conv_layers - 1):
            d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(d)
            d = BatchNormalization()(d)
            d = Activation('relu')(d)

        return d

    def deconv2d(layer_input, filters):
        u = Conv2DTranspose(filters, 2, strides=(2, 2), padding='same')(layer_input)
        u = BatchNormalization()(u)
        u = Activation('relu')(u)
        return u

    def attention_block(F_g, F_l, F_int):
        g = Conv2D(F_int, kernel_size=(1, 1), strides=(1, 1), padding='valid')(F_g)
        g = BatchNormalization()(g)
        x = Conv2D(F_int, kernel_size=(1, 1), strides=(1, 1), padding='valid')(F_l)
        x = BatchNormalization()(x)
        psi = Add()([g, x])
        psi = Activation('relu')(psi)

        psi = Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='valid')(psi)
        psi = Activation('sigmoid')(psi)

        return Multiply()([F_l, psi])

    inputs = Input(shape=(width, height, input_channels))

    conv1 = conv2d(inputs, filters, conv_layers=conv_layers)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = conv2d(pool1, filters * 2, conv_layers=conv_layers)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = conv2d(pool2, filters * 4, conv_layers=conv_layers)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = conv2d(pool3, filters * 8, conv_layers=conv_layers)
    pool4 = MaxPooling2D((2, 2))(conv4)

    conv5 = conv2d(pool4, filters * 16, conv_layers=conv_layers)

    up6 = deconv2d(conv5, filters * 8)
    conv6 = attention_block(up6, conv4, filters * 8)
    up6 = Concatenate()([up6, conv6])
    conv6 = conv2d(up6, filters * 8, conv_layers=conv_layers)

    up7 = deconv2d(conv6, filters * 4)
    conv7 = attention_block(up7, conv3, filters * 4)
    up7 = Concatenate()([up7, conv7])
    conv7 = conv2d(up7, filters * 4, conv_layers=conv_layers)

    up8 = deconv2d(conv7, filters * 2)
    conv8 = attention_block(up8, conv2, filters * 2)
    up8 = Concatenate()([up8, conv8])
    conv8 = conv2d(up8, filters * 2, conv_layers=conv_layers)

    up9 = deconv2d(conv8, filters)
    conv9 = attention_block(up9, conv1, filters)
    up9 = Concatenate()([up9, conv9])
    conv9 = conv2d(up9, filters, conv_layers=conv_layers)

    outputs = Conv2D(output_channels, kernel_size=(1, 1), strides=(1, 1), activation='softmax')(conv9)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def base_unet(filters, output_channels, width=None, height=None, input_channels=1, conv_layers=2):
    def conv2d(layer_input, filters, conv_layers=2):
        d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(layer_input)
        d = BatchNormalization()(d)
        d = Activation('relu')(d)

        for i in range(conv_layers - 1):
            d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(d)
            d = BatchNormalization()(d)
            d = Activation('relu')(d)

        return d

    def deconv2d(layer_input, filters):
        u = Conv2DTranspose(filters, 2, strides=(2, 2), padding='same')(layer_input)
        u = BatchNormalization()(u)
        u = Activation('relu')(u)
        return u

    inputs = Input(shape=(width, height, input_channels))

    conv1 = conv2d(inputs, filters, conv_layers=conv_layers)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = conv2d(pool1, filters * 2, conv_layers=conv_layers)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = conv2d(pool2, filters * 4, conv_layers=conv_layers)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = conv2d(pool3, filters * 8, conv_layers=conv_layers)
    pool4 = MaxPooling2D((2, 2))(conv4)

    conv5 = conv2d(pool4, filters * 16, conv_layers=conv_layers)

    up6 = deconv2d(conv5, filters * 8)
    up6 = Concatenate()([up6, conv4])
    conv6 = conv2d(up6, filters * 8, conv_layers=conv_layers)

    up7 = deconv2d(conv6, filters * 4)
    up7 = Concatenate()([up7, conv3])
    conv7 = conv2d(up7, filters * 4, conv_layers=conv_layers)

    up8 = deconv2d(conv7, filters * 2)
    up8 = Concatenate()([up8, conv2])
    conv8 = conv2d(up8, filters * 2, conv_layers=conv_layers)

    up9 = deconv2d(conv8, filters)
    up9 = Concatenate()([up9, conv1])
    conv9 = conv2d(up9, filters, conv_layers=conv_layers)

    # Changed sigmoid to softmax, also changed output from 1 to 4
    outputs = Conv2D(output_channels, kernel_size=(1, 1), strides=(1, 1), activation='softmax')(conv9)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def dense_unet(filters, output_channels, width=None, height=None, input_channels=1, conv_layers=2):
    def conv2d(layer_input, filters, conv_layers=2):
        concats = []

        d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(layer_input)
        d = BatchNormalization()(d)
        d = Activation('relu')(d)

        concats.append(d)
        M = d

        for i in range(conv_layers - 1):
            d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(M)
            d = BatchNormalization()(d)
            d = Activation('relu')(d)

            concats.append(d)
            M = concatenate(concats)

        return M

    def deconv2d(layer_input, filters):
        u = Conv2DTranspose(filters, 2, strides=(2, 2), padding='same')(layer_input)
        u = BatchNormalization()(u)
        u = Activation('relu')(u)
        return u

    inputs = Input(shape=(width, height, input_channels))

    conv1 = conv2d(inputs, filters, conv_layers=conv_layers)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv2d(pool1, filters * 2, conv_layers=conv_layers)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv2d(pool2, filters * 4, conv_layers=conv_layers)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv2d(pool3, filters * 8, conv_layers=conv_layers)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv2d(pool4, filters * 16, conv_layers=conv_layers)

    up6 = deconv2d(conv5, filters * 8)
    merge6 = concatenate([conv4, up6])
    conv6 = conv2d(merge6, filters * 8, conv_layers=conv_layers)

    up7 = deconv2d(conv6, filters * 4)
    merge7 = concatenate([conv3, up7])
    conv7 = conv2d(merge7, filters * 4, conv_layers=conv_layers)

    up8 = deconv2d(conv7, filters * 2)
    merge8 = concatenate([conv2, up8])
    conv8 = conv2d(merge8, filters * 2, conv_layers=conv_layers)

    up9 = deconv2d(conv8, filters)
    merge9 = concatenate([conv1, up9])
    conv9 = conv2d(merge9, filters, conv_layers=conv_layers)

    outputs = Conv2D(output_channels, kernel_size=(1, 1), strides=(1, 1), activation='softmax')(conv9)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def inception_unet(filters, output_channels, width=None, height=None, input_channels=1):
    def InceptionModule(inputs, filters):
        tower0 = Conv2D(filters, (1, 1), padding='same')(inputs)
        tower0 = BatchNormalization()(tower0)
        tower0 = Activation('relu')(tower0)

        tower1 = Conv2D(filters, (1, 1), padding='same')(inputs)
        tower1 = BatchNormalization()(tower1)
        tower1 = Activation('relu')(tower1)
        tower1 = Conv2D(filters, (3, 3), padding='same')(tower1)
        tower1 = BatchNormalization()(tower1)
        tower1 = Activation('relu')(tower1)

        tower2 = Conv2D(filters, (1, 1), padding='same')(inputs)
        tower2 = BatchNormalization()(tower2)
        tower2 = Activation('relu')(tower2)
        tower2 = Conv2D(filters, (3, 3), padding='same')(tower2)
        tower2 = Conv2D(filters, (3, 3), padding='same')(tower2)
        tower2 = BatchNormalization()(tower2)
        tower2 = Activation('relu')(tower2)

        tower3 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(inputs)
        tower3 = Conv2D(filters, (1, 1), padding='same')(tower3)
        tower3 = BatchNormalization()(tower3)
        tower3 = Activation('relu')(tower3)

        inception_module = concatenate([tower0, tower1, tower2, tower3], axis=3)

        return inception_module

    def deconv2d(layer_input, filters):
        u = Conv2DTranspose(filters, 2, strides=(2, 2), padding='same')(layer_input)
        u = BatchNormalization()(u)
        u = Activation('relu')(u)

        return u

    inputs = Input(shape=(width, height, input_channels))

    conv1 = InceptionModule(inputs, filters)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = InceptionModule(pool1, filters * 2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = InceptionModule(pool2, filters * 4)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = InceptionModule(pool3, filters * 8)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = InceptionModule(pool4, filters * 16)

    up6 = deconv2d(conv5, filters * 8)
    up6 = InceptionModule(up6, filters * 8)
    merge6 = concatenate([conv4, up6], axis=3)

    up7 = deconv2d(merge6, filters * 4)
    up7 = InceptionModule(up7, filters * 4)
    merge7 = concatenate([conv3, up7], axis=3)

    up8 = deconv2d(merge7, filters * 2)
    up8 = InceptionModule(up8, filters * 2)
    merge8 = concatenate([conv2, up8], axis=3)

    up9 = deconv2d(merge8, filters)
    up9 = InceptionModule(up9, filters)
    merge9 = concatenate([conv1, up9], axis=3)

    outputs = Conv2D(output_channels, kernel_size=(1, 1), strides=(1, 1), activation='softmax')(merge9)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def r2_unet(filters, output_channels, width=None, height=None, input_channels=1, conv_layers=2, rr_layers=2):
    def recurrent_block(layer_input, filters, conv_layers=2, rr_layers=2):
        convs = []
        for i in range(conv_layers - 1):
            a = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')
            convs.append(a)

        d = layer_input
        for i in range(len(convs)):
            a = convs[i]
            d = a(d)
            d = BatchNormalization()(d)
            d = Activation('relu')(d)

        for j in range(rr_layers):
            d = Add()([d, layer_input])
            for i in range(len(convs)):
                a = convs[i]
                d = a(d)
                d = BatchNormalization()(d)
                d = Activation('relu')(d)

        return d

    def RRCNN_block(layer_input, filters, conv_layers=2, rr_layers=2):
        d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(layer_input)
        d1 = recurrent_block(d, filters, conv_layers=conv_layers, rr_layers=rr_layers)
        return Add()([d, d1])

    def deconv2d(layer_input, filters):
        u = Conv2DTranspose(filters, 2, strides=(2, 2), padding='same')(layer_input)
        u = BatchNormalization()(u)
        u = Activation('relu')(u)
        return u

    inputs = Input(shape=(width, height, input_channels))

    conv1 = RRCNN_block(inputs, filters, conv_layers=conv_layers, rr_layers=rr_layers)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = RRCNN_block(pool1, filters * 2, conv_layers=conv_layers, rr_layers=rr_layers)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = RRCNN_block(pool2, filters * 4, conv_layers=conv_layers, rr_layers=rr_layers)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = RRCNN_block(pool3, filters * 8, conv_layers=conv_layers, rr_layers=rr_layers)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = RRCNN_block(pool4, filters * 16, conv_layers=conv_layers, rr_layers=rr_layers)

    conv6 = deconv2d(conv5, filters * 8)
    up6 = concatenate([conv6, conv4])
    up6 = RRCNN_block(up6, filters * 8, conv_layers=conv_layers, rr_layers=rr_layers)

    conv7 = Conv2DTranspose(filters * 4, 3, strides=(2, 2), padding='same')(up6)
    up7 = concatenate([conv7, conv3])
    up7 = RRCNN_block(up7, filters * 4, conv_layers=conv_layers, rr_layers=rr_layers)

    conv8 = Conv2DTranspose(filters * 2, 3, strides=(2, 2), padding='same')(up7)
    up8 = concatenate([conv8, conv2])
    up8 = RRCNN_block(up8, filters * 2, conv_layers=conv_layers, rr_layers=rr_layers)

    conv9 = Conv2DTranspose(filters, 3, strides=(2, 2), padding='same')(up8)
    up9 = concatenate([conv9, conv1])
    up9 = RRCNN_block(up9, filters, conv_layers=conv_layers, rr_layers=rr_layers)

    output_layer_noActi = Conv2D(output_channels, (1, 1), padding="same", activation=None)(up9)
    outputs = Activation('softmax')(output_layer_noActi)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def residual_unet(filters, output_channels, width=None, height=None, input_channels=1, conv_layers=2):
    def residual_block(x, filters, conv_layers=2):
        x = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(x)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        d = x
        for i in range(conv_layers - 1):
            d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(d)
            d = BatchNormalization()(d)
            d = Activation('relu')(d)

        x = Add()([d, x])

        return x

    def deconv2d(layer_input, filters):
        u = Conv2DTranspose(filters, 2, strides=(2, 2), padding='same')(layer_input)
        u = BatchNormalization()(u)
        u = Activation('relu')(u)
        return u

    inputs = Input(shape=(width, height, input_channels))

    conv1 = residual_block(inputs, filters, conv_layers=conv_layers)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = residual_block(pool1, filters * 2, conv_layers=conv_layers)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = residual_block(pool2, filters * 4, conv_layers=conv_layers)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = residual_block(pool3, filters * 8, conv_layers=conv_layers)
    pool4 = MaxPooling2D((2, 2))(conv4)

    conv5 = residual_block(pool4, filters * 16, conv_layers=conv_layers)

    conv6 = deconv2d(conv5, filters * 8)
    up6 = concatenate([conv6, conv4])
    up6 = residual_block(up6, filters * 8, conv_layers=conv_layers)

    conv7 = deconv2d(up6, filters * 4)
    up7 = concatenate([conv7, conv3])
    up7 = residual_block(up7, filters * 4, conv_layers=conv_layers)

    conv8 = deconv2d(up7, filters * 2)
    up8 = concatenate([conv8, conv2])
    up8 = residual_block(up8, filters * 2, conv_layers=conv_layers)

    conv9 = deconv2d(up8, filters)
    up9 = concatenate([conv9, conv1])
    up9 = residual_block(up9, filters, conv_layers=conv_layers)

    output_layer_noActi = Conv2D(output_channels, (1, 1), padding="same", activation=None)(up9)
    outputs = Activation('softmax')(output_layer_noActi)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def se_unet(filters, output_channels, width=None, height=None, input_channels=1, conv_layers=2):
    def conv2d(layer_input, filters, conv_layers=2):
        d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(layer_input)
        d = BatchNormalization()(d)
        d = Activation('relu')(d)

        for i in range(conv_layers - 1):
            d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(d)
            d = BatchNormalization()(d)
            d = Activation('relu')(d)

        return d

    def deconv2d(layer_input, filters):
        u = Conv2DTranspose(filters, 2, strides=(2, 2), padding='same')(layer_input)
        u = BatchNormalization()(u)
        u = Activation('relu')(u)
        return u

    def cse_block(inp, ratio=2):
        init = inp
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        filters = init.shape[channel_axis]
        se_shape = (1, 1, filters)

        se = GlobalAveragePooling2D()(init)
        se = Reshape(se_shape)(se)
        se = Dense(filters // ratio, activation='relu', use_bias=False)(se)
        se = Dense(filters, activation='sigmoid', use_bias=False)(se)

        if K.image_data_format() == 'channels_first':
            se = Permute((3, 1, 2))(se)

        x = multiply([init, se])
        return x

    def sse_block(inp):
        x = Conv2D(1, kernel_size=(1, 1), activation='sigmoid', use_bias=False)(inp)
        x = multiply([inp, x])

        return x

    def scse_block(inp, ratio=2):
        x1 = cse_block(inp, ratio)
        x2 = sse_block(inp)

        x = maximum([x1, x2])

        return x

    inputs = Input(shape=(width, height, input_channels))

    conv1 = conv2d(inputs, filters, conv_layers=conv_layers)
    conv1 = scse_block(conv1)
    pool1 = MaxPooling2D((2, 2))(conv1)

    conv2 = conv2d(pool1, filters * 2, conv_layers=conv_layers)
    conv2 = scse_block(conv2)
    pool2 = MaxPooling2D((2, 2))(conv2)

    conv3 = conv2d(pool2, filters * 4, conv_layers=conv_layers)
    conv3 = scse_block(conv3)
    pool3 = MaxPooling2D((2, 2))(conv3)

    conv4 = conv2d(pool3, filters * 8, conv_layers=conv_layers)
    conv4 = scse_block(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)

    conv5 = conv2d(pool4, filters * 16, conv_layers=conv_layers)
    conv5 = scse_block(conv5)

    up6 = deconv2d(conv5, filters * 8)
    up6 = Concatenate()([up6, conv4])
    conv6 = conv2d(up6, filters * 8, conv_layers=conv_layers)
    conv6 = scse_block(conv6)

    up7 = deconv2d(conv6, filters * 4)
    up7 = Concatenate()([up7, conv3])
    conv7 = conv2d(up7, filters * 4, conv_layers=conv_layers)
    conv7 = scse_block(conv7)

    up8 = deconv2d(conv7, filters * 2)
    up8 = Concatenate()([up8, conv2])
    conv8 = conv2d(up8, filters * 2, conv_layers=conv_layers)
    conv8 = scse_block(conv8)

    up9 = deconv2d(conv8, filters)
    up9 = Concatenate()([up9, conv1])
    conv9 = conv2d(up9, filters, conv_layers=conv_layers)
    conv9 = scse_block(conv9)

    outputs = Conv2D(output_channels, kernel_size=(1, 1), strides=(1, 1), activation='softmax')(conv9)

    model = Model(inputs=inputs, outputs=outputs)

    return model


def unetpp(filters, output_channels, width=None, height=None, input_channels=1, conv_layers=2):
    def conv2d(layer_input, filters, conv_layers=2):
        d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(layer_input)
        d = BatchNormalization()(d)
        d = Activation('relu')(d)

        for i in range(conv_layers - 1):
            d = Conv2D(filters, kernel_size=(3, 3), strides=(1, 1), padding='same')(d)
            d = BatchNormalization()(d)
            d = Activation('relu')(d)

        return d

    def deconv2d(layer_input, filters):
        u = Conv2DTranspose(filters, 2, strides=(2, 2), padding='same')(layer_input)
        u = BatchNormalization()(u)
        u = Activation('relu')(u)
        return u

    inputs = Input(shape=(width, height, input_channels))

    conv00 = conv2d(inputs, filters, conv_layers=conv_layers)
    pool0 = MaxPooling2D((2, 2))(conv00)

    conv10 = conv2d(pool0, filters * 2, conv_layers=conv_layers)
    pool1 = MaxPooling2D((2, 2))(conv10)

    conv01 = deconv2d(conv10, filters)
    conv01 = concatenate([conv00, conv01])
    conv01 = conv2d(conv01, filters, conv_layers=conv_layers)

    conv20 = conv2d(pool1, filters * 4, conv_layers=conv_layers)
    pool2 = MaxPooling2D((2, 2))(conv20)

    conv11 = deconv2d(conv20, filters)
    conv11 = concatenate([conv10, conv11])
    conv11 = conv2d(conv11, filters, conv_layers=conv_layers)

    conv02 = deconv2d(conv11, filters)
    conv02 = concatenate([conv00, conv01, conv02])
    conv02 = conv2d(conv02, filters, conv_layers=conv_layers)

    conv30 = conv2d(pool2, filters * 8, conv_layers=conv_layers)
    pool3 = MaxPooling2D((2, 2))(conv30)

    conv21 = deconv2d(conv30, filters)
    conv21 = concatenate([conv20, conv21])
    conv21 = conv2d(conv21, filters, conv_layers=conv_layers)

    conv12 = deconv2d(conv21, filters)
    conv12 = concatenate([conv10, conv11, conv12])
    conv12 = conv2d(conv12, filters, conv_layers=conv_layers)

    conv03 = deconv2d(conv12, filters)
    conv03 = concatenate([conv00, conv01, conv02, conv03])
    conv03 = conv2d(conv03, filters, conv_layers=conv_layers)

    conv40 = conv2d(pool3, filters * 16)

    conv31 = deconv2d(conv40, filters * 8)
    conv31 = concatenate([conv31, conv30])
    conv31 = conv2d(conv31, 8 * filters, conv_layers=conv_layers)

    conv22 = deconv2d(conv31, filters* 4)
    conv22 = concatenate([conv22, conv20, conv21])
    conv22 = conv2d(conv22, 4 * filters, conv_layers=conv_layers)

    conv13 = deconv2d(conv22, filters * 2)
    conv13 = concatenate([conv13, conv10, conv11, conv12])
    conv13 = conv2d(conv13, 2 * filters, conv_layers=conv_layers)

    conv04 = deconv2d(conv13, filters)
    conv04 = concatenate([conv04, conv00, conv01, conv02, conv03], axis=3)
    conv04 = conv2d(conv04, filters, conv_layers=conv_layers)

    outputs = Conv2D(output_channels, kernel_size=(1, 1), strides=(1, 1), activation='softmax')(conv04)

    model = Model(inputs=inputs, outputs=outputs)

    return model
