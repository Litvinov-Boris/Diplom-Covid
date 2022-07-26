from typing import Tuple
from warnings import filters

from tensorflow.keras import backend as K
from tensorflow.keras.layers import (
    BatchNormalization,
    concatenate,
    Conv2D,
    Conv2DTranspose,
    Dropout,
    Input,
    MaxPooling2D,
    Dense,
    GlobalAveragePooling2D,
    Multiply,
    Activation,
    Add,
)
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam


class SegmentationModel:
    _model: Model

    def __init__(self, input_shape):
        self._model = self._get_model(input_shape)

    def compile(self):
        self._model.compile(
                optimizer=Adam(
                        learning_rate=5e-5,
                        beta_1=0.9,
                        beta_2=0.99,
                ),
                loss=self.bce_dice_loss,
                metrics=[self.dice],
        )

    def fit(self, X, Y, batch_size, epochs, validation_data, callbacks, verbose=0):
        return self._model.fit(
                x=X,
                y=Y,
                batch_size=batch_size,
                epochs=epochs,
                verbose=verbose,
                validation_data=validation_data,
                callbacks=callbacks,
        )

    def load_weights(self, file_path):
        self._model.load_weights(filepath=file_path)

    def predict(self, X):
        return self._model.predict(X)

    @staticmethod
    def _get_model(input_shape: Tuple[int, int, int]) -> Model:
        inp = Input(input_shape)
        #coder
        x = Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer="he_normal")(inp)
        res_dil_32 =  SegmentationModel.res_dil(x, 32)

        x = Conv2D(64, (3,3), strides=2, activation='relu', padding='same', kernel_initializer="he_normal")(res_dil_32)
        res_dil_64 =  SegmentationModel.res_dil(x, 64)

        x = Conv2D(128, (3,3), strides=2, activation='relu', padding='same', kernel_initializer="he_normal")(res_dil_64)
        res_dil_128 =  SegmentationModel.res_dil(x, 128)

        x = Conv2D(256, (3,3), strides=2, activation='relu', padding='same', kernel_initializer="he_normal")(res_dil_128)
        res_dil_256 =  SegmentationModel.res_dil(x, 256)

        x = Conv2D(512, (3,3), strides=2, activation='relu', padding='same', kernel_initializer="he_normal")(res_dil_256)
        res_dil_512 =  SegmentationModel.res_dil(x, 512)

        #decoder
        x = Conv2DTranspose(256, (3, 3), strides=(2, 2), padding='same')(res_dil_512)
        res_dil_256 = SegmentationModel.at_mech(res_dil_256, 256)
        x = concatenate([x, res_dil_256])
        x = SegmentationModel.at_mech(x, 256*2)
        x = Conv2D(256, (3,3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
        res_dil_256 = SegmentationModel.res_dil(x, 256)
        deep_sup_256 = Conv2D(256, (1, 1))(res_dil_256)

        x = Conv2DTranspose(128, (3, 3), strides=(2, 2), padding='same')(res_dil_256)
        res_dil_128 = SegmentationModel.at_mech(res_dil_128, 128)
        x = concatenate([x, res_dil_128])
        x = SegmentationModel.at_mech(x, 128*2)
        x = Conv2D(128, (3,3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
        res_dil_128 = SegmentationModel.res_dil(x, 128)
        deep_sup_128 = Conv2D(128, (1, 1))(res_dil_128)
        deep_sup_256 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(deep_sup_256)
        deep_sup_128 = Add()([deep_sup_128, deep_sup_256])

        x = Conv2DTranspose(64, (3, 3), strides=(2, 2), padding='same')(res_dil_128)
        res_dil_64 = SegmentationModel.at_mech(res_dil_64, 64)
        x = concatenate([x, res_dil_64])
        x = SegmentationModel.at_mech(x, 64*2)
        x = Conv2D(64, (3,3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
        res_dil_64 = SegmentationModel.res_dil(x, 64)
        deep_sup_64 = Conv2D(64, (1, 1))(res_dil_64)
        deep_sup_128 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(deep_sup_128)
        deep_sup_64 = Add()([deep_sup_64, deep_sup_128])

        x = Conv2DTranspose(32, (3, 3), strides=(2, 2), padding='same')(res_dil_64)
        res_dil_32 = SegmentationModel.at_mech(res_dil_32, 32)
        x = concatenate([x, res_dil_32])
        x = SegmentationModel.at_mech(x, 32*2)
        x = Conv2D(32, (3,3), activation='relu', padding='same', kernel_initializer="he_normal")(x)
        res_dil_32 = SegmentationModel.res_dil(x, 32)
        deep_sup_32 = Conv2D(32, (1, 1))(res_dil_32)
        deep_sup_64 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(deep_sup_64)
        deep_sup_32 = Add()([deep_sup_32, deep_sup_64])

        out = Conv2D(1, (1, 1), activation='sigmoid', name='segmentation')(deep_sup_32)

        return Model(inputs=inp, outputs=out)

    @staticmethod
    def res_dil(inp, filters):
        x = Conv2D(filters, (3, 3), dilation_rate=2, activation='relu', padding='same', kernel_initializer="he_normal")(inp)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        x = Conv2D(filters, (3,3), dilation_rate= 4, activation='relu', padding='same', kernel_initializer="he_normal")(x)
        x = BatchNormalization()(x)
        x = Add()([inp, x])
        return x

    @staticmethod
    def at_mech(inp, filters):
        #chanal attention
        x1 = GlobalAveragePooling2D()(inp)
        x1 = Dense(filters/2, input_shape=(filters,), activation='relu', kernel_initializer="he_normal")(x1)
        x1 = Dense(filters, input_shape=(filters/2,),activation='sigmoid')(x1)
        x1 = Multiply()([inp, x1])
        #space attention
        x2 = Conv2D(filters, (1,1), activation='sigmoid')(inp)
        x2 = Multiply()([inp, x2])
        #result
        x = Add()([x1,x2])
        return x

    @staticmethod
    def dice(y_true, y_pred, smooth=1):
        intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
        union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])

        return K.mean((2. * intersection + smooth) / (union + smooth), axis=0)

    @staticmethod
    def dice_loss(y_true, y_pred):
        return 1.0 - SegmentationModel.dice(y_true, y_pred)

    @staticmethod
    def bce_dice_loss(y_true, y_pred):
        return 0.5 * binary_crossentropy(y_true, y_pred) + 0.5 * SegmentationModel.dice_loss(y_true, y_pred)
