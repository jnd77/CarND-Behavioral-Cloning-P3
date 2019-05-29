import keras

from keras.applications.inception_v3 import InceptionV3


def build_model(architecture):
    """
    Build the model
    :return: keras.Model
    """
    # Create the input
    raw_input = keras.layers.Input(shape=(160, 320, 3))

    # Crops the input
    cropped_input = keras.layers.Cropping2D(cropping=((50, 20), (0, 0)))(raw_input)

    # Preprocess the input
    preprocessed_input = keras.layers.Lambda(lambda img: img / 127.5 - 1.)(cropped_input)

    # Apply the model
    if architecture == 'inception':
        output = _build_inception(preprocessed_input)
    elif architecture == 'nvidia':
        output = _build_nvidia(preprocessed_input)
    else:
        raise ValueError(f'Unknown architecture {architecture}')

    # Finally build the model
    return keras.models.Model(inputs=raw_input, outputs=output)


def _build_inception(preprocessed_input):
    # Load the model without the top
    inception = InceptionV3(weights='imagenet', include_top=False,
                            input_shape=(90, 320, 3))

    # Freeze the weights
    for layer in inception.layers:
        layer.trainable = False

    # Apply the inception model
    inception_output = inception(preprocessed_input)

    # Add the fully-connected layers at the end
    x = keras.layers.GlobalAveragePooling2D()(inception_output)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(512, activation='elu')(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(128, activation='elu')(x)
    return keras.layers.Dense(1)(x)


def _build_nvidia(preprocessed_input):
    activation = 'relu'

    # The conv layers
    x = keras.layers.Conv2D(24, 5, strides=(2, 2), padding='valid', activation=activation)(preprocessed_input)
    x = keras.layers.Conv2D(36, 5, strides=(2, 2), padding='valid', activation=activation)(x)
    x = keras.layers.Conv2D(48, 5, strides=(2, 2), padding='valid', activation=activation)(x)
    x = keras.layers.Conv2D(64, 3, strides=(1, 1), padding='valid', activation=activation)(x)
    x = keras.layers.Conv2D(64, 3, strides=(1, 1), padding='valid', activation=activation)(x)

    # The fully-connected layers
    x = keras.layers.Flatten()(x)
    x = keras.layers.Dense(100, activation=activation)(x)
    x = keras.layers.Dense(50, activation=activation)(x)
    x = keras.layers. Dense(10, activation=activation)(x)

    return keras.layers.Dense(1)(x)
