from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.utils import plot_model


def create_model():
    pre_trained_model = MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )
    
    # Freeze all layers except the last block
    pre_trained_model.trainable = True
    set_trainable = False
    for layer in pre_trained_model.layers:
        if layer.name == 'block_16_expand':
            set_trainable = True
        layer.trainable = set_trainable
    
    # Add custom layers
    model = models.Sequential()
    model.add(pre_trained_model)
    model.add(layers.Flatten())
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(36, activation='softmax'))
    
    # Model summary
    model.summary()
    
    # Compile model
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model
