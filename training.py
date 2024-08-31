from tensorflow.keras.callbacks import (EarlyStopping, ModelCheckpoint,
                                        ReduceLROnPlateau)


def train_model(model, train_generator, validation_generator):
    # Callbacks
    checkpoint_cb = ModelCheckpoint('model.keras', save_best_only=True)
    earlystop_cb = EarlyStopping(patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6)
    
    # Train model
    history = model.fit(
        train_generator,
        steps_per_epoch=len(train_generator),
        epochs=100,
        validation_data=validation_generator,
        validation_steps=len(validation_generator),
        callbacks=[checkpoint_cb, earlystop_cb, reduce_lr]
    )
    
    return history
