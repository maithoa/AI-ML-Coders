from keras import optimizers, callbacks
import tensorflow as tf

def train_model(model, train_data, val_data, hyperparams):
    """
    Handles the compilation and fitting of the model.
    
    Args:
        model: The model instance from create_model().
        train_data: Training generator/iterator.
        val_data: Validation generator/iterator.
        hyperparams (dict): Dictionary of parameters (lr, epochs, etc.).
    """
    
    # Configure the learning process
    model.compile(
        loss='binary_crossentropy',
        optimizer=optimizers.RMSprop(learning_rate=hyperparams.get('learning_rate', 0.001)),
        metrics=['accuracy']
    )

    # Stop training if the validation loss doesn't improve for 3 epochs
    stop_early = callbacks.EarlyStopping(
        monitor='val_loss', 
        patience=5, 
        restore_best_weights=True
    )
 
    # Execute the training loop
    history = model.fit(
        x=train_data,
        steps_per_epoch=hyperparams.get('steps_per_epoch'),
        epochs=hyperparams.get('epochs', 15),
        validation_data=val_data,
        validation_steps=hyperparams.get('validation_steps'),
        callbacks=[stop_early],
        verbose=1
    )
    
    return history