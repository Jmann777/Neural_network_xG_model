"""
This module creates and runs a neural network to predict xG. Data is split 60% training, 20% validation and 20%
test and inputs are scaled. The neural network model is then created with 2 dense layers sized 10
followed by a ReLU activation and a final layer size 1 with sigmoid activation to compute the probabilities.
The model optimizes the brier score using the Adam optimizer with a learning rate of 0.001 default betas
We use as suggested early stopping with minimum delta 1e-5 and batch size 16.
However, we also use patience equal to 50 not to stop the first time when the validation loss is not changing.

The model is based off Javier Fernandez's model (see- https://link.springer.com/article/10.1007/s10994-021-05989-6)
and is taken from David Sumpter's soccermatics lesson 7.
"""


import joblib
import numpy as np

from keras import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def create_model() -> Sequential:
    """
    Create and compile a neural network model for xG prediction. The model has two hidden layers (10 neurons and ReLU)
    and an output layer (1 neuron and sigmoid).

    Returns:
    - Sequential: Compiled neural network model.
    """
    model = Sequential([
        Dense(10, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['accuracy'])
    return model


def setup_model(x: np.ndarray, y: np.ndarray) -> tuple:
    """
    Parameters:
    - x (np.ndarray): Array containing independent variables.
    - y (np.ndarray): Array containing the dependent variable.

    Returns:
    - Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        - x_train: Training set features.
        - x_val: Validation set features.
        - x_cal: Calibration set features.
        - y_train: Training set labels.
        - y_val: Validation set labels.
        - y_cal: Calibration set labels.
    """
    x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=0.6,
                                                        random_state=123, stratify=y)
    x_cal, x_val, y_cal, y_val = train_test_split(x_test, y_test, train_size=0.5,
                                                  random_state=123, stratify=y_test)
    # Scale the data to ensure equality of feature contribution
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_val = scaler.transform(x_val)
    x_cal = scaler.transform(x_cal)
    # Save fitted scaler for use in other file
    joblib.dump(scaler, '../fitted_scaler.joblib')
    return x_train, x_val, x_cal, y_train, y_val, y_cal


def run_model(x_train: np.ndarray, y_train: np.ndarray, x_val: np.ndarray, y_val: np.ndarray)\
        -> tuple[Sequential, np.ndarray]:
    """
    Trains the neural network model using the provided training and validation sets.

    Parameters:
    - X_train (np.ndarray): Training set features.
    - y_train (np.ndarray): Training set labels.
    - X_val (np.ndarray): Validation set features.
    - y_val (np.ndarray): Validation set labels.

    Returns:
    - Tuple[Sequential, History]:
        - Sequential: Trained neural network model.
        - History: Training history containing loss and accuracy metrics.
    """
    model = create_model()
    # early stopping object (callback)- https://www.tensorflow.org/api_docs/python/tf/keras/callbacks/EarlyStopping
    callback = EarlyStopping(min_delta=1e-5, patience=50, mode='min', monitor='val_loss', restore_best_weights=True)
    # Creating ModelCheckpoint callback
    checkpoint_filepath = '../best_model.h5'
    model_checkpoint = ModelCheckpoint(checkpoint_filepath, save_best_only=True, monitor='val_loss', mode='min')
    # Fit the model
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=1000,
                        verbose=1, batch_size=16, callbacks=[callback, model_checkpoint])
    return model, history
