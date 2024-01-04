import keras
from keras import Sequential
from keras.layers import Dense

def create_model():
    model = Sequential([
        Dense(10, activation='relu'),
        Dense(10, activation='relu'),
        Dense(1, activation='sigmoid'),
    ])
    opt = keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)
    model.compile(optimizer=opt, loss='mean_squared_error', metrics=['accuracy'])
    return model