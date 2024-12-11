import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam

# Assuming the input data shape is (n_samples, 4) for the differences and (n_samples, 4*4) for the concatenated vector
input_shape = (20,) # 4 differences + 16 concatenated vector

# Define the model
model = Sequential([
    Dense(64, input_shape=input_shape, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Assuming binary classification
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001),
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Print the model summary
model.summary()

# Assuming you have your data loaded in X_train, y_train, X_test, y_test
# X_train, X_test should be of shape (n_samples, 20)
# y_train, y_test should be of shape (n_samples,)

# Example training
# model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))