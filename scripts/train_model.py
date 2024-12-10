import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.utils.class_weight import compute_class_weight
from imblearn.over_sampling import RandomOverSampler
from preprocess import preprocess_data
import numpy as np

# Load preprocessed data
X_train, X_val, X_test, y_train, y_val, y_test, mlb = preprocess_data('./data/DatasetA1.xlsx')

# Function to apply RandomOverSampler for multi-label data
def apply_random_oversampling(X, y):
    ros = RandomOverSampler(random_state=42)
    
    # Flatten the multi-label target for consistent oversampling
    y_flat = ["".join(map(str, row)) for row in y]
    X_resampled, y_resampled_flat = ros.fit_resample(X, y_flat)
    
    # Reconstruct the multi-label target
    y_resampled = np.array([list(map(int, label)) for label in y_resampled_flat])
    return X_resampled, y_resampled

# Apply RandomOverSampler
X_train, y_train = apply_random_oversampling(X_train, y_train)

# Compute class weights
y_train_flat = y_train.argmax(axis=1)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.array(range(y_train.shape[1])),  # Convert range to numpy array
    y=y_train_flat
)
class_weights_dict = {i: weight for i, weight in enumerate(class_weights)}

# Define the Neural Network Model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.2),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(y_train.shape[1], activation='sigmoid')  # Multi-label output
])

# Compile the Model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train the Model
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=20,
    batch_size=16,
    verbose=1,
    class_weight=class_weights_dict  # Apply class weights
)

# Save the Trained Model
model.save('./models/model.h5')
print("Model saved at ./models/model.h5")
