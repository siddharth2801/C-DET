import tensorflow as tf
from sklearn.metrics import classification_report
from preprocess import preprocess_data

# Load preprocessed data
X_train, X_val, X_test, y_train, y_val, y_test, mlb = preprocess_data('./data/DatasetA1.xlsx')

# Load the trained model
model = tf.keras.models.load_model('./models/model.h5')

# Predict on test set
y_pred = model.predict(X_test)
y_pred_binary = (y_pred > 0.5).astype(int)

# Evaluate the model
print("Classification Report:")
print(classification_report(y_test, y_pred_binary, target_names=mlb.classes_))
