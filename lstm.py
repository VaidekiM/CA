# -*- coding: utf-8 -*-
"""LSTM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/14ZOEhdzpTk2sex1pc0mR_hQcbJBUIXZ4
"""

#!pip install ann_visualizer

# Importing required libraries
import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, f1_score
# from keras.utils import plot_model


# Load data from CSV file
data = pd.read_csv('OutputMobile_All_116.csv')
data = data.loc[data['User_ID'] <= 85]
data = data.drop(['User_ID'], axis=1)
data = data.drop(['Dept'], axis=1)
data = data.drop(['Key'], axis=1)
data = data.drop(['Digraph'], axis=1)

col = ['Key_Mean', 'Key_STD', 'K_PP_Time', 'K_RR_Time', 'K_PR_Time', 'K_RP_Time', 'K_PP_STD', 'K_RR_STD', 'K_PR_STD', 'K_RP_STD', 'Typing_Speed', 'Error_Frequency']

for x in col:
    q75, q25 = np.percentile(data.loc[:, x], [75, 25])
    intr_qr = q75 - q25

    max = q75 + (1.5 * intr_qr)
    min = q25 - (1.5 * intr_qr)

    data.loc[data[x] < min, x] = np.nan
    data.loc[data[x] > max, x] = np.nan

data = data.dropna(axis=0)
#data = data.sample(n = 30, replace = False)
# Separate the input features and target variable
X = data.drop(['L1_Gender'], axis=1)
y = data['L1_Gender']

smote = SMOTE(random_state=42)
X, y = smote.fit_resample(X, y)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the input features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Reshape the input features for LSTM
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(1, X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on test set
loss, accuracy = model.evaluate(X_test, y_test)

# Make predictions on test set
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)


plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training and validation loss per epoch
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Evaluate the model on the test set
y_pred_prob = model.predict(X_test)
y_pred = (y_pred_prob > 0.5).astype(int)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

if conf_matrix.shape[1] == 2:
    far = conf_matrix[0, 1] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if conf_matrix[0, 0] + conf_matrix[0, 1] != 0 else 0  # False Acceptance Rate (FAR)
else:
    far = 0

frr = conf_matrix[1, 0] / (conf_matrix[1, 0] + conf_matrix[1, 1])  # False Rejection Rate (FRR)

print('F1 score:', f1)
print('FAR:', far)
print('FRR:', frr)

loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)
# Print accuracy score
mean_accuracy = np.mean(accuracy)
mean_f1 = np.mean(f1)
mean_far = np.mean(far)
mean_frr = np.mean(frr)
std_accuracy = np.std(accuracy)
std_f1 = np.std(mean_f1)
std_far = np.std(mean_far)
std_frr = np.std(mean_frr)

print('Mean accuracy:', mean_accuracy)
print('Mean F1:', mean_f1)
print('Mean FAR:', mean_far)
print('Mean FRR:', mean_frr)
print('Standard deviation of accuracy:', std_accuracy)
print('Standard deviation of F1:', std_f1)
print('Standard deviation of FAR:', std_far)
print('Standard deviation of FRR:', std_frr)