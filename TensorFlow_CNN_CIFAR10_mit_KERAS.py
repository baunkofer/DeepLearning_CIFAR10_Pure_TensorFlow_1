#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 20:30:45 2025

@author: aunkofer
"""


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt

# Eager Mode aktivieren (WICHTIG für model.fit)
"""TensorFlow kann im "Eager Execution Mode" oder im "Graph Mode" arbeiten:

    Eager Mode: Standard in TensorFlow 2.x, erlaubt direktes Debugging und Python-typische Iterationen.
    Graph Mode: Optimierter Rechenmodus, aber restriktiver – z. B. kein direktes Iterieren über tf.data.Dataset.
"""
tf.config.run_functions_eagerly(True)

# CIFAR-10 laden
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Normalisieren: Werte von 0–255 auf 0–1 skalieren
x_train, x_test = x_train / 255.0, x_test / 255.0

# Klassenlabels sind 2D (z. B. [[6]]), wir machen sie 1D
y_train = y_train.flatten()
y_test = y_test.flatten()


model = models.Sequential([
    # 1. Convolutional Layer
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),

    # 2. Convolutional Layer
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),

    # 3. Convolutional Layer
    layers.Conv2D(64, (3, 3), activation='relu'),

    # Flatten + Dense Layer
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')  # 10 Klassen in CIFAR-10
])


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


history = model.fit(x_train, y_train, epochs=10, 
                    validation_data=(x_test, y_test))


# Plotten der Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.title('Training & Validation Accuracy')
plt.show()


test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.4f}")


"""
    Confusion Matrix
"""

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np

# 🔮 Vorhersagen berechnen
y_pred_probs = model.predict(x_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# ✅ Confusion Matrix berechnen
cm = confusion_matrix(y_test, y_pred)

# 🖼 Visualisierung
labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
          'dog', 'frog', 'horse', 'ship', 'truck']

disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
disp.plot(cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix – CIFAR-10")
plt.show()
