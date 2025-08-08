from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.utils import to_categorical
from keras._tf_keras.keras.optimizers import Adam
from keras._tf_keras.keras.models import Sequential, Model
from keras._tf_keras.keras.layers import Dense, Flatten, Dropout, GlobalAveragePooling2D
from keras._tf_keras.keras.applications import InceptionV3
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import os

# 📁 Dataset
dataset_path = 'animal'
img_size = (299, 299)
batch_size = 32

# 🔄 Préparation des données
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_gen = datagen.flow_from_directory(dataset_path, target_size=img_size, batch_size=batch_size,
                                        class_mode='categorical', subset='training', shuffle=True)
val_gen = datagen.flow_from_directory(dataset_path, target_size=img_size, batch_size=batch_size,
                                      class_mode='categorical', subset='validation', shuffle=False)

num_classes = len(train_gen.class_indices)

# 🧠 Modèle
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=img_size + (3,))
base_model.trainable = False

x = GlobalAveragePooling2D()(base_model.output)
x = Dropout(0.3)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(base_model.input, output)
model.compile(optimizer=Adam(1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# 🚂 Entraînement
history = model.fit(train_gen, validation_data=val_gen, epochs=5)

# ✅ Évaluation
train_acc = model.evaluate(train_gen, verbose=0)[1]
val_acc = model.evaluate(val_gen, verbose=0)[1]
print(f"✅ Train Accuracy: {train_acc*100:.2f}%")
print(f"✅ Val Accuracy: {val_acc*100:.2f}%")

# 📊 Matrice de confusion
y_true = val_gen.classes
y_pred = np.argmax(model.predict(val_gen), axis=1)

cm = confusion_matrix(y_true, y_pred)
labels = list(train_gen.class_indices.keys())

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.title("Matrice de confusion")
plt.xlabel("Prédit")
plt.ylabel("Réel")
plt.show()

# 🔍 Prédiction visuelle
def predict_and_show(index):
    img = val_gen[index][0][0]
    true_label = np.argmax(val_gen[index][1][0])
    pred_label = np.argmax(model.predict(img.reshape(1, 299, 299, 3))[0])
    plt.imshow(img)
    plt.title(f"Réel: {true_label} - Prédit: {pred_label}")
    plt.axis('off')
    plt.show()

predict_and_show(5)

# 💾 Sauvegarde du modèle
model.save("animal_classifier_inceptionv3.h5")
