import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split



df = pd.read_pickle("proper_dataset/audio_data.csv")

x = df['fitur'].values
x = np.concatenate(x, axis=0).reshape(len(x), 40)

y = np.array(df["label_kelas"].tolist())
y = tf.keras.utils.to_categorical(y)

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(512, activation = 'relu', input_shape=x_train[0].shape, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(512, activation = 'relu',  kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(2, activation = 'sigmoid'),
])

print(model.summary())
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0001, decay_steps=10000, decay_rate=0.9)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(
    optimizer = optimizer, loss='binary_crossentropy', metrics = ['accuracy']
)

history = model.fit(x_train,y_train, epochs=1000)
model.save("saved_model/WakeWord.h5")
print(model.evaluate(x_test, y_test))

y_pred = np.argmax(model.predict(x_test), axis=1)
print(classification_report(np.argmax(y_test, axis=1), y_pred))
print(y_pred)