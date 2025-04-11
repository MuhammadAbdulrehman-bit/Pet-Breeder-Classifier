import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import Model, layers
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input


(train_ds, test_ds), ds_info = tfds.load(
    'oxford_iiit_pet',
     split=["train[:80%]", "train[80%:]"],
     shuffle_files=True,
    as_supervised=True,
     with_info=True
)

def preprocess(image, label):
    image = tf.image.resize( image,(224, 224))
    image = tf.cast(image, tf.float32)             # still 0–255
    image = preprocess_input(image)
    label = label                # now –1→+1
    return image, label

train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
test_ds  = test_ds .map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

batch_size = 32
train_ds = train_ds.shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_ds = test_ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

base_model.trainable = False

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip('horizontal'),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.2)
])

inputs = tf.keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)
x = base_model(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
outputs = layers.Dense(37, activation="softmax")(x)
model = Model(inputs, outputs)

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])
model.fit(train_ds, epochs=5, validation_data=test_ds)

model.save("pet_breed_classifier.keras")