import os
import pickle
import numpy as np

from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model


# Dataset path
dataset_path = "dataset/Flickr8k_Dataset"


# Load pretrained ResNet50
base_model = ResNet50(weights="imagenet")


# Remove last classification layer
model = Model(
    inputs=base_model.inputs,
    outputs=base_model.layers[-2].output
)


features = {}


for img_name in os.listdir(dataset_path):

    img_path = os.path.join(dataset_path, img_name)

    img = image.load_img(
        img_path,
        target_size=(224,224)
    )

    img_array = image.img_to_array(img)

    img_array = np.expand_dims(
        img_array,
        axis=0
    )

    img_array = preprocess_input(img_array)

    feature = model.predict(
        img_array,
        verbose=0
    )

    features[img_name] = feature


pickle.dump(
    features,
    open("features.pkl","wb")
)

print("Feature Extraction Completed")