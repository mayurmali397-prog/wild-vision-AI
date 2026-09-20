import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# List of 90 classes
class_names = [
    'antelope', 'badger', 'bat', 'bear', 'bee', 'beetle', 'bison', 'boar', 'butterfly', 'cat',
    'caterpillar', 'chimpanzee', 'cockroach', 'cow', 'coyote', 'crab', 'crow', 'deer', 'dog', 'dolphin',
    'donkey', 'dragonfly', 'duck', 'eagle', 'elephant', 'flamingo', 'fly', 'fox', 'goat', 'goldfish',
    'goose', 'gorilla', 'grasshopper', 'hamster', 'hare', 'hedgehog', 'hippopotamus', 'hornet', 'horse', 'hyena',
    'jellyfish', 'kangaroo', 'koala', 'ladybugs', 'leopard', 'lion', 'lizard', 'lobster', 'mosquito', 'moth',
    'mouse', 'octopus', 'okapi', 'orangutan', 'otter', 'owl', 'ox', 'oyster', 'panda', 'parrot',
    'pelecanus', 'penguin', 'pig', 'pigeon', 'porcupine', 'possum', 'raccoon', 'rat', 'reindeer', 'rhinoceros',
    'sandpiper', 'seahorse', 'seal', 'shark', 'sheep', 'snake', 'sparrow', 'squid', 'squirrel', 'starfish',
    'swan', 'tiger', 'turkey', 'turtle', 'whale', 'wolf', 'wombat', 'woodpecker', 'zebra'
]

@st.cache_resource
def load_model():
    return tf.keras.models.load_model("animal_classifier_model.keras")

model = load_model()

st.title("🐾 WildVision AI: Animal Species Classifier")
st.write("Upload an animal photo to test our deep learning model (**Xception** with 93.7% accuracy).")

uploaded_file = st.file_uploader("Choose an animal image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess
    img = image.resize((299, 299))
    img_array = np.array(img)
    if img_array.shape[-1] == 4: # Handle RGBA
        img_array = img_array[..., :3]
    img_array = tf.expand_dims(img_array, 0)

    # Predict
    predictions = model.predict(img_array)[0]
    predicted_idx = np.argmax(predictions)
    confidence = float(predictions[predicted_idx]) * 100

    st.success(f"**Prediction:** {class_names[predicted_idx].capitalize()} ({confidence:.2f}% confidence)")

    # Show top 3
    st.subheader("Top 3 Predictions:")
    top_3_idx = np.argsort(predictions)[-3:][::-1]
    for idx in top_3_idx:
        st.write(f"- {class_names[idx].capitalize()}: {float(predictions[idx])*100:.2f}%")
