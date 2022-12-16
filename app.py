import os

from PIL import Image
from keras_preprocessing.image import ImageDataGenerator, load_img, img_to_array
import numpy as np
from mesonet_classifiers import *
import streamlit as st

MesoNet_classifier = Meso4()
MesoNet_classifier.load("mesonet_weights/Meso4_DF.h5")

num_to_label = {1: 'Real Photo', 0: 'Deepfake Photo'}

def processed_img(img_path):
    img=load_img(img_path,target_size=(256,256,3))
    img=img_to_array(img)
    img=img/255
    img=np.expand_dims(img,[0])
    answer=MesoNet_classifier.predict(img)
    y_class = [num_to_label[round(x[0])] for x in answer]
    print(y_class)
    y = " ".join(str(x) for x in y_class)
    #y = int(float(y))
    res = y

    print(answer)
    print(res)
    return res

def run():
    st.title("Deepfake Recognition Demo")

    st.markdown("Deep Learning is a powerful technique that is used in natural language processing,computer vision, image processing, and machine vision. "
            "Deep fakes employs deep learning techniques to synthesize and manipulate images of people so that humans cannot tell the difference between the real and the fake.")

    image = Image.open('deepfake.png')
    st.image(image, caption='Deepfake Recognition Model')

    st.subheader("This Deepfake Recognition Model Detects When an Image is Produced by Deepfake")
    st.text("Kindly upload an image in jpg or png file format")
    img_file = st.file_uploader("Choose an Image:", type=["jpg", "png"])

    if img_file is not None:
        st.image(img_file,use_column_width=False)

        if st.button("Predict"):
            result = processed_img()
            st.info("DeepFake Recognition: "+result)

    # Add sample image to predict
    st.markdown("***")
    sample_images = [("mesonet_test_images/df00204.jpg", "Sample Image 1"), ("mesonet_test_images/df01254.jpg", "Sample Image 2"), ("mesonet_test_images/real00240.jpg", "Sample Image 3"), ("mesonet_test_images/real00390.jpg", "Sample Image 4")]
    selected_image = st.selectbox("Select image to predict", options=sample_images)
    selected_image_file = selected_image[0]
    st.image(selected_image_file, use_column_width=False)

    if st.button("Predict"):
        result = processed_img(selected_image_file)
        st.info("DeepFake Recognition: "+result)

run()
