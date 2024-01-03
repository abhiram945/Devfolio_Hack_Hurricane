import cv2
import io
import numpy as np
import pyttsx3
import tempfile
import streamlit as st
from transformers import pipeline
from langchain import LLMChain, OpenAI
from langchain.prompts import PromptTemplate
from PIL import Image


def Speak(Text):
    engine = pyttsx3.init("sapi5")
    voices = engine.getProperty('voices')
    engine.setProperty('voices', voices[0].id)
    engine.setProperty('rate', 150)
    engine.say(Text)
    engine.runAndWait()

#image to text using Hugging Face pipeline
def img2text(img):
    image_to_text = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", max_new_tokens=100)
    text = image_to_text(img)
    return text[0]["generated_text"]

#convert image bytes to a PIL image object
def img2pil(img):
    img_pil = Image.open(io.BytesIO(img))
    return img_pil

#function to create Streamlit web application
def main():
    st.header("Turn Images into Audio Stories")

    uploaded_file = st.file_uploader("Choose an image..", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        with tempfile.NamedTemporaryFile(delete=False) as file:
            file.write(bytes_data)
            file_path = file.name

        st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        scene = img2text(file_path)

        with st.expander("Scene"):
            st.write(scene)

        speak = st.button("Speak")
        if speak:
            Speak(f"Scene: {scene}")


    else:
        img = st.camera_input(label="Take a Photo of the Scene")
        if img is not None:
            st.image(img, caption="Taken Photo", use_column_width=True)
            bytes_data = img.getvalue()
            with tempfile.NamedTemporaryFile(delete=False) as file:
                file.write(bytes_data)
                file_path = file.name

            scene = img2text(file_path)

            with st.expander("Scene"):
                st.write(scene)

            speak = st.button("Speak")
            if speak:
                Speak(f"Scene: {scene}")

if __name__ == "__main__":
    main()