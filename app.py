import streamlit as st
import asyncio
import time
import pickle
import faiss
import numpy as np
import hashlib
from PIL import Image, UnidentifiedImageError
import google.generativeai as genai
from deep_translator import GoogleTranslator
from ultralytics import YOLO
from decouple import config
from gtts import gTTS
import tempfile
import os

# ✅ Set custom page configuration
st.set_page_config(
    page_title="AI Image Analysis",
    page_icon="🖼️",
    layout="wide",
)

# ✅ Load API Key Securely
api_key = config("GEMINI_API_KEY")
# api_key = st.secrets["GEMINI_API_KEY"]
genai.configure(api_key=api_key)

# ✅ Load YOLOv8 Model
yolo_model = YOLO("yolov8n.pt")

# ✅ Constants for FAISS and Cache Paths
FAISS_INDEX_PATH = "faiss_index.bin"
EMBEDDINGS_LIST_PATH = "embeddings_list.pkl"
IMAGE_DESCRIPTIONS_PATH = "image_descriptions.pkl"
EMBEDDING_DIMENSION = 768

# ✅ Initialize session state for caching
if "cached_description" not in st.session_state:
    st.session_state.cached_description = None
if "cached_response" not in st.session_state:
    st.session_state.cached_response = None
if "response_time" not in st.session_state:
    st.session_state.response_time = None
if "translated_response" not in st.session_state:
    st.session_state.translated_response = None
if "last_translation_language" not in st.session_state:
    st.session_state.last_translation_language = None
if "image_uploaded" not in st.session_state:
    st.session_state.image_uploaded = None
if "image_hash" not in st.session_state:
    st.session_state.image_hash = None
if "sidebar_shown" not in st.session_state:
    st.session_state.sidebar_shown = True  # Sidebar is shown by default

# ✅ Sidebar Workflow Guide
if st.session_state.sidebar_shown:
    with st.sidebar:
        st.title("📋 Workflow Guide")
        st.markdown("""
        **Follow these steps to use the app:**
        1. **Upload an image (📂)**:
           - Drag and drop or browse to upload an image.
        2. **View the description (📜)**:
           - Automatically generated after uploading the image.
        3. **Ask a question (💬)**:
           - Type your question about the image and get AI-powered answers.
        4. **Translate responses (🌐)**:
           - Translate the answers into your preferred language.
        5. **Listen to responses (🔊)**:
           - Play the text as audio in your desired language.

        *You can close this guide once you're familiar with the workflow.*
        """)
        # Close Sidebar Button
        if st.button("Close Sidebar"):
            st.session_state.sidebar_shown = False

# ✅ YOLO Object Detection (Handles Errors)
def detect_objects(image):
    try:
        results = yolo_model(image)
        detected_objects = []

        for result in results:
            for box in result.boxes:
                detected_objects.append(yolo_model.names[int(box.cls[0])])

        return list(set(detected_objects)) if detected_objects else None
    except Exception as e:
        st.error(f"❌ YOLO Detection Error: {e}")
        return None

# ✅ Generate a Better Image Description
def generate_image_description(image):
    detected_objects = detect_objects(image)

    if detected_objects:
        prompt = f"""
        You are an AI trained to describe images. Based on the detected objects ({', '.join(detected_objects)}), provide a brief but informative summary of the scene.
        
        - Describe the setting.
        - Mention possible interactions between objects.
        - Keep the description concise but useful.
        - Include any relevant details.
        - bullet points for key observations.
        - subheadings for clarity.
        - well-structured paragraphs.
        """

        try:
            response = genai.GenerativeModel(model_name="gemini-1.5-flash-8b").generate_content([image, prompt])
            return response.text
        except Exception as e:
            st.error(f"❌ AI Model Error: {e}")
            return f"Detected Objects: {', '.join(detected_objects)}"
    else:
        return "No objects were detected in the image."

# ✅ Enhanced TTS Handler
class TextToSpeech:
    def __init__(self):
        self.temp_dir = tempfile.TemporaryDirectory()

    def preprocess_text(self, text):
        """Preprocess text for TTS by removing special characters and normalizing spaces."""
        import re
        cleaned_lines = []
        bullet_count = 1

        for line in text.split("\n"):
            line = line.strip()  # Remove leading and trailing spaces
            if line:
                # Replace colons, multiple spaces, and special characters
                # line = re.sub(r":", " is", line)  # Replace `:` with ` is`
                line = re.sub(r":", "  ", line)  # Replace `:` with `  `
                line = re.sub(r"\s{2,}", " ", line)  # Replace multiple spaces with a single space
                line = re.sub(r"[-•*]", "", line)  # Remove bullet symbols like `-`, `•`, and `*`

                if line.startswith(("Point", "-")):
                    # Add "Point X" before each bullet point for better audio clarity
                    cleaned_lines.append(f"Point {bullet_count}: {line.strip()}")
                    bullet_count += 1
                else:
                    cleaned_lines.append(line)
        

        return " ".join(cleaned_lines)  # Combine the cleaned lines into a single string
        

    def generate_audio(self, text, lang="en"):
        """Generate simple TTS audio."""
        try:
            # Preprocess the text to clean it up
            cleaned_text = self.preprocess_text(text)
            if not cleaned_text.strip():
                st.error("❌ Text-to-Speech Error: No text available for audio generation.")
                return None

            tts = gTTS(text=cleaned_text, lang=lang)
            audio_path = os.path.join(self.temp_dir.name, "audio.mp3")
            tts.save(audio_path)
            return audio_path
        except Exception as e:
            st.error(f"❌ Text-to-Speech Error: {e}")
            return None

class ImageProcessor:
    def __init__(self):
        self.model = genai.GenerativeModel(model_name="gemini-1.5-flash-8b")
        self.faiss_index = self.load_faiss_index()
        self.embeddings_list, self.image_descriptions = self.load_embeddings_data()

    def load_faiss_index(self):
        if os.path.exists(FAISS_INDEX_PATH):
            return faiss.read_index(FAISS_INDEX_PATH)
        return faiss.IndexFlatL2(EMBEDDING_DIMENSION)

    def save_faiss_index(self):
        faiss.write_index(self.faiss_index, FAISS_INDEX_PATH)

    def load_embeddings_data(self):
        if os.path.exists(EMBEDDINGS_LIST_PATH) and os.path.exists(IMAGE_DESCRIPTIONS_PATH):
            with open(EMBEDDINGS_LIST_PATH, "rb") as f:
                embeddings_list = pickle.load(f)
            with open(IMAGE_DESCRIPTIONS_PATH, "rb") as f:
                image_descriptions = pickle.load(f)
            return embeddings_list, image_descriptions
        return [], []

    def save_embeddings_data(self):
        with open(EMBEDDINGS_LIST_PATH, "wb") as f:
            pickle.dump(self.embeddings_list, f)
        with open(IMAGE_DESCRIPTIONS_PATH, "wb") as f:
            pickle.dump(self.image_descriptions, f)

    def generate_image_hash(self, image):
        return hashlib.md5(image.tobytes()).hexdigest()

    async def process_image(self, image):
        # Generate image hash
        image_hash = self.generate_image_hash(image)

        # Check if the image is already processed
        if st.session_state.image_hash != image_hash:
            try:
                st.session_state.cached_description = None
                st.session_state.image_hash = image_hash

                with st.spinner("Generating detailed image description..."):
                    st.session_state.cached_description = generate_image_description(image)
                    st.success("✅ Image description and embedding cached successfully!")

            except UnidentifiedImageError:
                st.error("❌ Unable to process the uploaded file as an image.")
                return

    async def process_question(self, image, question):
        if not question:
            return

        progress_bar = st.progress(0)
        start_time = time.time()

        if "last_question" not in st.session_state or st.session_state.last_question != question:
            st.session_state.cached_response = None
            st.session_state.translated_response = None
            st.session_state.response_time = None
            st.session_state.last_question = question

        if st.session_state.cached_response is None:
            prompt = f"""
            You are an expert AI assistant that provides well-structured and detailed answers.
            Analyze the uploaded image and respond to the question:
            **Question:** {question}
            **Response format:**
            - Well-structured paragraphs
            - Key observations in bullet points
            - Subheadings for clarity
            """
            with st.spinner("Generating response..."):
                response = self.model.generate_content([image, prompt])
                st.session_state.cached_response = response.text
                st.session_state.response_time = round(time.time() - start_time, 2)

        progress_bar.progress(100)

def main():
    st.title("🖼️ AI Image Analysis with TTS")

    tts_instance = TextToSpeech()

    uploaded_file = st.file_uploader("📂 Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        processor = ImageProcessor()
        asyncio.run(processor.process_image(image))

        # Center the Image and Description
        st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
        if st.session_state.cached_description:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            st.subheader("📜 Image Description")
            st.write(st.session_state.cached_description)

            # 🔊 TTS for Image Description
            if st.button("🔊 Listen to Image Description"):
                audio_path = tts_instance.generate_audio(st.session_state.cached_description)
                if audio_path:
                    st.audio(audio_path, format="audio/mp3")
        st.markdown("</div>", unsafe_allow_html=True)

        # Two-Column Layout for Question, Response, and Translation
        col1, col2 = st.columns([2, 1])

        with col1:
            question = st.text_input("💬 Ask a question about the image:")
            if question:
                asyncio.run(processor.process_question(image, question))
                st.subheader("📝 Response to Your Question")
                st.write(st.session_state.cached_response)
                # Success message moved to end
                if st.session_state.response_time:
                    st.success(
                        f"✅ Response generated successfully in {st.session_state.response_time} seconds."
                    )

                # 🔊 TTS for Response
                if st.button("🔊 Listen to Response"):
                    if st.session_state.cached_response and st.session_state.cached_response.strip():
                        audio_path = tts_instance.generate_audio(st.session_state.cached_response)
                        if audio_path:
                            st.audio(audio_path, format="audio/mp3")
                    else:
                        st.error("❌ Text-to-Speech Error: No valid response available for audio generation.")

        with col2:
            if st.session_state.cached_response:
                st.write("")
                st.write("")  # Adds two blank lines
                st.subheader("🌐 Translate the Response")
                col_hindi, col_telugu, col_tamil = st.columns(3)

                with col_hindi:
                    if st.button("Translate in Hindi"):
                        st.session_state.translated_response = GoogleTranslator(source="auto", target="hi").translate(
                            st.session_state.cached_response
                        )
                        st.session_state.last_translation_language = "Hindi"

                with col_telugu:
                    if st.button("Translate in Telugu"):
                        st.session_state.translated_response = GoogleTranslator(source="auto", target="te").translate(
                            st.session_state.cached_response
                        )
                        st.session_state.last_translation_language = "Telugu"

                with col_tamil:
                    if st.button("Translate in Tamil"):
                        st.session_state.translated_response = GoogleTranslator(source="auto", target="ta").translate(
                            st.session_state.cached_response
                        )
                        st.session_state.last_translation_language = "Tamil"

                # Translated Response with TTS
                if st.session_state.translated_response:
                    st.write(f"**Translation ({st.session_state.last_translation_language}):**")
                    st.write(st.session_state.translated_response)

                    if st.button(f"🔊 Listen to {st.session_state.last_translation_language} Translation"):
                        audio_path = tts_instance.generate_audio(
                            st.session_state.translated_response,
                            lang="hi" if st.session_state.last_translation_language == "Hindi"
                            else "te" if st.session_state.last_translation_language == "Telugu"
                            else "ta"
                        )
                        if audio_path:
                            st.audio(audio_path, format="audio/mp3")

if __name__ == "__main__":
    main()
