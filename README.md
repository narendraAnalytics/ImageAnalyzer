# AI Image Analysis with TTS

## Overview
This project is a **Streamlit-based AI Image Analysis Tool** that leverages **YOLOv8, Google Gemini AI, FAISS for similarity search, Deep Translator, and gTTS for text-to-speech conversion**. The application allows users to:

- **Upload an image** 📂
- **Generate an AI-powered description** 📜
- **Ask questions about the image** 💬
- **Translate the response into different languages** 🌐
- **Listen to responses using text-to-speech (TTS)** 🔊

## Features
- **Object Detection**: Uses YOLOv8 to detect objects in the uploaded image.
- **AI-Powered Description**: Generates detailed descriptions based on detected objects using Google Gemini AI.
- **Question Answering**: Allows users to ask questions about the image and receive AI-generated answers.
- **Translation Support**: Supports translation of responses into Hindi, Telugu, and Tamil.
- **Text-to-Speech (TTS)**: Converts text responses into speech for better accessibility.
- **FAISS for Image Embeddings**: Stores and retrieves image embeddings for optimized searches.

## Installation
To run this project, follow these steps:

### 1. Clone the Repository
```sh
git clone https://github.com/yourusername/ai-image-analysis.git
cd ai-image-analysis
```

### 2. Create a Virtual Environment
```sh
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Set Up API Keys
Create a `.env` file and add your **Google Gemini API key**:
```sh
GEMINI_API_KEY=your_api_key_here
```
Alternatively, store API keys using `st.secrets` in Streamlit.

### 5. Run the Application
```sh
streamlit run app.py
```

## Usage
1. **Upload an image**: Drag and drop or browse to upload an image.
2. **View the AI-generated description**.
3. **Ask a question** about the image to get AI-powered insights.
4. **Translate** responses into Hindi, Telugu, or Tamil.
5. **Listen to the response** in the selected language using TTS.

## Technologies Used
- **Streamlit**: For interactive UI
- **YOLOv8**: Object detection
- **Google Gemini AI**: AI-generated text responses
- **FAISS**: Efficient similarity search
- **Deep Translator**: Language translation
- **gTTS**: Text-to-speech conversion
- **Pillow (PIL)**: Image processing
- **Decouple**: Environment variable management

## File Structure
```
📂 ai-image-analysis
├── 📂 models/               # YOLO model files
├── 📂 data/                 # Image embeddings and FAISS index
├── 📂 utils/                # Helper functions
├── app.py                   # Main Streamlit application
├── requirements.txt          # Required dependencies
├── .env                      # API keys (not committed to Git)
├── README.md                 # Project documentation
```

## Future Enhancements
- Support for more languages in translation
- Additional object detection models
- Enhanced speech synthesis with more natural voice models
- Cloud-based image storage for scalability

## License
This project is licensed under the MIT License.

## Author
- **Your Name**  
- GitHub: [NarendraKumar](https://github.com/yourusername)

