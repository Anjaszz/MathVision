import os
import random
import time
import cv2
import numpy as np
import streamlit as st
from PIL import Image
import google.generativeai as genai
from src.HandDetector import HandDetector
from dotenv import load_dotenv

load_dotenv()
# Initialize Gemini API
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY") 
genai.configure(api_key=GOOGLE_API_KEY)

def set_streamlit_header():
    """Set up the Streamlit header with layout and styling"""
    st.set_page_config(page_title="MathVision", layout="wide", page_icon="🤖")
    
    # Custom CSS for header styling
    st.markdown("""
        <style>
        .header-container {
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 20px;
            padding: 20px;
        }
        .app-title {
            font-size: 42px;
            font-weight: bold;
            color: #1E88E5;
        }
        .app-slogan {
            font-size: 16px;
            color: #666;
            margin-top: 5px;
        }
        [data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
            width: 400px;
        }
        [data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
            width: 400px;
            margin-left: -400px;
        }
        </style>
    """, unsafe_allow_html=True)
    
    # Load image
    image = Image.open('resources/robot.png')
    
    # Create header with image and text
    st.markdown(
        f"""
        <div class="header-container">
            <img src="data:image/png;base64,{image_to_base64(image)}" width="120">
            <div>
                <div class="app-title">MathVision</div>
                <div class="app-slogan">Dari Coretan ke Jawaban, Semua di Satu Layar.</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    return image

def image_to_base64(image):
    """Convert PIL image to base64 string"""
    import io
    import base64
    
    # Convert PIL image to bytes
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    
    # Convert bytes to base64 string
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def set_streamlit_footer():
    """Set up the Streamlit footer"""
    col1, col2, col3 = st.columns([5, 3, 4])
    with col1:
        st.write(' ')
    with col2:
        st.markdown("""
                    ##### Create By [Anjaszzz](https://anjasrani.my.id)
                    """)
    with col3:
        st.write(' ')

def generate_user_prompt():
    """Generate the default user prompt"""
    prompt = r"""
         #### Selamat Datang di MathVision! 👋
         - Saya MathVision 🤖, asisten cerdas yang siap membantu Anda.
         - Silakan tunjukkan bentuk geometri yang ingin Anda hitung.
         - Saya akan membantu menghitung luas dan menjelaskan prosesnya secara detail.
        """
    return prompt

def response_generator(response, wait_time, my_bar=None, progress_text=""):
    """Generate streaming response with optional progress bar"""
    if my_bar:
        for percent_complete in range(100):
            time.sleep(random.uniform(0, 0.1))
            my_bar.progress(percent_complete + 1, text=progress_text)
        my_bar.empty()
    for word in response.split(" "):
        yield word + " "
        time.sleep(wait_time)

def set_basic_config():
    """Set up basic configuration including Gemini model and drawing tools"""
    # Initialize Gemini model
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    
    # Initialize hand detector and drawing parameters
    detector = HandDetector()
    brush_thick = 15
    eraser_thick = 40
    rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (30, 10))
    options = "--psm 8"
    counter_map = {
        "erase": 0,
        "write": 0,
        "go": 0
    }
    blkboard = np.zeros((720, 1280, 3), np.uint8)

    return gemini_model, detector, brush_thick, eraser_thick, rectKernel, options, counter_map, blkboard

def get_gemini_response(image_path, prompt_text=""):
    """Get response from Gemini model for image analysis"""
    try:
        # Get model from session state
        model = st.session_state.get("gemini_model")
        if not model:
            model = genai.GenerativeModel('gemini-1.5-flash')
            st.session_state["gemini_model"] = model
        
        # Open and process image
        image = Image.open(image_path)
        
        # Set default prompt if none provided
        if not prompt_text:
            prompt_text = "Analyze this mathematical image and provide a detailed solution in Markdown format. Break down the solution into clear steps. Use Indonesia Language"
        
        # Generate response
        response = model.generate_content([prompt_text, image])
        return response.text
    except Exception as e:
        error_message = f"Error processing image with Gemini: {str(e)}"
        st.error(error_message)
        return error_message

def chat_content():
    """Add content to session state"""
    st.session_state['contents'].append(st.session_state.content)

def process_image(image_array):
    """Process image array before sending to Gemini"""
    try:
        # Convert numpy array to PIL Image
        image = Image.fromarray(cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB))
        return image
    except Exception as e:
        st.error(f"Error processing image: {str(e)}")
        return None

def handle_response_stream(response_text):
    """Handle streaming of response text"""
    try:
        words = response_text.split()
        for word in words:
            yield word + " "
            time.sleep(0.05)
    except Exception as e:
        yield f"Error streaming response: {str(e)}"