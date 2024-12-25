import queue
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import streamlit as st
from src.utils.streamlitUtils import *

# Initialize the application
image = set_streamlit_header()
gemini_model, detector, brush_thick, eraser_thick, rectKernel, options, counter_map, blkboard = set_basic_config()

# Store the Gemini model in session state
if "gemini_model" not in st.session_state:
    st.session_state["gemini_model"] = gemini_model

# Initialize session state variables
if 'contents' not in st.session_state:
    st.session_state['contents'] = []
    border = False
else:
    border = True

if "messages" not in st.session_state:
    st.session_state.messages = []

def callback(frame, xp=0, yp=0):
    """Process video frames and handle hand gestures"""
    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)

    img = detector.find_hands(img)
    landmark_list = detector.find_position(img, draw=False)

    if len(landmark_list) != 0:
        _, x1, y1 = landmark_list[8]  # Tip of Index Finger
        _, x2, y2 = landmark_list[12]  # Tip of Middle Finger
        fingers = detector.fingers()

        if len(fingers) == 5:
            # Erase mode - Two fingers up
            if fingers[1] == 1 and fingers[2] == 1 and fingers[0] == 0 and fingers[3] == 0 and fingers[4] == 0:
                cv2.rectangle(img, (x1 - 25, y1 - 25), (x2 + 25, y2 + 25), (0, 0, 255), cv2.FILLED)
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                cv2.line(blkboard, (xp, yp), (x1, y1), (0, 0, 0), eraser_thick)
                counter_map["go"] = 0
                xp, yp = x1, y1

            # Write mode - One finger up
            elif fingers[1] == 1 and fingers[0] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                counter_map["go"] = 0
                if xp == 0 and yp == 0:
                    xp, yp = x1, y1
                cv2.line(blkboard, (xp, yp), (x1, y1), (0, 255, 0), brush_thick)
                xp, yp = x1, y1

            # Process mode - Thumb up
            elif fingers[0] == 1 and fingers[1] == 0 and fingers[2] == 0 and fingers[3] == 0 and fingers[4] == 0:
                xp, yp = 0, 0
                blackboard_gray = cv2.cvtColor(blkboard, cv2.COLOR_RGB2GRAY)
                blur1 = cv2.medianBlur(blackboard_gray, 15)
                blur1 = cv2.GaussianBlur(blur1, (5, 5), 0)
                thresh1 = cv2.threshold(blur1, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
                blackboard_cnts, _ = cv2.findContours(thresh1.copy(), cv2.RETR_EXTERNAL,
                                                    cv2.CHAIN_APPROX_SIMPLE)
                if len(blackboard_cnts) >= 1:
                    bounding_boxes = []
                    for cnt in sorted(blackboard_cnts, key=cv2.contourArea, reverse=True):
                        if cv2.contourArea(cnt) > 800:
                            x, y, w, h = cv2.boundingRect(cnt)
                            bounding_boxes.append((x, y))
                            bounding_boxes.append((x + w, y + h))
                    box = cv2.minAreaRect(np.asarray(bounding_boxes))
                    (x1, y1), (x2, y2), (x3, y3), (x4, y4) = cv2.boxPoints(box)
                    a1 = min(x1, x2, x3, x4)
                    a2 = max(x1, x2, x3, x4)
                    b1 = min(y1, y2, y3, y4)
                    b2 = max(y1, y2, y3, y4)
                    cv2.rectangle(img, (int(a1), int(b1)), (int(a2), int(b2)), (0, 255, 0), 2)
                    digit = blackboard_gray
                    counter_map["go"] += 1
                    if counter_map["go"] > 20:
                        result_queue.put(True)
                        counter_map["go"] = 0
                        cv2.imwrite("math.png", digit)
            else:
                xp, yp = 0, 0
                counter_map["go"] = 0

    # Combine drawing with camera feed
    gray = cv2.cvtColor(blkboard, cv2.COLOR_BGR2GRAY)
    _, inv = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY_INV)
    if img.shape[0] == 720 and img.shape[1] == 1280:
        inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)
        img = cv2.bitwise_and(img, inv)
        img = cv2.bitwise_or(img, blkboard)

    return av.VideoFrame.from_ndarray(img)

# Initialize queues
result_queue = queue.Queue()
chat_queue = []

# Create layout
col1, col2 = st.columns([6, 6])

# Video feed column
with col1:
    with st.container(height=590):
        ctx = webrtc_streamer(
            key="MathVision",
            mode=WebRtcMode.SENDRECV,
            async_processing=True,
            rtc_configuration={
                "iceServers": [
                    {"urls": ["stun:stun.l.google.com:19302"]},
                    {
                        'urls': ['turn:relay1.expressturn.com:3478'],
                        'username': 'efBWHN',
                        'credential': 'QWqbR2HkP0'
                    }
                ],
                "iceTransportPolicy": "all",
                "bundlePolicy": "max-bundle",
                "rtcpMuxPolicy": "require",
            },
            video_frame_callback=callback,
            media_stream_constraints={
                "video": {
                    "width": {"ideal": 1280, "min": 640},
                    "height": {"ideal": 720, "min": 480},
                    "frameRate": {"ideal": 30, "max": 30},
                    "facingMode": "user"
                },
                "audio": False,
            },
            translations={
                "start": "Mulai Kamera",
                "stop": "Stop Kamera",
                "select_device": "Pilih Kamera",
                "media_api_not_available": "Media API tidak tersedia",
                "device_ask_permission": "Mohon izinkan akses kamera",
                "device_not_available": "Kamera tidak tersedia",
                "device_access_denied": "Akses kamera ditolak"
            }
        )

# Chat interface column
with col2:
    if ctx is not None:
        with st.container(height=532):
            # Header
            col1, col2, col3 = st.columns([4, 2, 4])
            with col1:
                st.write(' ')
            with col2:
                st.markdown("""## Jawaban """)
            with col3:
                st.write(' ')

            # Main chat loop
            while ctx.state.playing:
                try:
                    chat = None
                    result = None
                    try:
                        result = result_queue.get(timeout=1.0)
                    except queue.Empty:
                        result = False

                    chat_queue = st.session_state['contents']
                    if len(chat_queue) > 0:
                        chat = chat_queue[-1]

                    # Handle initial image analysis
                    if result and chat is None:
                        with st.chat_message("user", avatar="👩‍🚀"):
                            st.write_stream(response_generator(generate_user_prompt(), 0.01))
                            col1, col2, col3 = st.columns([2, 3, 3])
                            with col1:
                                st.write(' ')
                            with col2:
                                st.image('math.png', width=300, caption='Image submitted by user')
                            with col3:
                                st.write(' ')

                        with st.chat_message("assistant", avatar="🤖"):
                            progress_bar = st.progress(0, text="Menganalisis soal matematika...")
                            response = get_gemini_response("math.png")
                            # Simulate progress while waiting for response
                            response = st.write_stream(response_generator(
                                response,
                                0.01,
                                my_bar=progress_bar,
                                progress_text="Menghasilkan solusi..."
                            ))
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response
                            })
                            result_queue.queue.clear()

                    # Handle follow-up chat
                    if chat and len(chat) > 0:
                        # Display message history
                        for message in st.session_state.messages:
                            with st.chat_message(message["role"], avatar="🤷‍♂️" if message["role"] == "user" else "🤖"):
                                st.markdown(message["content"])

                        # Display current user message
                        with st.chat_message("user", avatar="🤷‍♂️"):
                            st.write_stream(response_generator(chat, 0.01))
                            col1, col2, col3 = st.columns([2, 3, 3])
                            with col1:
                                st.write(' ')
                            with col2:
                                st.image('math.png', width=300, caption='Image submitted by user')
                            with col3:
                                st.write(' ')

                        st.session_state.messages.append({"role": "user", "content": chat})
                        st.session_state['contents'] = []

                        # Get and display assistant response
                        with st.chat_message("assistant", avatar="🤖"):
                            progress_bar = st.progress(0, text="Memproses pertanyaan lanjutan...")
                            response = get_gemini_response("math.png", f"ini adalah perintah lanjutan untuk soal anda: {chat}")
                            # Simulate progress while waiting for response
                            response = st.write_stream(response_generator(
                                response,
                                0.01,
                                my_bar=progress_bar,
                                progress_text="Menghasilkan jawaban..."
                            ))
                            st.session_state.messages.append({
                                "role": "assistant",
                                "content": response
                            })

                except queue.Empty:
                    continue

        # Chat input
        with st.container(border=False):
            with st.container():
                prompt = st.chat_input(placeholder="Ketik Perintah lanjutan disini 🤖", key='content', on_submit=chat_content)
                if prompt is not None:
                    chat_queue.append(prompt)

# Set footer
set_streamlit_footer()