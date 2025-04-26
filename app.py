import streamlit as st
import cv2
import numpy as np
import pandas as pd
import time
import threading
import datetime
import os
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from emotion_detection.facial_analysis import FacialEmotionAnalyzer
from emotion_detection.voice_analysis import VoiceEmotionAnalyzer
from emotion_detection.utils import draw_emotions_chart, combine_emotion_predictions, check_devices
from emotion_detection.database import EmotionDatabase

# Page configuration
st.set_page_config(
    page_title="AI Emotion Detector",
    page_icon="😀",
    layout="wide",
)

# Initialize the database connection
db = EmotionDatabase()

# Initialize session state
if 'facial_emotions' not in st.session_state:
    st.session_state.facial_emotions = {
        'happy': 0.0,
        'sad': 0.0,
        'angry': 0.0,
        'surprised': 0.0,
        'neutral': 100.0,
        'fear': 0.0
    }

if 'voice_emotions' not in st.session_state:
    st.session_state.voice_emotions = {
        'happy': 0.0,
        'sad': 0.0,
        'angry': 0.0,
        'surprised': 0.0,
        'neutral': 100.0,
        'fear': 0.0
    }

if 'combined_emotions' not in st.session_state:
    st.session_state.combined_emotions = {
        'happy': 0.0,
        'sad': 0.0,
        'angry': 0.0,
        'surprised': 0.0,
        'neutral': 100.0,
        'fear': 0.0
    }

if 'is_recording' not in st.session_state:
    st.session_state.is_recording = False

if 'recording_thread' not in st.session_state:
    st.session_state.recording_thread = None

if 'audio_data' not in st.session_state:
    st.session_state.audio_data = None

if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0

if 'emotion_history' not in st.session_state:
    st.session_state.emotion_history = []

if 'run_camera' not in st.session_state:
    st.session_state.run_camera = False

if 'database_session_id' not in st.session_state:
    # Start a new database session
    session_id = db.start_session("Streamlit session")
    st.session_state.database_session_id = session_id

if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None

if 'uploaded_audio' not in st.session_state:
    st.session_state.uploaded_audio = None

# Initialize analyzers
facial_analyzer = FacialEmotionAnalyzer()
voice_analyzer = VoiceEmotionAnalyzer()

# App title and intro
st.title("AI Emotion Detector")
st.markdown("""
This application analyzes your facial expressions and voice to detect emotions.
Allow access to your camera and microphone for the best experience or upload files for analysis.
""")

# Check device availability
camera_available, mic_available = check_devices()

# Setup tabs for different sections
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Live Analysis", "File Upload", "Voice Analysis", "Dashboard",
    "Database Records"
])

with tab1:
    st.header("Live Facial and Voice Analysis")

    # Define tab1 placeholders
    tab1_camera_placeholder = st.empty()
    tab1_face_emotion_placeholder = st.empty()
    tab1_voice_emotion_placeholder = st.empty()
    tab1_combined_emotions_placeholder = st.empty()

    if not camera_available:
        st.error(
            "Camera not available. Please check your device and browser permissions or use the File Upload tab."
        )
    else:
        # Camera settings
        st.markdown("### Facial Expression Analysis")
        camera_options = st.columns([1, 1])
        with camera_options[0]:
            st.session_state.run_camera = st.checkbox(
                "Enable Camera", value=st.session_state.run_camera)

        # Emotions from facial analysis
        st.markdown("### Current Facial Emotions")

    # Voice analysis (simulated)
    st.markdown("### Voice Analysis")

    # Use colored box to make controls more prominent
    st.markdown("""
    <div style="background-color:#f0f2f6; padding:15px; border-radius:10px; margin-bottom:15px;">
    <h4 style="color:#0C2D57;">Input Controls</h4>
    <p>Click the button below to generate random voice emotion data.</p>
    </div>
    """,
                unsafe_allow_html=True)

    voice_col1, voice_col2 = st.columns([1, 1])

    with voice_col1:
        if not st.session_state.is_recording:
            if st.button("🎙️ Simulate Voice Analysis",
                         use_container_width=True):
                st.session_state.is_recording = True

                def simulate_voice_analysis():
                    # Simulate processing time
                    time.sleep(2)

                    # Generate simulated voice emotion data
                    # This would normally come from actual audio analysis
                    voice_emotions = {
                        'happy': np.random.uniform(10, 40),
                        'sad': np.random.uniform(5, 20),
                        'angry': np.random.uniform(5, 15),
                        'surprised': np.random.uniform(5, 25),
                        'neutral': np.random.uniform(20, 50),
                        'fear': np.random.uniform(5, 15)
                    }

                    # Normalize to 100%
                    total = sum(voice_emotions.values())
                    for emotion in voice_emotions:
                        voice_emotions[emotion] = round(
                            (voice_emotions[emotion] / total) * 100.0, 1)

                    st.session_state.voice_emotions = voice_emotions

                    # Combine with facial emotions
                    st.session_state.combined_emotions = combine_emotion_predictions(
                        st.session_state.facial_emotions, voice_emotions)

                    # Add to emotion history
                    timestamp = time.strftime("%H:%M:%S")
                    history_entry = {
                        "timestamp": timestamp,
                        "emotions": voice_emotions.copy()
                    }
                    st.session_state.emotion_history.append(history_entry)

                    st.session_state.is_recording = False
                    st.rerun()

                # Start simulation in a thread to avoid blocking the UI
                thread = threading.Thread(target=simulate_voice_analysis)
                thread.start()
                st.session_state.recording_thread = thread
        else:
            st.write("Analyzing voice patterns...")
            # Show a spinner while simulating
            with st.spinner("Processing voice data..."):
                time.sleep(0.1)  # Small delay to allow UI to update

    # Display voice emotion results
    st.markdown("### Current Voice Emotions")

with tab2:
    st.header("File Upload Analysis")

    st.markdown("""
    This tab allows you to upload images and audio files for emotion analysis.
    Upload a facial image to analyze expressions or an audio file to analyze voice patterns.
    """)

    # Create two columns for file upload
    upload_col1, upload_col2 = st.columns(2)

    with upload_col1:
        st.markdown("### Upload Facial Image")
        uploaded_image = st.file_uploader("Choose an image file",
                                          type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            st.session_state.uploaded_image = uploaded_image
            # Display the uploaded image
            image_bytes = uploaded_image.getvalue()
            image_array = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            # Display image
            st.image(image, channels="BGR", caption="Uploaded Image")

            # Analyze the image
            if st.button("Analyze Facial Expression",
                         use_container_width=True):
                with st.spinner("Analyzing facial expression..."):
                    # Process the frame for emotion detection
                    emotions, processed_frame = facial_analyzer.analyze_frame(
                        image)

                    # Update session state with new emotions
                    if emotions:
                        st.session_state.facial_emotions = emotions

                        # Combine with voice emotions
                        st.session_state.combined_emotions = combine_emotion_predictions(
                            emotions, st.session_state.voice_emotions)

                        # Add to emotion history
                        timestamp = time.strftime("%H:%M:%S")
                        history_entry = {
                            "timestamp": timestamp,
                            "emotions": emotions.copy()
                        }
                        st.session_state.emotion_history.append(history_entry)

                        # Display the results
                        st.markdown("### Analysis Results")

                        # Display the processed image with annotations
                        rgb_frame = cv2.cvtColor(processed_frame,
                                                 cv2.COLOR_BGR2RGB)
                        st.image(rgb_frame,
                                 channels="RGB",
                                 caption="Analyzed Image")

                        # Display emotions chart
                        emotions_chart = draw_emotions_chart(
                            emotions, "Facial Emotions")
                        st.plotly_chart(emotions_chart,
                                        use_container_width=True)
                    else:
                        st.error(
                            "No face detected in the image. Please upload a clear image with a face."
                        )

    with upload_col2:
        st.markdown("### Upload Audio File")
        uploaded_audio = st.file_uploader("Choose an audio file",
                                          type=["wav", "mp3"])

        if uploaded_audio is not None:
            st.session_state.uploaded_audio = uploaded_audio

            # Save the uploaded audio file temporarily
            audio_path = f"temp_audio_{int(time.time())}.wav"
            with open(audio_path, "wb") as f:
                f.write(uploaded_audio.getvalue())

            st.audio(uploaded_audio, format="audio/wav")

            # Analyze the audio
            if st.button("Analyze Voice Emotion", use_container_width=True):
                with st.spinner("Analyzing voice emotion..."):
                    # We'll use a simulated result for demo purposes
                    # In a real application, you would use:
                    # voice_emotions = voice_analyzer.analyze_audio(audio_path)

                    # Simulated emotions based on random values
                    voice_emotions = {
                        'happy': np.random.uniform(10, 40),
                        'sad': np.random.uniform(5, 20),
                        'angry': np.random.uniform(5, 15),
                        'surprised': np.random.uniform(5, 25),
                        'neutral': np.random.uniform(20, 50),
                        'fear': np.random.uniform(5, 15)
                    }

                    # Normalize to 100%
                    total = sum(voice_emotions.values())
                    for emotion in voice_emotions:
                        voice_emotions[emotion] = round(
                            (voice_emotions[emotion] / total) * 100.0, 1)

                    # Update session state
                    st.session_state.voice_emotions = voice_emotions

                    # Combine with facial emotions
                    st.session_state.combined_emotions = combine_emotion_predictions(
                        st.session_state.facial_emotions, voice_emotions)

                    # Add to emotion history
                    timestamp = time.strftime("%H:%M:%S")
                    history_entry = {
                        "timestamp": timestamp,
                        "emotions": voice_emotions.copy()
                    }
                    st.session_state.emotion_history.append(history_entry)

                    # Display the results
                    st.markdown("### Analysis Results")

                    # Display the audio waveform
                    try:
                        # Create a placeholder waveform visualization
                        fig, ax = plt.subplots(figsize=(10, 2))
                        # Generate random waveform for visualization
                        audio_length = 16000 * 5
                        audio_data = np.sin(
                            np.linspace(0, 100 * np.pi,
                                        audio_length)) * np.random.normal(
                                            0.5, 0.5, audio_length)
                        ax.plot(audio_data[:1000])
                        ax.set_xlabel("Time")
                        ax.set_ylabel("Amplitude")
                        ax.set_title("Audio Waveform")
                        st.pyplot(fig)
                    except Exception as e:
                        st.warning(f"Could not display waveform: {e}")

                    # Display emotions chart
                    emotions_chart = draw_emotions_chart(
                        voice_emotions, "Voice Emotions")
                    st.plotly_chart(emotions_chart, use_container_width=True)

                    # Clean up temporary file
                    try:
                        os.remove(audio_path)
                    except:
                        pass

with tab3:
    st.header("Voice Emotion Analysis")

    st.markdown("""
    This tab allows you to analyze voice emotion data in multiple ways:
    1. Simulate voice patterns with predefined emotion settings
    2. Enter custom emotion values manually 
    """)

    # Use colored box to make controls more prominent
    st.markdown("""
    <div style="background-color:#f0f2f6; padding:20px; border-radius:10px; margin-bottom:20px;">
    <h3 style="color:#0C2D57;">Input Controls</h3>
    <p>Use these controls to simulate different emotional patterns or enter custom values.</p>
    </div>
    """,
                unsafe_allow_html=True)

    analysis_type = st.radio("Choose analysis method:",
                             ["Predefined Patterns", "Custom Values"],
                             horizontal=True)

    if analysis_type == "Predefined Patterns":
        record_col1, record_col2 = st.columns([1, 1])

        with record_col1:
            # Create different emotion pattern options
            emotion_pattern = st.selectbox(
                "Select emotion pattern to simulate:", [
                    "Neutral", "Happy", "Sad", "Angry", "Surprised", "Fearful",
                    "Mixed"
                ])

            if not st.session_state.is_recording:
                if st.button("🎙️ Simulate Voice Analysis",
                             key="simulate_tab2",
                             use_container_width=True):
                    st.session_state.is_recording = True

                    def simulate_voice_analysis_tab2():
                        # Simulate processing time
                        time.sleep(2)

                        # Generate simulated voice emotion data based on selected pattern
                        if emotion_pattern == "Happy":
                            voice_emotions = {
                                'happy': np.random.uniform(60, 90),
                                'sad': np.random.uniform(0, 10),
                                'angry': np.random.uniform(0, 5),
                                'surprised': np.random.uniform(5, 15),
                                'neutral': np.random.uniform(5, 15),
                                'fear': np.random.uniform(0, 5)
                            }
                        elif emotion_pattern == "Sad":
                            voice_emotions = {
                                'happy': np.random.uniform(0, 10),
                                'sad': np.random.uniform(60, 90),
                                'angry': np.random.uniform(5, 15),
                                'surprised': np.random.uniform(0, 5),
                                'neutral': np.random.uniform(5, 15),
                                'fear': np.random.uniform(5, 15)
                            }
                        elif emotion_pattern == "Angry":
                            voice_emotions = {
                                'happy': np.random.uniform(0, 5),
                                'sad': np.random.uniform(5, 15),
                                'angry': np.random.uniform(60, 90),
                                'surprised': np.random.uniform(5, 15),
                                'neutral': np.random.uniform(0, 10),
                                'fear': np.random.uniform(5, 15)
                            }
                        elif emotion_pattern == "Surprised":
                            voice_emotions = {
                                'happy': np.random.uniform(10, 20),
                                'sad': np.random.uniform(0, 5),
                                'angry': np.random.uniform(0, 10),
                                'surprised': np.random.uniform(60, 90),
                                'neutral': np.random.uniform(0, 10),
                                'fear': np.random.uniform(5, 15)
                            }
                        elif emotion_pattern == "Fearful":
                            voice_emotions = {
                                'happy': np.random.uniform(0, 5),
                                'sad': np.random.uniform(10, 20),
                                'angry': np.random.uniform(5, 15),
                                'surprised': np.random.uniform(5, 15),
                                'neutral': np.random.uniform(0, 10),
                                'fear': np.random.uniform(60, 90)
                            }
                        elif emotion_pattern == "Mixed":
                            voice_emotions = {
                                'happy': np.random.uniform(10, 30),
                                'sad': np.random.uniform(10, 30),
                                'angry': np.random.uniform(10, 30),
                                'surprised': np.random.uniform(10, 30),
                                'neutral': np.random.uniform(10, 30),
                                'fear': np.random.uniform(10, 30)
                            }
                        else:  # Neutral
                            voice_emotions = {
                                'happy': np.random.uniform(5, 15),
                                'sad': np.random.uniform(5, 15),
                                'angry': np.random.uniform(0, 10),
                                'surprised': np.random.uniform(0, 10),
                                'neutral': np.random.uniform(60, 90),
                                'fear': np.random.uniform(0, 5)
                            }

                        # Normalize to 100%
                        total = sum(voice_emotions.values())
                        for emotion in voice_emotions:
                            voice_emotions[emotion] = round(
                                (voice_emotions[emotion] / total) * 100.0, 1)

                        # Update session state
                        st.session_state.voice_emotions = voice_emotions

                        # Combine with facial emotions
                        st.session_state.combined_emotions = combine_emotion_predictions(
                            st.session_state.facial_emotions, voice_emotions)

                        # Generate simulated audio data
                        audio_length = 16000 * 5  # 5 seconds at 16kHz
                        audio_data = np.sin(
                            np.linspace(0, 100 * np.pi,
                                        audio_length)) * np.random.normal(
                                            0.5, 0.5, audio_length)
                        st.session_state.audio_data = audio_data

                        # Add to emotion history
                        timestamp = time.strftime("%H:%M:%S")
                        history_entry = {
                            "timestamp": timestamp,
                            "emotions": voice_emotions.copy()
                        }
                        st.session_state.emotion_history.append(history_entry)

                        st.session_state.is_recording = False
                        st.rerun()

                    # Start simulation in a thread to avoid blocking the UI
                    thread = threading.Thread(
                        target=simulate_voice_analysis_tab2)
                    thread.start()
                    st.session_state.recording_thread = thread
            else:
                st.write("Analyzing voice patterns...")
                # Show a spinner while simulating
                with st.spinner("Processing voice data..."):
                    time.sleep(0.1)  # Small delay to allow UI to update
    else:
        # Custom emotion values input
        st.write(
            "Enter custom percentages for each emotion (total should equal 100%):"
        )
        happy_val = st.slider("Happy", 0, 100, 20)
        sad_val = st.slider("Sad", 0, 100, 10)
        angry_val = st.slider("Angry", 0, 100, 10)
        surprised_val = st.slider("Surprised", 0, 100, 10)
        neutral_val = st.slider("Neutral", 0, 100, 40)
        fear_val = st.slider("Fear", 0, 100, 10)

        total = happy_val + sad_val + angry_val + surprised_val + neutral_val + fear_val

        if total != 100:
            st.warning(
                f"Total percentage is {total}%. Please adjust values to total 100%."
            )

        if st.button("Apply Custom Values", use_container_width=True):
            # Set custom emotions
            custom_emotions = {
                'happy': float(happy_val),
                'sad': float(sad_val),
                'angry': float(angry_val),
                'surprised': float(surprised_val),
                'neutral': float(neutral_val),
                'fear': float(fear_val)
            }

            # Update session state
            st.session_state.voice_emotions = custom_emotions

            # Combine with facial emotions
            st.session_state.combined_emotions = combine_emotion_predictions(
                st.session_state.facial_emotions, custom_emotions)

            # Add to emotion history
            timestamp = time.strftime("%H:%M:%S")
            history_entry = {
                "timestamp": timestamp,
                "emotions": custom_emotions.copy()
            }
            st.session_state.emotion_history.append(history_entry)

            st.success("Custom emotion values applied!")
            st.rerun()

    # Display audio waveform if available
    if st.session_state.audio_data is not None:
        st.markdown("### Simulated Audio Waveform")
        fig, ax = plt.subplots(figsize=(10, 2))
        ax.plot(st.session_state.audio_data[:1000]
                )  # Plot just a portion for performance
        ax.set_xlabel("Time")
        ax.set_ylabel("Amplitude")
        ax.set_title("Voice Pattern")
        st.pyplot(fig)

    # Display voice emotion chart
    st.markdown("### Voice Emotion Analysis Results")
    if st.session_state.voice_emotions:
        voice_chart = draw_emotions_chart(st.session_state.voice_emotions,
                                          "Voice Emotions")
        st.plotly_chart(voice_chart,
                        use_container_width=True,
                        key="voice_tab3_chart")
    else:
        st.info(
            "No voice analysis data available. Click 'Simulate Voice Analysis' to generate data."
        )

    # Display emotion history
    if st.session_state.emotion_history:
        st.markdown("### Emotion History")
        history_df = pd.DataFrame([{
            "Time": entry["timestamp"],
            "Happy": entry["emotions"]["happy"],
            "Sad": entry["emotions"]["sad"],
            "Angry": entry["emotions"]["angry"],
            "Surprised": entry["emotions"]["surprised"],
            "Neutral": entry["emotions"]["neutral"],
            "Fear": entry["emotions"]["fear"]
        } for entry in st.session_state.emotion_history])
        st.dataframe(history_df)

with tab4:
    st.header("Emotion Dashboard")

    # Combined emotions
    st.markdown("### Combined Emotion Analysis")
    st.markdown(
        "This chart shows the combined analysis from both facial expressions and voice."
    )
    tab4_combined_emotions_placeholder = st.empty()

    # Historical data
    st.markdown("### Emotion Detection Over Time")

    if len(st.session_state.emotion_history) > 0:
        # Create a line chart of emotions over time
        fig = go.Figure()
        emotions = ["happy", "sad", "angry", "surprised", "neutral", "fear"]
        colors = [
            "#FFC107", "#3F51B5", "#F44336", "#9C27B0", "#4CAF50", "#607D8B"
        ]

        timestamps = [
            entry["timestamp"] for entry in st.session_state.emotion_history
        ]

        for i, emotion in enumerate(emotions):
            values = [
                entry["emotions"][emotion]
                for entry in st.session_state.emotion_history
            ]
            fig.add_trace(
                go.Scatter(x=timestamps,
                           y=values,
                           mode='lines+markers',
                           name=emotion.capitalize(),
                           line=dict(color=colors[i], width=2),
                           marker=dict(size=8)))

        fig.update_layout(title="Emotion Trends Over Time",
                          xaxis_title="Time",
                          yaxis_title="Emotion Percentage",
                          yaxis=dict(range=[0, 100]),
                          legend=dict(orientation="h",
                                      yanchor="bottom",
                                      y=1.02,
                                      xanchor="right",
                                      x=1),
                          height=400)

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(
            "No emotion history available. Record your voice or enable the camera to start collecting data."
        )

with tab5:
    st.header("Database Records")

    st.markdown("""
    This tab shows emotion data stored in the PostgreSQL database. 
    Data is saved automatically every 30 seconds during analysis.
    """)

    # Session information
    st.subheader("Current Session")
    session_id = st.session_state.database_session_id
    if session_id:
        st.write(f"Session ID: {session_id}")
        st.write(
            f"Started at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        # Get session data from database
        session_records = db.get_session_emotions(session_id)
        if session_records:
            st.subheader("Session Emotion Records")

            # Convert to DataFrame for display
            records_data = []
            for record in session_records:
                emotions = record['emotions']
                record_data = {
                    "ID": record['id'],
                    "Timestamp": record['timestamp'],
                    "Source": record['source'],
                    "Happy": emotions.get('happy', 0),
                    "Sad": emotions.get('sad', 0),
                    "Angry": emotions.get('angry', 0),
                    "Surprised": emotions.get('surprised', 0),
                    "Neutral": emotions.get('neutral', 0),
                    "Fear": emotions.get('fear', 0)
                }
                records_data.append(record_data)

            if records_data:
                df = pd.DataFrame(records_data)
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No emotion records found for this session yet.")

            # Add button to store current emotions
            if st.button("Save Current Emotions to Database"):
                # Save facial emotions if available
                if sum(st.session_state.facial_emotions.values()) > 0:
                    db.record_emotion(session_id, 'facial',
                                      st.session_state.facial_emotions)

                # Save voice emotions if available
                if sum(st.session_state.voice_emotions.values()) > 0:
                    db.record_emotion(session_id, 'voice',
                                      st.session_state.voice_emotions)

                # Save combined emotions if available
                if sum(st.session_state.combined_emotions.values()) > 0:
                    db.record_emotion(session_id, 'combined',
                                      st.session_state.combined_emotions)

                st.success("Current emotions saved to database!")
                st.rerun()
        else:
            st.info("No emotion records in the database yet.")

        # Historical sessions
        st.subheader("Previous Sessions")
        recent_sessions = db.get_recent_sessions(5)
        if recent_sessions:
            sessions_data = []
            for session in recent_sessions:
                if session['id'] != session_id:  # Skip current session
                    session_data = {
                        "ID":
                        session['id'],
                        "Start Time":
                        session['start_time'],
                        "End Time":
                        session['end_time']
                        if session['end_time'] else "Active",
                        "Records":
                        session['record_count']
                    }
                    sessions_data.append(session_data)

            if sessions_data:
                st.dataframe(pd.DataFrame(sessions_data),
                             use_container_width=True)
            else:
                st.info("No previous sessions found.")
        else:
            st.info("No previous sessions found in the database.")

# Database storage for emotion data (every 30 seconds)
if 'last_db_update' not in st.session_state:
    st.session_state.last_db_update = time.time()

current_time = time.time()
if current_time - st.session_state.last_db_update > 30 and st.session_state.database_session_id:
    # Save current emotions to database
    if st.session_state.facial_emotions:
        db.record_emotion(st.session_state.database_session_id, 'facial',
                          st.session_state.facial_emotions)
    if st.session_state.voice_emotions:
        db.record_emotion(st.session_state.database_session_id, 'voice',
                          st.session_state.voice_emotions)
    if st.session_state.combined_emotions:
        db.record_emotion(st.session_state.database_session_id, 'combined',
                          st.session_state.combined_emotions)

    # Update the last db update time
    st.session_state.last_db_update = current_time

# Initialize camera variables
tab1_camera_active = False
camera_cap = None

# Main application loop
if st.session_state.run_camera and camera_available:
    tab1_camera_active = True
    camera_cap = cv2.VideoCapture(0)

    try:
        while st.session_state.run_camera:
            ret, frame = camera_cap.read()
            if not ret:
                st.error("Failed to capture image from camera.")
                break

            # Process the frame for emotion detection
            emotions, processed_frame = facial_analyzer.analyze_frame(frame)

            # Update session state with new emotions
            if emotions:
                st.session_state.facial_emotions = emotions

                # Combine with voice emotions
                st.session_state.combined_emotions = combine_emotion_predictions(
                    emotions, st.session_state.voice_emotions)

                # Every 30 frames (about 1 second), add to history
                st.session_state.frame_count += 1
                if st.session_state.frame_count % 30 == 0:
                    timestamp = time.strftime("%H:%M:%S")
                    history_entry = {
                        "timestamp": timestamp,
                        "emotions": emotions.copy()
                    }
                    st.session_state.emotion_history.append(history_entry)

            # Convert colors from BGR to RGB for display
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

            # Display the processed frame
            tab1_camera_placeholder.image(rgb_frame,
                                          channels="RGB",
                                          use_column_width=True)

            # Display emotions as chart
            face_emotion_fig = draw_emotions_chart(emotions, "Facial Emotions")
            tab1_face_emotion_placeholder.plotly_chart(
                face_emotion_fig,
                use_container_width=True,
                key="face_emotions_live")

            # Display voice emotions
            voice_emotion_fig = draw_emotions_chart(
                st.session_state.voice_emotions, "Voice Emotions")
            tab1_voice_emotion_placeholder.plotly_chart(
                voice_emotion_fig,
                use_container_width=True,
                key="voice_emotions_live")

            # Display combined emotions
            combined_emotion_fig = draw_emotions_chart(
                st.session_state.combined_emotions, "Combined Emotions")
            tab1_combined_emotions_placeholder.plotly_chart(
                combined_emotion_fig,
                use_container_width=True,
                key="combined_emotions_live")
            tab4_combined_emotions_placeholder.plotly_chart(
                combined_emotion_fig,
                use_container_width=True,
                key="combined_emotions_dashboard")

            # Small delay to allow UI updates
            time.sleep(0.03)

            # Check if the checkbox has been unchecked
            if not st.session_state.run_camera:
                break

    finally:
        if camera_cap and camera_cap.isOpened():
            camera_cap.release()
else:
    # Display static info when camera is off
    tab1_camera_placeholder.info(
        "Camera is turned off. Enable it to see live analysis.")

    # Still show the latest emotion data
    face_emotion_fig = draw_emotions_chart(st.session_state.facial_emotions,
                                           "Facial Emotions")
    tab1_face_emotion_placeholder.plotly_chart(face_emotion_fig,
                                               use_container_width=True,
                                               key="face_emotions_static")

    voice_emotion_fig = draw_emotions_chart(st.session_state.voice_emotions,
                                            "Voice Emotions")
    tab1_voice_emotion_placeholder.plotly_chart(voice_emotion_fig,
                                                use_container_width=True,
                                                key="voice_emotions_static")

    combined_emotion_fig = draw_emotions_chart(
        st.session_state.combined_emotions, "Combined Emotions")
    tab1_combined_emotions_placeholder.plotly_chart(
        combined_emotion_fig,
        use_container_width=True,
        key="combined_emotions_static")
    tab4_combined_emotions_placeholder.plotly_chart(
        combined_emotion_fig,
        use_container_width=True,
        key="combined_emotions_dashboard_static")

# Display instructions at the bottom
st.markdown("""
### Instructions for optimal usage:
1. **Live Analysis Tab**: Enable the camera for real-time facial analysis and use the Simulate button for voice
2. **File Upload Tab**: Upload images or audio files for offline analysis
3. **Voice Analysis Tab**: Simulate different emotion patterns or set custom values
4. **Dashboard Tab**: See emotion trends over time and combined analysis
5. **Database Records**: View stored emotion data and previous sessions
""")
