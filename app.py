#Air writing 
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Streamlit App 
st.title("Air Writing System with Finger Count")

# Create a placeholder for the video feed
FRAME_WINDOW = st.image([])

# Open Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        st.error("Failed to capture video feed")
        break

    # Flip the frame horizontally for a more natural feel
    frame = cv2.flip(frame, 1)

    # Convert frame to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame for hand detection
    results = hands.process(rgb_frame)

    # Draw hand landmarks and count fingers
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get landmarks
            landmarks = hand_landmarks.landmark

            # Define tips of fingers
            tip_ids = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
            fingers = []

            # Check thumb (special case)
            if landmarks[tip_ids[0]].x < landmarks[tip_ids[0] - 1].x:
                fingers.append(1)  # Thumb is open
            else:
                fingers.append(0)  # Thumb is closed

            # Check other fingers
            for tip_id in tip_ids[1:]:
                if landmarks[tip_id].y < landmarks[tip_id - 2].y:
                    fingers.append(1)  # Finger is open
                else:
                    fingers.append(0)  # Finger is closed

            # Count open fingers
            total_fingers = fingers.count(1)

            # Display finger count on frame
            cv2.putText(frame, f"Fingers: {total_fingers}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display frame in Streamlit
    FRAME_WINDOW.image(frame, channels="BGR")

cap.release()
