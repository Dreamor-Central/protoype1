import streamlit as st
import os
from backend_test import chat_with_multi_agent, process_image
from PIL import Image

st.set_page_config(page_title="Vasavi AI Assistant", page_icon="👗", layout="wide")

st.title("👗 Vasavi AI - Your Virtual Fashion Assistant")
st.write("Chat with Vasavi AI for fashion advice, trend insights, and outfit recommendations!")

# Sidebar for Image Upload
st.sidebar.header("Upload an Image")
uploaded_image = st.sidebar.file_uploader("Choose a fashion image...", type=["jpg", "jpeg", "png"])

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "👋 Hi! I’m Vasavi AI, your virtual stylist. How can I assist you today?"}]

# Display previous chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Process Image Input
if uploaded_image:
    st.sidebar.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
    st.sidebar.success("Processing image...")

    # Save image temporarily
    image_path = f"temp_{uploaded_image.name}"
    with open(image_path, "wb") as f:
        f.write(uploaded_image.getbuffer())

    # Get AI response for image
    response = process_image(image_path)

    # Add image response to chat
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").markdown(response)

    # Clean up saved image
    os.remove(image_path)

# Process Text Input
user_input = st.chat_input("Type your fashion query...")

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.chat_message("user").markdown(user_input)

    # Get AI response
    response = chat_with_multi_agent(user_input)

    # Display assistant response
    st.session_state.messages.append({"role": "assistant", "content": response})
    st.chat_message("assistant").markdown(response)
