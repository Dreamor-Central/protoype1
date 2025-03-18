import streamlit as st
import os
from backendex import chat_with_multi_agent, process_image
from PIL import Image

# Page configuration
st.set_page_config(
    page_title="Vasavi AI Assistant", 
    page_icon="👗", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-family: 'Helvetica Neue', sans-serif;
        color: #FF4B4B;
    }
    .stChat message {
        border-radius: 15px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .user-message {
        background-color: #F0F2F6;
    }
    .assistant-message {
        background-color: #FFE4E4;
    }
</style>
""", unsafe_allow_html=True)

# Header section
st.markdown("<h1 class='main-header'>👗 Vasavi AI - Your Virtual Fashion Assistant</h1>", unsafe_allow_html=True)
st.markdown("#### Chat with Vasavi AI for fashion advice, trend insights, and outfit recommendations!")

# Create columns for layout
col1, col2 = st.columns([1, 2])

# Sidebar for image upload
with col1:
    st.subheader("Upload an Image")
    uploaded_image = st.file_uploader("Choose a fashion image...", type=["jpg", "jpeg", "png"])
    
    # Display uploaded image
    if uploaded_image:
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)
        st.success("Image uploaded successfully")

# Main chat area
with col2:
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hi! I'm Vasavi AI, your virtual stylist. How can I assist you today?"}
        ]
    
    # Display chat messages container
    chat_container = st.container()
    
    # Create input area at the bottom
    user_input = st.chat_input("Type your fashion query...")
    
    # Display previous chat messages
    with chat_container:
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
    
    # Process image input
    if uploaded_image and "last_processed_image" not in st.session_state:
        st.session_state.last_processed_image = uploaded_image.name
        
        # Save image temporarily
        image_path = f"temp_{uploaded_image.name}"
        with open(image_path, "wb") as f:
            f.write(uploaded_image.getbuffer())
        
        try:
            # Processing indicator
            with st.spinner("Analyzing your fashion image..."):
                # Get AI response for image
                response = process_image(image_path)
                
                # Add image response to chat
                st.session_state.messages.append({"role": "assistant", "content": response})
                
                # Force UI refresh
                with chat_container:
                    with st.chat_message("assistant"):
                        st.markdown(response)
        
        except Exception as e:
            error_msg = f"Error processing image: {str(e)}"
            st.error(error_msg)
        
        finally:
            # Clean up saved image
            if os.path.exists(image_path):
                os.remove(image_path)
    
    # Clear image processing flag when new image is uploaded
    elif uploaded_image and st.session_state.get("last_processed_image") != uploaded_image.name:
        st.session_state.last_processed_image = None
    
    # Process text input
    if user_input:
        # Display user message
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        # Force UI refresh for user message
        with chat_container:
            with st.chat_message("user"):
                st.markdown(user_input)
        
        try:
            # Processing indicator
            with st.spinner("Thinking..."):
                # Get AI response with error handling
                response = chat_with_multi_agent(user_input)
            
            # Display assistant response
            st.session_state.messages.append({"role": "assistant", "content": response})
            
            # Force UI refresh for assistant response
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown(response)
        
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            st.error(error_msg)
            
            st.session_state.messages.append({
                "role": "assistant", 
                "content": "I'm sorry, I encountered an error processing your request."
            })
            
            with chat_container:
                with st.chat_message("assistant"):
                    st.markdown("I'm sorry, I encountered an error processing your request.")

# Footer
st.markdown("---")
st.markdown("Visit [Vasavi.co](https://vasavi.co/) | Follow [@vasavi.co](https://instagram.com/vasavi.co) on Instagram | Contact: support@vasavi.co")