import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions

st.set_page_config(
    page_title="AI Vision",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        letter-spacing: -0.5px;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #94a3b8;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background-color: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1.5rem;
    }
    .top-class {
        font-size: 1.6rem;
        font-weight: 700;
        color: #38bdf8;
        text-transform: capitalize;
    }
    .top-confidence {
        font-size: 1.1rem;
        font-weight: 500;
        color: #e2e8f0;
    }
    .stat-label {
        font-size: 0.85rem;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

@st.cache_resource(show_spinner=False)
def load_classification_model():
    return MobileNetV2(weights="imagenet")

def process_and_predict(image, model):
    img = image.convert("RGB").resize((224, 224))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    preprocessed_img = preprocess_input(img_array)
    preds = model.predict(preprocessed_img, verbose=0)
    return decode_predictions(preds, top=5)[0]

with st.sidebar:
    st.markdown("### Model Information")
    st.markdown(
        """
        - **Architecture:** MobileNetV2
        - **Weights:** ImageNet-1k
        - **Input Resolution:** 224 × 224 px
        - **Framework:** TensorFlow / Keras
        - **Total Parameters:** ~3.5 Million
        - **Classes:** 1,000 Categories
        """
    )
    st.markdown("---")
    st.markdown(
        """
        **AI Vision** runs inference directly using MobileNetV2 pretrained on ImageNet without manual checkpoint management.
        """
    )

st.markdown('<div class="main-header">AI Vision</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Image classification powered by TensorFlow</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Choose an image to classify",
    type=["jpg", "jpeg", "png", "webp"],
    help="Supported formats: JPG, JPEG, PNG, WEBP"
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1], gap="large")
        
        with col1:
            st.subheader("Uploaded Image")
            st.image(image, use_container_width=True)
            st.caption(f"Original resolution: {image.size[0]} × {image.size[1]} px | Format: {image.format}")
            
        with col2:
            st.subheader("Classification Results")
            
            with st.spinner("Analyzing image..."):
                try:
                    model = load_classification_model()
                    predictions = process_and_predict(image, model)
                    
                    top_id, top_label, top_score = predictions[0]
                    formatted_top_label = top_label.replace("_", " ")
                    top_pct = top_score * 100
                    
                    st.markdown(
                        f"""
                        <div class="prediction-card">
                            <div class="stat-label">Primary Prediction</div>
                            <div class="top-class">{formatted_top_label}</div>
                            <div class="top-confidence">Confidence: {top_pct:.2f}%</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    
                    st.markdown("#### Top 5 Predictions")
                    for _, label, score in predictions:
                        clean_label = label.replace("_", " ").title()
                        pct = score * 100
                        col_label, col_pct = st.columns([3, 1])
                        with col_label:
                            st.write(f"**{clean_label}**")
                        with col_pct:
                            st.write(f"{pct:.2f}%")
                        st.progress(min(max(float(score), 0.0), 1.0))
                        
                except Exception:
                    st.error("An error occurred during inference. Please check that the uploaded file is a valid image.")
                    
    except Exception:
        st.error("Unable to load the specified image file. Please upload a valid JPG, PNG, or WEBP file.")
else:
    st.info("Upload an image above to get real-time classifications and confidence scores.")
