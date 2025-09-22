# ui_quality_check.py
import streamlit as st
import os
from MainQualitycheck import EyeDetector, FocusDetector, IlluminationDetector, ReflectionDetector, CompletenessDetector

st.title("👁️ Eye Image Quality Checker")

uploaded_file = st.file_uploader("Upload an eye image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img_path = os.path.join("temp_uploaded.jpg")
    with open(img_path, "wb") as f:
        f.write(uploaded_file.read())

    st.image(img_path, caption="Uploaded Image", use_column_width=True)

    # Load detectors (update paths with your model dirs)
    eye_detector = EyeDetector(model_dir="./models/peakmodels")
    focus_detector = FocusDetector(
    "./models/focus_svm_model.joblib",
    "./models/focus_scaler.joblib",
    "./models/focus_feature_names.txt"
)
    illum_detector = IlluminationDetector(model_dir="./models")
    refl_detector = ReflectionDetector("./models/best_mobilevit_model.pth", device="cpu")

    complete_detector = CompletenessDetector(
    "./models/resnet_completeness2.pth",
    "./models/xgboost_completeness2.json",
    device="cpu"
)

    # Run predictions
    with st.spinner("Analyzing..."):
        eye_result = eye_detector.predict(img_path)
        focus_result = focus_detector.predict(img_path)
        illum_result = illum_detector.predict(img_path)
        refl_result = refl_detector.predict(img_path)
        comp_result = complete_detector.predict(img_path)

    st.subheader("🔍 Results")
    st.write("👁️ Eye Presence:", eye_result)
    st.write("📷 Focus:", focus_result)
    st.write("💡 Illumination:", illum_result)
    st.write("✨ Reflection:", refl_result)
    st.write("⭕ Completeness:", comp_result)
