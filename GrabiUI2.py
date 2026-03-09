import streamlit as st
import os
import shutil
import tempfile
import numpy as np
import pandas as pd
from datetime import datetime
from enum import Enum
from PIL import Image

from MainQualitycheck import (
    EyeDetector,
    FocusDetector,
    IlluminationDetector,
    ReflectionDetector,
    CompletenessDetector,
    ResolutionDetector
)

# ---------------- ENUMS ---------------- #

class QualityState(Enum):
    YES = "Y"
    NO = "N"
    PARTIAL = "P"

class OverallQuality(Enum):
    BAD = "Bad Quality"
    USABLE = "Usable Quality"
    GOOD = "Good Quality"


# ---------------- FUNCTIONS ---------------- #

def map_quality_state(code: str) -> str:
    if code == QualityState.YES.value:
        return "Good"
    elif code == QualityState.NO.value:
        return "Bad"
    elif code == QualityState.PARTIAL.value:
        return "Partial"
    return "Unknown"


def compute_overall_quality(results: dict) -> OverallQuality:

    states = []

    for key, res in results.items():

        if "quality_state" in res:
            states.append(res["quality_state"])

        elif key == "Eye Presence":
            states.append("Y" if res.get("has_eye", False) else "N")

        elif key == "Illumination":
            states.append("Y" if res.get("lighting_correct", False) else "N")

    if any(s == "N" for s in states):
        return OverallQuality.BAD

    elif all(s == "Y" for s in states):
        return OverallQuality.GOOD

    else:
        return OverallQuality.USABLE


# ---------------- PAGE ---------------- #

st.set_page_config(
    page_title="Eye Image Quality Checker",
    page_icon="👁️",
)

st.title("👁️ Eye Image Quality Checker")

st.info("Upload an eye image or capture using camera.")


# ---------------- LOAD MODELS ---------------- #

@st.cache_resource
def load_detectors():

    eye_detector = EyeDetector(
        model_dir="./models/peakmodels"
    )

    focus_detector = FocusDetector(
        "./models/focus_svm_model.joblib",
        "./models/focus_scaler.joblib",
        "./models/focus_feature_names.txt",
    )

    illum_detector = IlluminationDetector(
        model_dir="./models"
    )

    refl_detector = ReflectionDetector(
        "./models/best_mobilevit_model.pth",
        device="cpu",
    )

    complete_detector = CompletenessDetector(
        "./models/resnet_completeness2.pth",
        "./models/xgboost_completeness2.json",
        device="cpu",
    )

    resolution_detector = ResolutionDetector(
        "./models/resnet_resolution.pth",
        "./models/xgboost_resolution_model.pkl",
        device="cpu",
    )

    return (
        eye_detector,
        focus_detector,
        illum_detector,
        refl_detector,
        complete_detector,
        resolution_detector,
    )


(
    eye_detector,
    focus_detector,
    illum_detector,
    refl_detector,
    complete_detector,
    resolution_detector,
) = load_detectors()


# ---------------- IMAGE INPUT ---------------- #

uploaded_file = st.file_uploader(
    "Upload an eye image",
    type=["jpg", "jpeg", "png"],
)

camera_file = st.camera_input("Or take a picture")

img_file = uploaded_file if uploaded_file else camera_file


# ---------------- IMAGE PROCESSING ---------------- #

if img_file is not None:

    temp_dir = tempfile.gettempdir()

    img_path = os.path.join(
        temp_dir,
        "temp_eye_image.jpg"
    )

    with open(img_path, "wb") as f:
        f.write(img_file.getbuffer())

    # Safe image display
    try:
        img = Image.open(img_path).convert("RGB")
        img_np = np.array(img)

        st.image(
            img_np,
            caption="Selected Image",
            use_container_width=True,
        )

    except Exception as e:
        st.error(f"Image display failed: {e}")
        st.stop()

    # ---------------- MODEL INFERENCE ---------------- #

    with st.spinner("🔍 Analyzing image quality..."):

        try:

            results = {

                "Eye Presence": eye_detector.predict(img_path),

                "Focus": focus_detector.predict(img_path),

                "Illumination": illum_detector.predict(img_path),

                "Reflection": refl_detector.predict(img_path),

                "Completeness": complete_detector.predict(img_path),

                "Resolution": resolution_detector.predict(img_path),

            }

        except Exception as e:
            st.error(f"Model inference failed: {e}")
            st.stop()

    # ---------------- RESULTS TABLE ---------------- #

    table_data = []

    eye_res = results["Eye Presence"]

    table_data.append([
        "👁️ Eye Presence",
        "Yes" if eye_res.get("has_eye") else "No",
        f"{eye_res.get('confidence', 0):.2f}"
    ])

    focus_res = results["Focus"]

    table_data.append([
        "📷 Focus",
        focus_res.get("prediction", "Unknown"),
        f"{focus_res.get('confidence', 0):.2f}"
    ])

    illum_res = results["Illumination"]

    table_data.append([
        "💡 Illumination",
        "Correct" if illum_res.get("lighting_correct") else "Incorrect",
        "-"
    ])

    refl_res = results["Reflection"]

    table_data.append([
        "✨ Reflection",
        map_quality_state(refl_res.get("quality_state")),
        f"{refl_res.get('confidence', 0):.2f}"
    ])

    comp_res = results["Completeness"]

    table_data.append([
        "⭕ Completeness",
        map_quality_state(comp_res.get("quality_state")),
        f"{comp_res.get('confidence', 0):.2f}"
    ])

    resol_res = results["Resolution"]

    table_data.append([
        "🖼️ Resolution",
        map_quality_state(resol_res.get("quality_state")),
        f"{resol_res.get('confidence', 0):.2f}"
    ])

    df = pd.DataFrame(
        table_data,
        columns=["Category", "Result", "Confidence"]
    )

    st.markdown("## 📊 Quality Check Results")

    st.table(df)


    # ---------------- OVERALL QUALITY ---------------- #

    overall = compute_overall_quality(results)

    st.markdown(
        f"### 🏆 Overall Quality: **{overall.value}**"
    )

    if overall == OverallQuality.BAD:

        st.warning(
            "⚠️ Image quality is bad. Please retake or upload a better image."
        )

        st.stop()

    # ---------------- SAVE GOOD IMAGE ---------------- #

    base_dir = os.path.dirname(
        os.path.abspath(__file__)
    )

    save_dir = os.path.join(
        base_dir,
        "saved_good_images"
    )

    os.makedirs(save_dir, exist_ok=True)

    file_name = f"quality_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"

    save_path = os.path.join(
        save_dir,
        file_name
    )

    try:

        shutil.copy(img_path, save_path)

        st.success(
            f"✅ Image saved successfully at `{save_path}`"
        )

    except Exception as e:

        st.warning(
            f"Image processed but could not save: {e}"
        )

else:

    st.info(
        "Please upload an image or capture using camera."
    )
