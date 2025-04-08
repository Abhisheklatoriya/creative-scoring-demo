# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import io
import base64

st.set_page_config(page_title="Creative Scoring System", layout="wide")
st.title("🎯 Creative Scoring System")

# --- Sidebar Options ---
st.sidebar.header("Choose Insights to View")
show_profile = st.sidebar.checkbox("Visual Profile", value=True)
show_color = st.sidebar.checkbox("Color Distribution")
show_saliency = st.sidebar.checkbox("Simulated Saliency Map")
show_summary = st.sidebar.checkbox("Summary View", value=True)

# --- File Upload ---
uploaded_images = st.file_uploader("Upload Creative Images", type=["png", "jpg"], accept_multiple_files=True)

# --- Helper Functions ---
def get_color_distribution(image):
    img = image.resize((100, 100))
    pixels = np.array(img).reshape(-1, 3)
    df = pd.DataFrame(pixels, columns=["R", "G", "B"])
    df = df.astype(int)
    return df.mean()

def compute_visual_metrics(img):
    gray = img.convert("L")
    arr = np.array(gray)
    edges = np.abs(np.diff(arr, axis=0)).sum() + np.abs(np.diff(arr, axis=1)).sum()
    edge_density = edges / arr.size
    sharpness = arr.var()
    contour_approx = np.mean(np.abs(np.gradient(arr)))
    clarity_score = (0.4 * (1 - min(edge_density * 5, 1)) + 0.3 * min(sharpness / 1000, 1) + 0.3 * (1 - min(contour_approx / 20, 1)))
    return round(clarity_score, 2), round(edge_density, 4), round(sharpness, 2), round(contour_approx, 4)

def plot_radar(labels, values, title):
    labels = labels + [labels[0]]
    values = values + [values[0]]
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]
    fig, ax = plt.subplots(subplot_kw={'polar': True}, figsize=(4, 4))
    ax.plot(angles, values, 'r-', linewidth=2)
    ax.fill(angles, values, 'salmon', alpha=0.5)
    ax.set_thetagrids(np.degrees(angles), labels)
    ax.set_title(title)
    st.pyplot(fig)

def plot_color_bar(rgb_values, title):
    colors = ["Red", "Green", "Blue"]
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(colors, rgb_values, color=["red", "green", "blue"])
    ax.set_ylim(0, 255)
    ax.set_ylabel("Intensity")
    ax.set_title(title)
    st.pyplot(fig)

def show_saliency_simulation(img):
    gray = img.convert("L")
    sal = np.array(gray).astype(np.uint8)
    sal = np.clip(255 - sal, 0, 255)
    fig, ax = plt.subplots(1, 2, figsize=(8, 4))
    ax[0].imshow(img)
    ax[0].set_title("Creative")
    ax[0].axis("off")
    ax[1].imshow(sal, cmap='hot')
    ax[1].set_title("Simulated Saliency")
    ax[1].axis("off")
    st.pyplot(fig)

def get_strengths_weaknesses(score, edge, sharp, contours):
    strengths = []
    weaknesses = []
    if score > 0.75:
        strengths.append("High overall clarity")
    elif score < 0.5:
        weaknesses.append("Low clarity may reduce message clarity")

    if sharp > 200:
        strengths.append("Sharp image likely good for mobile")
    else:
        weaknesses.append("Image may appear soft on small screens")

    if edge < 0.1:
        strengths.append("Low visual noise")
    else:
        weaknesses.append("Potential visual clutter")

    if contours > 15:
        weaknesses.append("Many objects detected — layout may be crowded")
    else:
        strengths.append("Simple composition")

    return strengths, weaknesses

# --- Process Uploaded Images ---
if uploaded_images:
    for image_file in uploaded_images:
        st.markdown(f"### 🖼️ {image_file.name}")
        image = Image.open(image_file).convert("RGB")
        col1, col2 = st.columns([1, 2])

        with col1:
            st.image(image, caption="Uploaded Creative", use_column_width=True)

        with col2:
            if show_summary:
                score, edge, sharp, contours = compute_visual_metrics(image)

                # Color Heat Meter
                color = "green" if score > 0.75 else "orange" if score > 0.5 else "red"
                st.markdown(f"<div style='padding:10px; background:{color}; color:white; font-size:18px;'>\n                Visual Score: <strong>{score:.2f}</strong>\n                </div>", unsafe_allow_html=True)

                st.metric("📐 Edge Density", f"{edge}")
                st.metric("📈 Sharpness", f"{sharp}")
                st.metric("🔲 Contour Estimate", f"{contours}")

                # Media Context Translations
                st.markdown("#### 📋 Media Interpretation")
                st.markdown("- High clarity suggests better readability at a glance.")
                st.markdown("- Lower edge density indicates visual simplicity.")
                st.markdown("- Higher sharpness benefits display on mobile.")

                # Strengths & Weaknesses
                strengths, weaknesses = get_strengths_weaknesses(score, edge, sharp, contours)
                st.markdown("#### ✅ Top Strengths")
                for s in strengths:
                    st.markdown(f"- {s}")
                st.markdown("#### ⚠️ Areas to Improve")
                for w in weaknesses:
                    st.markdown(f"- {w}")

            if show_profile:
                metrics = compute_visual_metrics(image)
                plot_radar(["Clarity", "Edge", "Sharpness", "Contour"], list(metrics), "Visual Profile")

            if show_color:
                rgb_avg = get_color_distribution(image)
                plot_color_bar(rgb_avg.tolist(), "Average Color Composition")

            if show_saliency:
                show_saliency_simulation(image)

    st.success("✅ Analysis complete.")
else:
    st.info("⬆️ Please upload creative images to get started.")
