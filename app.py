# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import cv2
import numpy as np
import os
from PIL import Image
import tempfile

st.set_page_config(page_title="Creative Scoring Demo", layout="wide")
st.title("🎨 Creative Scoring Web App")

# --- Upload Section ---
uploaded_images = st.file_uploader("Upload creative images", type=["png", "jpg"], accept_multiple_files=True)
uploaded_csv = st.file_uploader("Upload performance CSV", type=["csv"])

# --- Compute visual properties ---
def compute_scores(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    image = cv2.resize(image, (400, 400))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 100, 200)
    edge_density = np.sum(edges > 0) / edges.size
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_count = len(contours)
    edge_score = 1 - min(edge_density * 5, 1)
    sharp_score = min(lap_var / 1000, 1)
    object_score = 1 - min(contour_count / 200, 1)
    clarity_score = (0.4 * edge_score + 0.3 * sharp_score + 0.3 * object_score)
    return image, clarity_score, edge_density, lap_var, contour_count

# --- Radar chart ---
def radar_plot(scores, labels, title):
    values = scores + [scores[0]]
    labels = labels + [labels[0]]
    angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False).tolist()
    
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'polar': True})
    ax.plot(angles, values, 'b-', linewidth=2)  # ✅ FIXED
    ax.fill(angles, values, 'skyblue', alpha=0.5)
    ax.set_thetagrids(np.degrees(angles), labels)
    ax.set_title(title, size=12)
    ax.grid(True)
    st.pyplot(fig)

# --- Saliency Map ---
def show_saliency_map(image, title):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
    saliency_map = cv2.magnitude(sobelx, sobely)
    saliency_map = np.uint8(255 * saliency_map / np.max(saliency_map))

    st.image([image[..., ::-1], saliency_map], caption=[title, "Approx. Saliency Map"], width=300)


# --- Color Distribution ---
def color_distribution(image):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape(-1, 3)
    df_colors = pd.DataFrame(pixels, columns=['R', 'G', 'B'])
    df_colors['hex'] = df_colors.apply(lambda row: '#%02x%02x%02x' % (row.R, row.G, row.B), axis=1)
    top_colors = df_colors['hex'].value_counts().head(5).reset_index()
    top_colors.columns = ['Color', 'Count']
    fig = px.bar(top_colors, x='Color', y='Count', color='Color', title="Top 5 Colors in Creative")
    fig.update_layout(showlegend=False)
    st.plotly_chart(fig, use_container_width=True)

# --- Analysis Section ---
if uploaded_images:
    st.subheader("🧠 Creative Analysis")
    for image_file in uploaded_images:
        st.markdown(f"### 🖼️ {image_file.name}")
        image, score, edge, sharp, contours = compute_scores(image_file)

        with st.expander("🔍 Summary"):
            st.write(f"• **Clarity Score**: {score:.2f}")
            st.write(f"• **Edge Density**: {edge:.3f}")
            st.write(f"• **Sharpness (Laplacian Var)**: {sharp:.2f}")
            st.write(f"• **Contour Count**: {contours}")

        cols = st.columns(3)

        with cols[0]:
            st.image(image[..., ::-1], caption="Original Creative", use_column_width=True)

        with cols[1]:
            st.markdown("**Radar Profile**")
            metrics = [score, edge, sharp / 1000, min(contours / 200, 1)]
            radar_plot(metrics, ["Clarity", "Edge", "Sharp", "Contours"], "Visual Profile")

        with cols[2]:
            st.markdown("**Saliency Map**")
            show_saliency_map(image, image_file.name)

        st.markdown("**🎨 Color Breakdown**")
        color_distribution(image)
        st.markdown("---")

# --- Performance CSV ---
if uploaded_csv:
    st.subheader("📊 Performance Data")
    df_perf = pd.read_csv(uploaded_csv)
    st.dataframe(df_perf)
    st.bar_chart(df_perf.select_dtypes(include=np.number))
