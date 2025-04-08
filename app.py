# app.py
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import numpy as np
import os
from PIL import Image
import tempfile

st.set_page_config(page_title="Creative Scoring Demo", layout="wide")

st.title("🎨 Creative Scoring Web App")

uploaded_images = st.file_uploader("Upload creative images", type=["png", "jpg"], accept_multiple_files=True)
uploaded_csv = st.file_uploader("Upload performance CSV", type=["csv"])

# Visual clarity score function
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
    return clarity_score, edge_density, lap_var, contour_count

# Radar chart
def radar_plot(scores, labels, title):
    values = scores + [scores[0]]
    labels = labels + [labels[0]]
    angles = np.linspace(0, 2 * np.pi, len(values), endpoint=False)
    fig, ax = plt.subplots(figsize=(5, 5), subplot_kw={'polar': True})
    ax.plot(angles, values, 'b-', linewidth=2)
    ax.fill(angles, values, 'skyblue', alpha=0.5)
    ax.set_thetagrids(angles * 180/np.pi, labels)
    ax.set_title(title, size=12)
    ax.grid(True)
    st.pyplot(fig)

if uploaded_images:
    st.subheader("🧠 Visual Profiles")
    for image_file in uploaded_images:
        score, edge, sharp, contours = compute_scores(image_file)
        metrics = [score, edge, sharp / 1000, min(contours / 200, 1)]
        radar_plot(metrics, ["Clarity", "Edge", "Sharp", "Contours"], image_file.name)

if uploaded_csv:
    st.subheader("📈 Performance Data")
    df_perf = pd.read_csv(uploaded_csv)
    st.dataframe(df_perf)

