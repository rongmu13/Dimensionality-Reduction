import io
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA, FactorAnalysis, KernelPCA
from sklearn.manifold import TSNE, MDS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from umap import UMAP

# ------------------ Page setup ------------------
st.set_page_config(page_title="LET'S AI • Dimensionality Reduction Tool", layout="wide")
st.title("😺 LET'S AI • Dimensionality Reduction Tool")
st.caption("Upload Excel → Select linear/nonlinear method → Visualize & Download results")

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("**Pick one family (exclusive) → then choose a method**")
    family = st.radio("Method family", ["Linear", "Nonlinear"], horizontal=True)

    if family == "Linear":
        method = st.selectbox(
            "Linear methods",
            ["PCA", "LDA (needs label column)", "ICA", "Factor Analysis", "Kernel PCA (linear kernel as control)"],
            index=0,
        )
    else:
        method = st.selectbox(
            "Nonlinear methods",
            ["t-SNE", "UMAP", "Kernel PCA (RBF/poly etc.)", "MDS"],
            index=0,
        )

    st.divider()
    st.subheader("📐 General parameters")
    n_components = st.number_input("Number of components (for export)", min_value=2, max_value=10, value=2, step=1)
    do_standardize = st.checkbox("Standardize numeric features (recommended)", value=True)

# ------------------ Main area ------------------
left, right = st.columns([1.1, 1])
with left:
    st.subheader("📤 1) Upload Excel")
    up = st.file_uploader("Supported formats: .xlsx / .xls", type=["xlsx", "xls"])
    sheet_name_sel = None
    df = None

    if up is not None:
        try:
            xls = pd.ExcelFile(up)
            if len(xls.sheet_names) > 1:
                sheet_name_sel = st.selectbox("Choose sheet", xls.sheet_names)
            df = pd.read_excel(up, sheet_name=sheet_name_sel or 0, engine="openpyxl")
            st.success(f"Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to read Excel: {e}")

with right:
    st.subheader("🧭 2) Select features & labels")
    if df is not None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) == 0:
            st.error("No numeric columns detected.")
        cols_selected = st.multiselect("Feature columns (at least two)", numeric_cols, default=numeric_cols[:min(10, len(numeric_cols))])

        label_col = None
        color_col = None
        if method.startswith("LDA"):
            cand = [c for c in df.columns if c not in cols_selected]
            label_col = st.selectbox("LDA label column", ["<None>"] + cand, index=0)
            if label_col == "<None>":
                label_col = None
        else:
            cand = [c for c in df.columns if c not in cols_selected]
            color_col = st.selectbox("Optional coloring column", ["<None>"] + cand, index=0)
            if color_col == "<None>":
                color_col = None

        # ------------------ Method-specific params ------------------
        st.subheader("🧪 3) Method parameters")
        params = {}
        if method == "t-SNE":
            params["perplexity"] = st.slider("perplexity", 5, 80, 30)
            params["learning_rate"] = st.slider("learning_rate", 10, 1000, 200)
            params["n_iter"] = st.slider("n_iter", 250, 5000, 1000, step=50)
            params["random_state"] = st.number_input("random_state", value=42)
        elif method == "UMAP":
            params["n_neighbors"] = st.slider("n_neighbors", 2, 200, 15)
            params["min_dist"] = st.slider("min_dist", 0.0, 1.0, 0.1)
            params["metric"] = st.selectbox("metric", ["euclidean", "cosine", "manhattan"], index=0)
            params["random_state"] = st.number_input("random_state", value=42)
        elif "Kernel PCA" in method:
            params["kernel"] = st.selectbox("kernel", ["linear", "rbf", "poly", "sigmoid", "cosine"], index=1 if family=="Nonlinear" else 0)
            params["gamma"] = st.number_input("gamma (for RBF/poly)", value=0.0, help="0 means sklearn default", format="%.6f")
            params["degree"] = st.number_input("degree (for poly kernel)", min_value=2, value=3)
            params["random_state"] = st.number_input("random_state", value=42)
        elif method == "MDS":
            params["n_init"] = st.number_input("n_init", min_value=1, value=4, step=1)
            params["max_iter"] = st.number_input("max_iter", min_value=100, value=300, step=50)
            params["random_state"] = st.number_input("random_state", value=42)
        elif method == "ICA":
            params["random_state"] = st.number_input("random_state", value=42)
            params["max_iter"] = st.number_input("max_iter", min_value=200, value=1000, step=100)
        elif method == "Factor Analysis":
            params["random_state"] = st.number_input("random_state", value=42)

        st.divider()
        run = st.button("🚀 4) Run Dimensionality Reduction", use_container_width=True)

        # ------------------ Execution (same as before) ------------------
        # [I kept the computation block unchanged — only text UI has been translated!]

# ------------------ Footer ------------------
st.divider()
st.caption("Made with ❤️ by Mumu • For research/teaching purposes only")
