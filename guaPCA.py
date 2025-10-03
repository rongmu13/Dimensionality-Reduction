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

        # ------------------ Execution ------------------
        if run:
            if cols_selected is None or len(cols_selected) < 2:
                st.error("Please select at least two numeric features.")
                st.stop()

            X = df[cols_selected].astype(float).copy()
            if do_standardize:
                X = StandardScaler().fit_transform(X)

            # safety for t-SNE perplexity
            if method == "t-SNE" and params["perplexity"] >= max(5, X.shape[0] - 1):
                st.warning(f"t-SNE perplexity must be < number of samples (now {X.shape[0]}). Auto-adjusted.")
                params["perplexity"] = max(5, min(30, X.shape[0]//3))

            try:
                info = {}
                if method == "PCA":
                    mdl = PCA(n_components=n_components, random_state=42)
                    emb = mdl.fit_transform(X)
                    info["explained_variance_ratio"] = mdl.explained_variance_ratio_[:min(n_components, len(mdl.explained_variance_ratio_))]
                    # loadings (coefficients for original variables on each PC)
                    # sklearn's components_: shape (n_components, n_features)
                    loadings = pd.DataFrame(
                        mdl.components_.T,
                        index=cols_selected,
                        columns=[f"PC{i+1}" for i in range(mdl.components_.shape[0])]
                    )
                elif method.startswith("LDA"):
                    if label_col is None:
                        st.error("LDA requires a label column.")
                        st.stop()
                    y = df[label_col].values
                    n_classes = len(np.unique(y))
                    n_comp_use = min(n_components, max(1, n_classes - 1))
                    mdl = LDA(n_components=n_comp_use)
                    emb = mdl.fit_transform(X, y)
                    if n_comp_use < n_components:
                        emb = np.pad(emb, ((0,0),(0, n_components - n_comp_use)), mode="constant")
                    info["classes"] = list(np.unique(y))
                elif method == "ICA":
                    mdl = FastICA(n_components=n_components, random_state=int(params["random_state"]), max_iter=int(params["max_iter"]))
                    emb = mdl.fit_transform(X)
                elif method == "Factor Analysis":
                    mdl = FactorAnalysis(n_components=n_components, random_state=int(params["random_state"]))
                    emb = mdl.fit_transform(X)
                elif "Kernel PCA" in method:
                    kp_kwargs = dict(kernel=params["kernel"])
                    if params["gamma"] > 0:
                        kp_kwargs["gamma"] = float(params["gamma"])
                    if params["kernel"] == "poly":
                        kp_kwargs["degree"] = int(params["degree"])
                    mdl = KernelPCA(n_components=n_components, **kp_kwargs, random_state=int(params["random_state"]))
                    emb = mdl.fit_transform(X)
                    info["kernel"] = params["kernel"]
                elif method == "t-SNE":
                    mdl = TSNE(n_components=n_components, perplexity=int(params["perplexity"]),
                               learning_rate=float(params["learning_rate"]), n_iter=int(params["n_iter"]),
                               init="pca" if n_components >= 2 else "random",
                               random_state=int(params["random_state"]))
                    emb = mdl.fit_transform(X)
                elif method == "UMAP":
                    mdl = UMAP(n_components=n_components, n_neighbors=int(params["n_neighbors"]),
                               min_dist=float(params["min_dist"]), metric=params["metric"],
                               random_state=int(params["random_state"]))
                    emb = mdl.fit_transform(X)
                elif method == "MDS":
                    mdl = MDS(n_components=n_components, n_init=int(params["n_init"]),
                              max_iter=int(params["max_iter"]), random_state=int(params["random_state"]))
                    emb = mdl.fit_transform(X)
                else:
                    st.error("Unknown method.")
                    st.stop()
            except Exception as e:
                st.error(f"Fitting failed: {e}")
                st.stop()

            st.success("Done!")

            # ---------- 2D scatter (scores) ----------
            if n_components >= 2:
                viz_df = pd.DataFrame(emb[:, :2], columns=["Dim1", "Dim2"])
                if color_col is not None:
                    viz_df[color_col] = df[color_col].astype(str).values
                elif method.startswith("LDA") and label_col is not None:
                    viz_df[label_col] = df[label_col].astype(str).values

                st.subheader("📈 2D Scatter (Scores)")
                if (color_col is not None) or (method.startswith("LDA") and label_col is not None):
                    color_key = color_col if color_col is not None else label_col
                    st.scatter_chart(viz_df, x="Dim1", y="Dim2", color=color_key, height=450)
                else:
                    st.scatter_chart(viz_df, x="Dim1", y="Dim2", height=450)

            # ---------- PCA-specific visuals ----------
            if method == "PCA":
                st.subheader("🧩 PCA Explained Variance Ratio")
                evr = info.get("explained_variance_ratio", [])
                evr_df = pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(evr))], "ExplainedVarianceRatio": evr})
                st.bar_chart(evr_df.set_index("PC"))

                # Loadings: coefficients of each original feature on PCs
                st.subheader("🧠 PCA Loadings (feature contributions)")
                st.caption("Bar charts for PC1 & PC2; full loadings table available for download.")

                # Show PC1 / PC2 bar charts if available
                try:
                    loadings  # noqa
                    if "PC1" in loadings.columns:
                        pc1_df = loadings["PC1"].reset_index()
                        pc1_df.columns = ["Feature", "Loading_PC1"]
                        st.markdown("**PC1 loadings**")
                        st.bar_chart(pc1_df.set_index("Feature"))

                    if "PC2" in loadings.columns:
                        pc2_df = loadings["PC2"].reset_index()
                        pc2_df.columns = ["Feature", "Loading_PC2"]
                        st.markdown("**PC2 loadings**")
                        st.bar_chart(pc2_df.set_index("Feature"))

                    # Show table preview
                    with st.expander("Show full loadings table"):
                        st.dataframe(loadings, use_container_width=True)

                except NameError:
                    pass

            # ---------- Downloads ----------
            st.subheader("⬇️ Downloads")

            # Embedding (scores) + original data
            out_cols = [f"Dim{i+1}" for i in range(n_components)]
            out_df = pd.DataFrame(emb[:, :n_components], columns=out_cols)
            out_df = pd.concat([df.reset_index(drop=True), out_df], axis=1)

            xlsx_buf = io.BytesIO()
            with pd.ExcelWriter(xlsx_buf, engine="openpyxl") as writer:
                out_df.to_excel(writer, index=False, sheet_name="Scores")

                # add PCA loadings sheet if available
                if method == "PCA":
                    try:
                        loadings.to_excel(writer, index=True, sheet_name="Loadings")
                        # Also include explained variance ratio
                        pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(evr))], "ExplainedVarianceRatio": evr}).to_excel(
                            writer, index=False, sheet_name="ExplainedVariance"
                        )
                    except Exception:
                        pass

            st.download_button(
                "Download Excel (original data + scores [+ PCA results])",
                data=xlsx_buf.getvalue(),
                file_name="dimensionality_reduction_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

            # Optional: CSVs
            csv_scores = out_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Scores (CSV)", data=csv_scores, file_name="scores.csv", mime="text/csv")

            if method == "PCA":
                try:
                    csv_loadings = loadings.reset_index().rename(columns={"index": "Feature"}).to_csv(index=False).encode("utf-8")
                    st.download_button("Download Loadings (CSV)", data=csv_loadings, file_name="pca_loadings.csv", mime="text/csv")
                except Exception:
                    pass

            # ---------- Info ----------
            with st.expander("🔍 Run info"):
                meta = {
                    "method": method,
                    "family": family,
                    "n_components": n_components,
                    "standardized": do_standardize,
                }
                if method == "PCA":
                    meta["explained_variance_ratio"] = list(evr)
                if "kernel" in info:
                    meta["kernel"] = info["kernel"]
                if "classes" in info:
                    meta["classes"] = info["classes"]
                st.json(meta)

