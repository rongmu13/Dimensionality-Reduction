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
    family = st.radio("Method family", ["Linear", "Nonlinear"], horizontal=True)

    if family == "Linear":
        method = st.selectbox(
            "Linear methods",
            ["PCA", "LDA (needs label column)", "ICA", "Factor Analysis", "Kernel PCA (linear kernel)"],
            index=0,
        )
    else:
        method = st.selectbox(
            "Nonlinear methods",
            ["t-SNE", "UMAP", "Kernel PCA (nonlinear)", "MDS"],
            index=0,
        )

    st.divider()
    st.subheader("📐 General parameters")
    n_components = st.number_input("Number of components (for export)", min_value=2, max_value=10, value=2, step=1)
    do_standardize = st.checkbox("Standardize numeric features (recommended)", value=True)

# ------------------ Utility: run & return results ------------------
def run_method(method, X, y=None, params=None):
    info, loadings, emb = {}, None, None
    if method == "PCA":
        mdl = PCA(n_components=n_components, random_state=42)
        emb = mdl.fit_transform(X)
        info["explained_variance_ratio"] = mdl.explained_variance_ratio_
        loadings = pd.DataFrame(
            mdl.components_.T,
            index=cols_selected,
            columns=[f"PC{i+1}" for i in range(mdl.components_.shape[0])]
        )
    elif method.startswith("LDA"):
        mdl = LDA(n_components=min(n_components, len(np.unique(y))-1))
        emb = mdl.fit_transform(X, y)
        info["classes"] = list(np.unique(y))
    elif method == "ICA":
        mdl = FastICA(n_components=n_components, random_state=42, max_iter=1000)
        emb = mdl.fit_transform(X)
        loadings = pd.DataFrame(
            mdl.components_.T,
            index=cols_selected,
            columns=[f"IC{i+1}" for i in range(mdl.components_.shape[0])]
        )
    elif method == "Factor Analysis":
        mdl = FactorAnalysis(n_components=n_components, random_state=42)
        emb = mdl.fit_transform(X)
        loadings = pd.DataFrame(
            mdl.components_.T,
            index=cols_selected,
            columns=[f"Factor{i+1}" for i in range(mdl.components_.shape[0])]
        )
    elif "Kernel PCA" in method:
        kernel = "rbf" if "nonlinear" in method else "linear"
        mdl = KernelPCA(n_components=n_components, kernel=kernel, random_state=42)
        emb = mdl.fit_transform(X)
        info["kernel"] = kernel
    elif method == "t-SNE":
        mdl = TSNE(n_components=n_components, random_state=42, perplexity=min(30, X.shape[0]-1))
        emb = mdl.fit_transform(X)
    elif method == "UMAP":
        mdl = UMAP(n_components=n_components, random_state=42)
        emb = mdl.fit_transform(X)
    elif method == "MDS":
        mdl = MDS(n_components=n_components, random_state=42)
        emb = mdl.fit_transform(X)
    return emb, info, loadings

# ------------------ Main area ------------------
left, right = st.columns([1.1, 1])
with left:
    st.subheader("📤 1) Upload Excel")
    up = st.file_uploader("Supported formats: .xlsx / .xls", type=["xlsx", "xls"])
    df, sheet_name_sel = None, None
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
        cols_selected = st.multiselect("Feature columns", numeric_cols, default=numeric_cols[:min(10,len(numeric_cols))])
        label_col, color_col = None, None
        if method.startswith("LDA"):
            cand = [c for c in df.columns if c not in cols_selected]
            label_col = st.selectbox("LDA label column", ["<None>"]+cand, index=0)
            if label_col=="<None>": label_col=None
        else:
            cand = [c for c in df.columns if c not in cols_selected]
            color_col = st.selectbox("Optional coloring column", ["<None>"]+cand, index=0)
            if color_col=="<None>": color_col=None

        st.divider()
        run = st.button("🚀 3) Run Dimensionality Reduction", use_container_width=True)

        if run:
            X = df[cols_selected].astype(float).copy()
            if do_standardize: X = StandardScaler().fit_transform(X)
            y = df[label_col] if label_col is not None else None

            emb, info, loadings = run_method(method, X, y)

            st.success("Finished!")

            # scatter plot
            if n_components>=2:
                viz = pd.DataFrame(emb[:,:2], columns=["Dim1","Dim2"])
                if color_col is not None: viz[color_col]=df[color_col].astype(str)
                elif label_col is not None: viz[label_col]=df[label_col].astype(str)
                st.subheader("📈 2D Scatter")
                st.scatter_chart(viz, x="Dim1", y="Dim2", color=viz.columns[-1] if viz.shape[1]>2 else None, height=450)

            # PCA / ICA / FA loadings
            if loadings is not None:
                st.subheader("📊 Component Loadings")
                st.dataframe(loadings, use_container_width=True)

            # PCA special
            if method=="PCA":
                evr=info.get("explained_variance_ratio",[])
                if len(evr)>0:
                    st.subheader("🧩 Explained Variance Ratio")
                    evr_df=pd.DataFrame({"PC":[f"PC{i+1}" for i in range(len(evr))],"ExplainedVariance":evr})
                    st.bar_chart(evr_df.set_index("PC"))

            # Downloads
            st.subheader("⬇️ Downloads")
            out_cols=[f"Dim{i+1}" for i in range(n_components)]
            out_df=pd.concat([df.reset_index(drop=True), pd.DataFrame(emb[:,:n_components],columns=out_cols)],axis=1)

            buf=io.BytesIO()
            with pd.ExcelWriter(buf,engine="openpyxl") as writer:
                out_df.to_excel(writer,index=False,sheet_name="Scores")
                if loadings is not None: loadings.to_excel(writer,sheet_name="Loadings")
                if method=="PCA":
                    pd.DataFrame({"PC":[f"PC{i+1}" for i in range(len(evr))],"ExplainedVariance":evr}).to_excel(writer,sheet_name="ExplainedVariance",index=False)

            st.download_button("Download Excel",data=buf.getvalue(),file_name="results.xlsx",mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            csv_scores=out_df.to_csv(index=False).encode("utf-8")
            st.download_button("Download Scores (CSV)",data=csv_scores,file_name="scores.csv",mime="text/csv")

            if loadings is not None:
                csv_loadings=loadings.reset_index().to_csv(index=False).encode("utf-8")
                st.download_button("Download Loadings (CSV)",data=csv_loadings,file_name="loadings.csv",mime="text/csv")

            with st.expander("🔍 Run info"):
                st.json({"method":method,"family":family,"n_components":n_components,"standardized":do_standardize,**info})

# ------------------ Footer ------------------
st.divider()
st.caption("For research/teaching purposes only")
