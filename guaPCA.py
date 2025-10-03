import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from matplotlib.patches import Ellipse
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA, FactorAnalysis, KernelPCA
from sklearn.manifold import TSNE, MDS, trustworthiness
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, pairwise_distances
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from umap import UMAP

# ------------------ Page setup ------------------
st.set_page_config(page_title="LET'S AI • Dimensionality Reduction Tool", layout="wide")
st.title("😺 LET'S AI • Dimensionality Reduction Tool")
st.caption("Upload Excel → Pick linear/nonlinear method → Visualize → Download results")

# ------------------ Helpers ------------------
def sanitize_columns(df: pd.DataFrame):
    """Make all column names strings and unique (sklearn >=1.4 feature-name checks)."""
    raw_cols = list(df.columns)
    new_cols = [str(c) for c in raw_cols]
    seen, unique_cols = set(), []
    for c in new_cols:
        base, k, name = c, 1, c
        while name in seen:
            name = f"{base}__{k}"
            k += 1
        seen.add(name); unique_cols.append(name)
    out = df.copy()
    out.columns = unique_cols
    mapping = {raw: new for raw, new in zip(raw_cols, unique_cols) if str(raw) != new}
    return out, mapping

def pca_biplot(scores, model, feature_names, hue=None, hue_name=None, fig_size=(7.5, 5.5), dpi=160):
    """Draw PCA biplot: scatter + loading vectors (assumes X standardized)."""
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)

    if hue is None:
        ax.scatter(scores[:, 0], scores[:, 1], s=18, alpha=0.85)
    else:
        cats = pd.Series(hue).astype(str)
        colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)
        for i, lv in enumerate(cats.unique()):
            m = (cats == lv).values
            ax.scatter(scores[m, 0], scores[m, 1], s=18, alpha=0.85,
                       label=lv, c=None if colors is None else colors[i % len(colors)])
        if cats.nunique() <= 20:
            ax.legend(title=hue_name, frameon=False, loc="best")

    comps = model.components_.T[:, :2]  # (n_features, 2)
    L = comps * np.sqrt(model.explained_variance_[:2])  # scale by variance
    max_scores = np.max(np.abs(scores[:, :2]))
    max_vec = np.max(np.abs(L)) if np.max(np.abs(L)) > 0 else 1.0
    L_scaled = L / max_vec * (max_scores * 0.9)

    for i, (x, y) in enumerate(L_scaled):
        ax.arrow(0, 0, x, y, color="black", lw=1.1,
                 head_width=0.02 * max_scores, length_includes_head=True)
        ax.text(x * 1.06, y * 1.06, str(feature_names[i]), fontsize=9,
                ha="center", va="center")

    ax.axhline(0, color="#888", lw=0.5); ax.axvline(0, color="#888", lw=0.5)
    ax.set_xlabel(f"PC1 ({model.explained_variance_ratio_[0]*100:.2f}%)")
    ax.set_ylabel(f"PC2 ({model.explained_variance_ratio_[1]*100:.2f}%)")
    ax.set_title("PCA Biplot (scores + variable loadings)")
    fig.tight_layout()
    return fig

def plot_class_ellipse(ax, X2, label, color=None):
    """95% ellipse for class scatter in 2D."""
    mu = X2.mean(axis=0)
    cov = np.cov(X2.T)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    theta = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    width, height = 2.4477 * np.sqrt(vals)  # ~95% chi-square in 2D
    ell = Ellipse(xy=mu, width=width, height=height, angle=theta,
                  edgecolor=color or "black", facecolor="none", lw=1.2, alpha=0.9)
    ax.add_patch(ell)
    ax.scatter(*mu, c=color or "black", s=45, marker="x", linewidths=1.5, zorder=5)

def heatmap_loadings(loadings_df, title="Component Loadings", cmap="viridis", fig_size=(7.5, 5.5), dpi=160):
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    im = ax.imshow(loadings_df.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(loadings_df.shape[1])); ax.set_xticklabels(loadings_df.columns, rotation=45, ha="right")
    ax.set_yticks(range(loadings_df.shape[0])); ax.set_yticklabels(loadings_df.index)
    ax.set_title(title); fig.colorbar(im, ax=ax, shrink=0.85)
    fig.tight_layout()
    return fig

def shepard_diagram(X_high, X_low, fig_size=(6.8, 5.2), dpi=160):
    Dh = pairwise_distances(X_high)
    Dl = pairwise_distances(X_low)
    tri = np.triu_indices_from(Dh, k=1)
    x = Dh[tri].reshape(-1, 1)
    y = Dl[tri]
    reg = LinearRegression().fit(x, y)
    r2 = reg.score(x, y)
    xs = np.linspace(x.min(), x.max(), 200).reshape(-1, 1)
    fig, ax = plt.subplots(figsize=fig_size, dpi=dpi)
    ax.scatter(x, y, s=6, alpha=0.35)
    ax.plot(xs, reg.predict(xs), lw=2)
    ax.set_xlabel("High-dimensional distance")
    ax.set_ylabel("Low-dimensional distance")
    ax.set_title(f"Shepard Diagram (R² = {r2:.3f})")
    fig.tight_layout()
    return fig, r2

# ------------------ Sidebar ------------------
with st.sidebar:
    st.header("⚙️ Settings")
    st.markdown("**Pick one family (exclusive) → then choose a method**")
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
    dropna = st.checkbox("Drop rows with missing values in selected columns", value=True)

# ------------------ Layout ------------------
left, right = st.columns([1.1, 1])

# ------------------ Upload ------------------
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

            # Normalize column names for sklearn compatibility
            df, renamed = sanitize_columns(df)
            if renamed:
                with st.expander("Column names normalized (original → new)"):
                    st.write(renamed)

            st.success(f"Data loaded: {df.shape[0]} rows × {df.shape[1]} columns")
            st.dataframe(df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"Failed to read Excel: {e}")

# ------------------ Feature / label selection ------------------
with right:
    st.subheader("🧭 2) Select features & labels")
    if df is not None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) == 0:
            st.error("No numeric columns detected.")
        default_pick = numeric_cols[:min(10, len(numeric_cols))]
        cols_selected = st.multiselect("Feature columns (at least two)", numeric_cols, default=default_pick)

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

        st.divider()
        run = st.button("🚀 3) Run Dimensionality Reduction", use_container_width=True)

        # --------------- Run ----------------
        if run:
            if cols_selected is None or len(cols_selected) < 2:
                st.error("Please select at least two numeric features.")
                st.stop()

            # Drop rows with NA in selected columns (and label if needed)
            df_used = df.copy()
            if dropna:
                subset = cols_selected + ([label_col] if (label_col is not None) else [])
                df_used = df_used.dropna(subset=subset).reset_index(drop=True)

            Xraw = df_used[cols_selected]
            try:
                X = Xraw.astype(float).values
            except Exception:
                st.error("Selected features include non-numeric values that cannot be cast to float.")
                st.stop()

            y = df_used[label_col].astype(str).values if label_col is not None else None

            if do_standardize:
                X = StandardScaler().fit_transform(X)

            # Fit per method
            info = {}
            loadings = None
            emb = None
            mdl = None

            try:
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
                    if label_col is None:
                        st.error("LDA requires a label column.")
                        st.stop()
                    n_classes = len(np.unique(y))
                    n_comp_use = min(n_components, max(1, n_classes - 1))
                    mdl = LDA(n_components=n_comp_use)
                    emb = mdl.fit_transform(X, y)
                    if n_comp_use < n_components:
                        emb = np.pad(emb, ((0, 0), (0, n_components - n_comp_use)), mode="constant")
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
                elif method == "Kernel PCA (linear kernel)":
                    mdl = KernelPCA(n_components=n_components, kernel="linear", random_state=42)
                    emb = mdl.fit_transform(X)
                    info["kernel"] = "linear"
                elif method == "Kernel PCA (nonlinear)":
                    mdl = KernelPCA(n_components=n_components, kernel="rbf", random_state=42)
                    emb = mdl.fit_transform(X)
                    info["kernel"] = "rbf"
                elif method == "t-SNE":
                    p = max(5, min(30, X.shape[0] - 1))
                    mdl = TSNE(n_components=n_components, perplexity=p, random_state=42, n_iter=1000, learning_rate=200, init="pca")
                    emb = mdl.fit_transform(X)
                elif method == "UMAP":
                    mdl = UMAP(n_components=n_components, random_state=42)
                    emb = mdl.fit_transform(X)
                elif method == "MDS":
                    mdl = MDS(n_components=n_components, random_state=42)
                    emb = mdl.fit_transform(X)
                else:
                    st.error("Unknown method.")
                    st.stop()
            except Exception as e:
                st.error(f"Fitting failed: {e}")
                st.stop()

            st.success("Finished!")

            # ------------ 2D scatter (scores) -------------
            if n_components >= 2:
                st.subheader("📈 2D Scatter (Scores)")
                viz_df = pd.DataFrame(emb[:, :2], columns=["Dim1", "Dim2"])
                if color_col is not None:
                    viz_df[color_col] = df_used[color_col].astype(str).values
                    st.scatter_chart(viz_df, x="Dim1", y="Dim2", color=color_col, height=450)
                elif label_col is not None and method.startswith("LDA"):
                    viz_df[label_col] = df_used[label_col].astype(str).values
                    st.scatter_chart(viz_df, x="Dim1", y="Dim2", color=label_col, height=450)
                else:
                    st.scatter_chart(viz_df, x="Dim1", y="Dim2", height=450)

            # ------------ Method-specific visuals -------------
            metrics = {}

            # PCA extras
            if method == "PCA":
                # explained variance
                evr = info.get("explained_variance_ratio", [])
                if len(evr) > 0:
                    st.subheader("🧩 PCA Explained Variance Ratio")
                    evr_df = pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(evr))],
                                           "ExplainedVariance": evr})
                    st.bar_chart(evr_df.set_index("PC"))
                # biplot
                st.subheader("🧭 PCA Biplot (scores + loadings)")
                if n_components >= 2:
                    hue_series, hue_name = None, None
                    if color_col is not None:
                        hue_series = df_used[color_col]; hue_name = color_col
                    fig_bi = pca_biplot(emb[:, :2], mdl, cols_selected, hue=hue_series, hue_name=hue_name)
                    st.pyplot(fig_bi, use_container_width=True)
                # loadings table
                if loadings is not None:
                    with st.expander("Show full loadings table"):
                        st.dataframe(loadings, use_container_width=True)

            # LDA extras: class ellipses + optional confusion matrix
            if method.startswith("LDA") and n_components >= 2:
                st.subheader("🎯 LDA scores with class ellipses")
                fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=160)
                Z = emb[:, :2]
                y_show = df_used[label_col].astype(str).values
                classes = pd.Series(y_show).unique()
                colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', None)
                for i, c in enumerate(classes):
                    m = (y_show == c)
                    ax.scatter(Z[m, 0], Z[m, 1], s=18, alpha=0.85,
                               label=c, c=None if colors is None else colors[i % len(colors)])
                    plot_class_ellipse(ax, Z[m, :], c, color=None if colors is None else colors[i % len(colors)])
                ax.axhline(0, color="#aaa", lw=0.5); ax.axvline(0, color="#aaa", lw=0.5)
                ax.set_xlabel("LD1"); ax.set_ylabel("LD2"); ax.legend(frameon=False)
                ax.set_title("LDA Scores with 95% Ellipses")
                st.pyplot(fig, use_container_width=True)

                with st.expander("Quick confusion matrix (80/20 hold-out)"):
                    Xtmp = Xraw.astype(float).values
                    if do_standardize:
                        Xtmp = StandardScaler().fit_transform(Xtmp)
                    Xtr, Xte, ytr, yte = train_test_split(Xtmp, y_show, test_size=0.2, random_state=42, stratify=y_show)
                    lda_tmp = LDA(n_components=min(2, len(np.unique(y_show)) - 1)).fit(Xtr, ytr)
                    ypred = lda_tmp.predict(Xte)
                    cm = confusion_matrix(yte, ypred, labels=classes)
                    fig2, ax2 = plt.subplots(dpi=160)
                    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
                    disp.plot(ax=ax2, cmap="Blues", colorbar=False)
                    ax2.set_title("Confusion matrix (hold-out)")
                    st.pyplot(fig2, use_container_width=True)

            # ICA / FA: loadings heatmap (+ communalities for FA)
            if method == "ICA" and loadings is not None:
                st.subheader("🧠 ICA component loadings (heatmap)")
                fig_h = heatmap_loadings(loadings, title="ICA Loadings (features × ICs)")
                st.pyplot(fig_h, use_container_width=True)

            if method == "Factor Analysis" and loadings is not None:
                st.subheader("🧠 Factor loadings (heatmap)")
                fig_f = heatmap_loadings(loadings, title="Factor Loadings (features × factors)", cmap="magma")
                st.pyplot(fig_f, use_container_width=True)
                with st.expander("Show communalities"):
                    comm = (loadings.values ** 2).sum(axis=1)
                    figc, axc = plt.subplots(dpi=160)
                    axc.bar(loadings.index, comm)
                    axc.set_title("Communalities (sum of squared loadings)")
                    axc.tick_params(axis='x', rotation=45)
                    st.pyplot(figc, use_container_width=True)

            # Trustworthiness for manifold methods / Kernel PCA
            if method in ["t-SNE", "UMAP", "Kernel PCA (nonlinear)", "Kernel PCA (linear kernel)"]:
                try:
                    tw = trustworthiness(Xraw.astype(float).values, emb[:, :2], n_neighbors=5)
                    metrics["trustworthiness_k5"] = float(tw)
                    st.caption(f"Trustworthiness (k=5): {tw:.3f}")
                except Exception:
                    pass

            # MDS: Shepard diagram + stress
            if method == "MDS":
                try:
                    fig_shep, r2 = shepard_diagram(Xraw.astype(float).values, emb[:, :2])
                    st.subheader("📐 Shepard Diagram")
                    st.pyplot(fig_shep, use_container_width=True)
                    metrics["shepard_R2"] = float(r2)
                except Exception:
                    pass
                try:
                    Dh = pairwise_distances(Xraw.astype(float).values)
                    Dl = pairwise_distances(emb[:, :2])
                    tri = np.triu_indices_from(Dh, k=1)
                    stress = np.sqrt(((Dh[tri] - Dl[tri]) ** 2).sum() / (Dh[tri] ** 2).sum())
                    metrics["stress1_approx"] = float(stress)
                    st.caption(f"Kruskal stress-1 (approx): {stress:.3f}")
                except Exception:
                    pass

            # ------------ Downloads -------------
            st.subheader("⬇️ Downloads")
            out_cols = [f"Dim{i+1}" for i in range(n_components)]
            scores_df = pd.DataFrame(emb[:, :n_components], columns=out_cols)
            export_df = pd.concat([df_used.reset_index(drop=True), scores_df], axis=1)

            xbuf = io.BytesIO()
            with pd.ExcelWriter(xbuf, engine="openpyxl") as writer:
                export_df.to_excel(writer, index=False, sheet_name="Scores")
                if loadings is not None:
                    loadings.to_excel(writer, sheet_name="Loadings")
                if method == "PCA":
                    evr = info.get("explained_variance_ratio", [])
                    if len(evr) > 0:
                        pd.DataFrame({"PC": [f"PC{i+1}" for i in range(len(evr))],
                                      "ExplainedVariance": evr}).to_excel(writer, index=False, sheet_name="ExplainedVariance")
                if len(metrics) > 0:
                    pd.DataFrame([metrics]).to_excel(writer, index=False, sheet_name="Metrics")

            st.download_button(
                "Download Excel (Scores [+ Loadings / Metrics])",
                data=xbuf.getvalue(),
                file_name="dimensionality_reduction_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

            st.download_button(
                "Download Scores (CSV)",
                data=export_df.to_csv(index=False).encode("utf-8"),
                file_name="scores.csv",
                mime="text/csv"
            )

            if loadings is not None:
                st.download_button(
                    "Download Loadings (CSV)",
                    data=loadings.reset_index().to_csv(index=False).encode("utf-8"),
                    file_name="loadings.csv",
                    mime="text/csv"
                )

            with st.expander("🔍 Run info"):
                meta = {"method": method, "family": family, "n_components": n_components,
                        "standardized": do_standardize, **info, **metrics}
                st.json(meta)

# ------------------ Footer ------------------
st.divider()
st.caption("Made with ❤️ by Mumu • For research/teaching purposes only")
