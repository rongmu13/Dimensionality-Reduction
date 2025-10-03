import io
import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, FastICA, FactorAnalysis, KernelPCA
from sklearn.manifold import TSNE, MDS
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from umap import UMAP

# ------------------ 页面基本设置 ------------------
st.set_page_config(page_title="LET'S AI • 降维小工具", layout="wide")
st.title("😺 LET'S AI • 降维小工具")
st.caption("上传 Excel → 选择线性/非线性方法 → 一键可视化并下载结果")

# ------------------ 侧边栏：方法选择 ------------------
with st.sidebar:
    st.header("⚙️ 设置")
    st.markdown("**先选类别（只能选其一）→ 再选具体方法**")
    family = st.radio("方法类别", ["线性 (Linear)", "非线性 (Nonlinear)"], horizontal=True)

    if family == "线性 (Linear)":
        method = st.selectbox(
            "线性方法",
            ["PCA", "LDA（需标签列）", "ICA", "Factor Analysis", "Kernel PCA（线性核当作线性对照）"],
            index=0,
        )
    else:
        method = st.selectbox(
            "非线性方法",
            ["t-SNE", "UMAP", "Kernel PCA（RBF/多项式等）", "MDS"],
            index=0,
        )

    st.divider()
    st.subheader("📐 通用参数")
    n_components = st.number_input("降维到几个维度（用于下载）", min_value=2, max_value=10, value=2, step=1)
    do_standardize = st.checkbox("标准化数值特征（建议开启）", value=True)

# ------------------ 主区：数据上传 ------------------
left, right = st.columns([1.1, 1])
with left:
    st.subheader("📤 1) 上传 Excel")
    up = st.file_uploader("支持 .xlsx / .xls", type=["xlsx", "xls"])
    sheet_name_sel = None
    df = None

    if up is not None:
        try:
            xls = pd.ExcelFile(up)
            if len(xls.sheet_names) > 1:
                sheet_name_sel = st.selectbox("选择工作表 (Sheet)", xls.sheet_names)
            df = pd.read_excel(up, sheet_name=sheet_name_sel or 0, engine="openpyxl")
            st.success(f"已读取数据，共 {df.shape[0]} 行 × {df.shape[1]} 列")
            st.dataframe(df.head(20), use_container_width=True)
        except Exception as e:
            st.error(f"读取失败：{e}")

with right:
    st.subheader("🧭 2) 选择列 & 标签")
    if df is not None:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
        if len(numeric_cols) == 0:
            st.error("未检测到数值列，请检查 Excel。")
        cols_selected = st.multiselect("作为特征的数值列（至少两列）", numeric_cols, default=numeric_cols[:min(10, len(numeric_cols))])

        label_col = None
        color_col = None
        # LDA 需要标签列；其他方法可选一个颜色列用于着色
        if method.startswith("LDA"):
            cand = [c for c in df.columns if c not in cols_selected]
            label_col = st.selectbox("LDA 标签列（分类标签）", ["<未选择>"] + cand, index=0)
            if label_col == "<未选择>":
                label_col = None
        else:
            cand = [c for c in df.columns if c not in cols_selected]
            color_col = st.selectbox("可选：着色列（分类或分组）", ["<不着色>"] + cand, index=0)
            if color_col == "<不着色>":
                color_col = None

        # ------------------ 参数区（按方法显示） ------------------
        st.subheader("🧪 3) 方法参数")
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
            params["kernel"] = st.selectbox("kernel", ["linear", "rbf", "poly", "sigmoid", "cosine"], index=1 if family=="非线性 (Nonlinear)" else 0)
            params["gamma"] = st.number_input("gamma（RBF/多项式）", value=0.0, help="0 表示使用 sklearn 默认", format="%.6f")
            params["degree"] = st.number_input("degree（多项式核）", min_value=2, value=3)
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
        run = st.button("🚀 4) 运行降维", use_container_width=True)

        # ------------------ 执行降维 ------------------
        if run:
            if cols_selected is None or len(cols_selected) < 2:
                st.error("请至少选择两列数值特征。")
                st.stop()

            X = df[cols_selected].astype(float).copy()
            if do_standardize:
                X = StandardScaler().fit_transform(X)

            # 动态限制 t-SNE perplexity
            if method == "t-SNE" and params["perplexity"] >= max(5, X.shape[0] - 1):
                st.warning(f"t-SNE 的 perplexity 必须小于样本数（当前样本 {X.shape[0]}）。已自动调整为 {max(5, min(30, X.shape[0]//3))}")
                params["perplexity"] = max(5, min(30, X.shape[0]//3))

            try:
                if method == "PCA":
                    mdl = PCA(n_components=n_components, random_state=42)
                    emb = mdl.fit_transform(X)
                    info = {"explained_variance_ratio": mdl.explained_variance_ratio_[:min(n_components, len(mdl.explained_variance_ratio_))]}
                elif method.startswith("LDA"):
                    if label_col is None:
                        st.error("LDA 需要选择一个标签列。")
                        st.stop()
                    y = df[label_col].values
                    # LDA 可降到最多 (n_classes - 1) 维
                    n_classes = len(np.unique(y))
                    n_comp_use = min(n_components, max(1, n_classes - 1))
                    mdl = LDA(n_components=n_comp_use)
                    emb = mdl.fit_transform(X, y)
                    # 若 n_comp_use < n_components，用零填充导出
                    if n_comp_use < n_components:
                        emb = np.pad(emb, ((0,0),(0, n_components - n_comp_use)), mode="constant")
                    info = {"classes": list(np.unique(y))}
                elif method == "ICA":
                    mdl = FastICA(n_components=n_components, random_state=int(params["random_state"]), max_iter=int(params["max_iter"]))
                    emb = mdl.fit_transform(X)
                    info = {}
                elif method == "Factor Analysis":
                    mdl = FactorAnalysis(n_components=n_components, random_state=int(params["random_state"]))
                    emb = mdl.fit_transform(X)
                    info = {}
                elif "Kernel PCA" in method:
                    kp_kwargs = dict(kernel=params["kernel"])
                    if params["gamma"] > 0:
                        kp_kwargs["gamma"] = float(params["gamma"])
                    if params["kernel"] == "poly":
                        kp_kwargs["degree"] = int(params["degree"])
                    mdl = KernelPCA(n_components=n_components, **kp_kwargs, random_state=int(params["random_state"]))
                    emb = mdl.fit_transform(X)
                    info = {"kernel": params["kernel"]}
                elif method == "t-SNE":
                    mdl = TSNE(n_components=n_components, perplexity=int(params["perplexity"]),
                               learning_rate=float(params["learning_rate"]), n_iter=int(params["n_iter"]),
                               init="pca" if n_components >= 2 else "random",
                               random_state=int(params["random_state"]))
                    emb = mdl.fit_transform(X)
                    info = {}
                elif method == "UMAP":
                    mdl = UMAP(n_components=n_components, n_neighbors=int(params["n_neighbors"]),
                               min_dist=float(params["min_dist"]), metric=params["metric"],
                               random_state=int(params["random_state"]))
                    emb = mdl.fit_transform(X)
                    info = {}
                elif method == "MDS":
                    mdl = MDS(n_components=n_components, n_init=int(params["n_init"]),
                              max_iter=int(params["max_iter"]), random_state=int(params["random_state"]))
                    emb = mdl.fit_transform(X)
                    info = {}
                else:
                    st.error("未知方法。")
                    st.stop()
            except Exception as e:
                st.error(f"建模失败：{e}")
                st.stop()

            # ------------------ 结果展示 ------------------
            st.success("降维完成！")

            # 2D可视化（若降到≥2维）
            if n_components >= 2:
                viz_df = pd.DataFrame(emb[:, :2], columns=["Dim1", "Dim2"])
                if color_col is not None:
                    viz_df[color_col] = df[color_col].astype(str).values
                elif method.startswith("LDA") and label_col is not None:
                    viz_df[label_col] = df[label_col].astype(str).values

                st.subheader("📈 2D 可视化")
                if (color_col is not None) or (method.startswith("LDA") and label_col is not None):
                    color_key = color_col if color_col is not None else label_col
                    st.scatter_chart(viz_df, x="Dim1", y="Dim2", color=color_key, size=None, height=450)
                else:
                    st.scatter_chart(viz_df, x="Dim1", y="Dim2", height=450)

                if method == "PCA":
                    evr = info.get("explained_variance_ratio", [])
                    if len(evr) >= 2:
                        st.caption(f"PCA 方差贡献：Dim1={evr[0]:.3f}，Dim2={evr[1]:.3f}")

            # 可下载坐标
            out_cols = [f"Dim{i+1}" for i in range(n_components)]
            out_df = pd.DataFrame(emb[:, :n_components], columns=out_cols)
            out_df = pd.concat([df.reset_index(drop=True), out_df], axis=1)

            st.subheader("⬇️ 下载结果")
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="openpyxl") as writer:
                out_df.to_excel(writer, index=False, sheet_name="embedding")
            st.download_button(
                "下载 Excel（原始数据 + 降维坐标）",
                data=buf.getvalue(),
                file_name="embedding_result.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

            # 额外信息
            with st.expander("🔍 运行信息 / 提示"):
                st.json({
                    "method": method,
                    "family": family,
                    "n_components": n_components,
                    "standardized": do_standardize,
                    **({ "pca_explained_variance_ratio": list(info.get("explained_variance_ratio", [])) } if method=="PCA" else {}),
                    **({ "classes": info.get("classes", []) } if method.startswith("LDA") else {}),
                    **({ "kernel": info.get("kernel") } if "Kernel PCA" in method else {})
                })

# ------------------ 页脚 ------------------
st.divider()
st.caption("Made with ❤️ by 木木 • 仅作教学与研究用途")
