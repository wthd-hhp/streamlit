import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem
from rdkit.Chem.Draw import MolDraw2DSVG
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, descriptors
import pandas as pd
from autogluon.tabular import TabularPredictor
import numpy as np
import gc
import re

# ---------------- 页面样式 ----------------
st.markdown(
    """
    <style>
    .stApp { border: 2px solid #808080; border-radius: 20px; margin: 50px auto; max-width: 42%; background-color: #f9f9f9f9; padding: 20px; box-sizing: border-box; }
    .rounded-container h2 { text-align: center; background-color: #e0e0e0e0; padding: 10px; border-radius: 10px; }
    .rounded-container blockquote { text-align: left; margin: 20px auto; background-color: #f0f0f0; padding: 10px; font-size: 1.1em; border-radius: 10px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='rounded-container'>
        <h2>Predict Heat Capacity (Cp) of Organic Molecules</h2>
        <blockquote>
            输入 SMILES，并预测分子热容（Cp）。页面会显示特征矩阵形状、类型等调试信息，便于定位问题。
        </blockquote>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- 用户输入 ----------
smiles = st.text_input("输入分子的 SMILES：", placeholder="例如：C1=CC=CC=C1O")
submit_button = st.button("提交并预测")

# ---------- 你的模型使用的关键特征（示例） ----------
required_descriptors = ["ATS0se", "EState_VSA5", "ATSC0dv"]

# ---------- 模型加载（缓存） ----------
@st.cache_resource(show_spinner=False, max_entries=1)
def load_predictor():
    return TabularPredictor.load("./autogluon")  # ← 改成你模型路径

# ---------- 绘图函数 ----------
def mol_to_image(mol, size=(300, 300)):
    d2d = MolDraw2DSVG(size[0], size[1])
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()
    svg = re.sub(r'<rect[^>]*>', '', svg, flags=re.DOTALL)
    return svg

# ---------- RDKit 描述符 ----------
def calc_rdkit_descriptors(smiles_list):
    desc_names = [desc_name for desc_name, _ in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    results = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        desc_values = calculator.CalcDescriptors(mol)
        results.append(desc_values)
    return pd.DataFrame(results, columns=desc_names)

# ---------- Mordred 描述符（防护版） ----------
def calc_mordred_descriptors(smiles_list):
    calc = Calculator(descriptors, ignore_3D=True)
    results = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        try:
            res = calc(mol)
            desc_dict = {}
            for key, val in res.asdict().items():
                if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
                    desc_dict[key] = val[0] if len(val) > 0 else np.nan
                elif val is None or isinstance(val, complex):
                    desc_dict[key] = np.nan
                elif hasattr(val, "__class__") and val.__class__.__name__ == "Missing":
                    desc_dict[key] = np.nan
                else:
                    desc_dict[key] = val
            results.append(desc_dict)
        except Exception as e:
            st.warning(f"⚠️ Mordred 描述符计算跳过 SMILES={smi}，错误：{e}")
            results.append({})  # 保持行数一致
    return pd.DataFrame(results)

# ---------- 更强健的清洗函数 ----------
def clean_descriptor_dataframe(df):
    """把列中可能的嵌套/序列/Series 展平为单个数值，并强制为数值型"""
    if df is None or df.shape[0] == 0:
        return pd.DataFrame()
    cleaned = df.copy()
    problem_cols = []
    for col in cleaned.columns:
        # 检查是否存在序列或 Series
        has_seq = cleaned[col].apply(lambda x: isinstance(x, (list, tuple, np.ndarray, pd.Series))).any()
        if has_seq:
            problem_cols.append(col)
            cleaned[col] = cleaned[col].apply(
                lambda x: x[0] if isinstance(x, (list, tuple, np.ndarray, pd.Series)) and len(x) > 0 else np.nan
            )
    # 强制转为数字（无法转换的变为 NaN）
    cleaned = cleaned.apply(pd.to_numeric, errors="coerce")
    if problem_cols:
        st.warning(f"⚠️ 自动清洗这些列（含嵌套序列）：{', '.join(problem_cols)}")
    return cleaned

# ---------- 合并函数 ----------
def merge_features_without_duplicates(original_df, *feature_dfs):
    merged = pd.concat([original_df] + list(feature_dfs), axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged

# ---------- 准备最终输入（按模型期望特征并强制数值） ----------
def prepare_input_df(raw_df, predictor, fallback_keep=None):
    """
    raw_df: 原始包含特征的 DataFrame（列可能多）
    predictor: 已加载的 TabularPredictor
    fallback_keep: 若无法获取模型特征，则使用这个列表作为保留列
    """
    # 1) 读取模型期望的特征名（尽量使用原生 API）
    try:
        model_features = predictor.feature_metadata.get_features()
    except Exception:
        try:
            # fallback: 有时 API 名称不同
            model_features = predictor.feature_metadata.get_features_valid()
        except Exception:
            model_features = None

    if model_features is None:
        if fallback_keep:
            model_features = [c for c in fallback_keep if c in raw_df.columns]
        else:
            # 最后退回到 raw_df 的所有列
            model_features = list(raw_df.columns)

    # 2) 只保留这些特征且按该顺序
    keep = [f for f in model_features if f in raw_df.columns]
    if len(keep) == 0:
        # 没有匹配到特征，退回到 raw_df 的列
        keep = list(raw_df.columns)

    df = raw_df[keep].copy()

    # 3) 再次确保没有嵌套类型并为数值
    df = df.applymap(lambda x: x[0] if isinstance(x, (list, tuple, np.ndarray, pd.Series)) and len(x) > 0 else x)
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0.0).astype(float)

    return df, keep

# ---------------- 主逻辑 ----------------
if submit_button:
    if not smiles:
        st.error("请输入 SMILES。")
    else:
        with st.spinner("处理并预测中..."):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if not mol:
                    st.error("SMILES 无法解析，请检查格式。")
                    st.stop()

                mol = Chem.AddHs(mol)
                AllChem.Compute2DCoords(mol)
                st.markdown(f'<div style="text-align:center;">{mol_to_image(mol)}</div>', unsafe_allow_html=True)
                st.markdown(f"**分子量：** {Descriptors.MolWt(mol):.2f} g/mol")

                smiles_list = [smiles]
                rdkit_features = calc_rdkit_descriptors(smiles_list)
                mordred_features = calc_mordred_descriptors(smiles_list)

                rdkit_features = clean_descriptor_dataframe(rdkit_features)
                mordred_features = clean_descriptor_dataframe(mordred_features)

                merged_features = merge_features_without_duplicates(rdkit_features, mordred_features)
                st.write("🧪 特征矩阵形状（clean后）:", merged_features.shape)

                # 显示前几列及其类型，帮助诊断
                st.write("前20个特征列（名称与数据类型）：")
                show_df = pd.DataFrame({
                    "col": merged_features.columns[:20],
                    "dtype": [str(merged_features[c].dtype) for c in merged_features.columns[:20]]
                })
                st.table(show_df)

                # 取出用于构造输入的关键特征（如果你的模型训练只用部分特征）
                # 先确保 required_descriptors 在 merged_features 中
                missing_req = [f for f in required_descriptors if f not in merged_features.columns]
                if missing_req:
                    st.warning(f"⚠️ 模型所需特征缺失：{missing_req}. 将尽量用可用特征预测。")

                raw_input_df = pd.DataFrame({
                    f: [merged_features.iloc[0][f] if f in merged_features.columns else 0.0] 
                    for f in set(list(merged_features.columns) + required_descriptors)
                })

                st.write("原始候选输入（前50列）：")
                st.dataframe(raw_input_df.iloc[:, :50])

                # 加载模型
                predictor = load_predictor()

                # 根据模型期望特征准备最终输入
                final_df, used_features = prepare_input_df(raw_input_df, predictor, fallback_keep=required_descriptors)
                st.write("✅ final_df shape:", final_df.shape)
                st.write("final_df dtypes:")
                st.write(final_df.dtypes)
                st.write("模型将使用以下特征（按顺序）：")
                st.write(used_features)

                # 预测
                prediction = predictor.predict(final_df)
                st.success(f"Predicted Cp: {prediction.values[0]:.2f} J/(mol·K)")

                # 清理
                del predictor
                gc.collect()

            except Exception as e:
                st.error(f"发生错误：{e}")
