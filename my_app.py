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
    .stApp {
        border: 2px solid #808080;
        border-radius: 20px;
        margin: 50px auto;
        max-width: 40%;
        background-color: #f9f9f9f9;
        padding: 20px;
        box-sizing: border-box;
    }
    .rounded-container h2 {
        margin-top: -80px;
        text-align: center;
        background-color: #e0e0e0e0;
        padding: 10px;
        border-radius: 10px;
    }
    .rounded-container blockquote {
        text-align: left;
        margin: 20px auto;
        background-color: #f0f0f0;
        padding: 10px;
        font-size: 1.1em;
        border-radius: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- 页面标题 ----------------
st.markdown(
    """
    <div class='rounded-container'>
        <h2>Predict Heat Capacity (Cp) of Organic Molecules</h2>
        <blockquote>
            1. This web app predicts the heat capacity (Cp) of organic molecules based on their SMILES structure using a trained machine learning model.<br>
            2. Enter a valid SMILES string below to get the predicted result.
        </blockquote>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- 用户输入 ----------------
smiles = st.text_input(
    "Enter the SMILES representation of the molecule:", 
    placeholder="e.g., C1=CC=CC=C1O"
)

submit_button = st.button("Submit and Predict")

# 模型特征（与你的 AutoGluon 模型保持一致）
required_descriptors = ["ATS0se", "EState_VSA5", "ATSC0dv"]

# ---------------- 模型加载 ----------------
@st.cache_resource(show_spinner=False, max_entries=1)
def load_predictor():
    """加载训练好的 AutoGluon 热容预测模型"""
    return TabularPredictor.load("./autogluon")  # ← 改成你的模型文件夹名称

# ---------------- 分子绘图 ----------------
def mol_to_image(mol, size=(300, 300)):
    d2d = MolDraw2DSVG(size[0], size[1])
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()
    svg = re.sub(r'<rect[^>]*>', '', svg, flags=re.DOTALL)
    return svg

# ---------------- 清洗描述符函数 ----------------
def clean_descriptor_dataframe(df):
    """确保所有描述符为单值浮点数，防止 AutoGluon 报 shape 错误"""
    cleaned = df.copy()
    for col in cleaned.columns:
        cleaned[col] = cleaned[col].apply(
            lambda x: x[0] if isinstance(x, (list, tuple, np.ndarray, pd.Series)) and len(x) > 0 else x
        )
    cleaned = cleaned.apply(pd.to_numeric, errors='coerce')
    return cleaned

# ---------------- RDKit 描述符 ----------------
def calc_rdkit_descriptors(smiles_list):
    desc_names = [name for name, _ in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    results = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        desc = calc.CalcDescriptors(mol)
        results.append(desc)
    return pd.DataFrame(results, columns=desc_names)

# ---------------- Mordred 描述符 ----------------
def calc_mordred_descriptors(smiles_list):
    calc = Calculator(descriptors, ignore_3D=True)
    results = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        res = calc(mol)
        desc_dict = {}
        for key, val in res.asdict().items():
            if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
                desc_dict[key] = val[0] if len(val) > 0 else np.nan
            elif val is None or isinstance(val, complex):
                desc_dict[key] = np.nan
            elif hasattr(val, '__class__') and val.__class__.__name__ == 'Missing':
                desc_dict[key] = np.nan
            else:
                desc_dict[key] = val
        results.append(desc_dict)
    return pd.DataFrame(results)

# ---------------- 特征合并 ----------------
def merge_features_without_duplicates(original_df, *feature_dfs):
    merged = pd.concat([original_df] + list(feature_dfs), axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged

# ---------------- 主预测逻辑 ----------------
if submit_button:
    if not smiles:
        st.error("Please enter a valid SMILES string.")
    else:
        with st.spinner("Processing molecule and predicting heat capacity..."):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if not mol:
                    st.error("Invalid SMILES format.")
                    st.stop()

                # 绘制分子结构
                mol = Chem.AddHs(mol)
                AllChem.Compute2DCoords(mol)
                svg = mol_to_image(mol)
                st.markdown(f'<div style="text-align:center;">{svg}</div>', unsafe_allow_html=True)

                # 分子量
                mol_weight = Descriptors.MolWt(mol)
                st.markdown(f"**Molecular Weight:** {mol_weight:.2f} g/mol")

                # 计算并清洗描述符
                smiles_list = [smiles]
                rdkit_features = clean_descriptor_dataframe(calc_rdkit_descriptors(smiles_list))
                mordred_features = clean_descriptor_dataframe(calc_mordred_descriptors(smiles_list))

                merged_features = merge_features_without_duplicates(rdkit_features, mordred_features)
                merged_features = clean_descriptor_dataframe(merged_features)

                st.write(f"🧩 特征矩阵形状: {merged_features.shape}")

                # 提取模型需要的特征
                data = merged_features.loc[:, required_descriptors]
                st.success("✅ 预测输入数据:")
                st.dataframe(data)

                # 加载模型并预测
                predictor = load_predictor()
                prediction = predictor.predict(data)
                st.success(f"**Predicted Heat Capacity (Cp): {prediction.values[0]:.2f} J/(mol·K)**")

                del predictor
                gc.collect()

            except Exception as e:
                st.error(f"❌ 出现错误: {str(e)}")
