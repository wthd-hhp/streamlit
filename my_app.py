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
            1. 本网页工具基于机器学习模型，可根据分子结构（SMILES）预测有机物的热容（Cp）。<br>
            2. 请输入正确的 SMILES 字符串，系统将自动计算分子描述符并进行预测。
        </blockquote>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- 用户输入 ----------------
smiles = st.text_input(
    "输入分子的 SMILES 表示式：", 
    placeholder="例如：C1=CC=CC=C1O"
)
submit_button = st.button("提交并预测")

# ---------------- 需要的描述符（与你模型一致） ----------------
required_descriptors = ["ATS0se", "EState_VSA5", "ATSC0dv"]

# ---------------- 模型加载 ----------------
@st.cache_resource(show_spinner=False, max_entries=1)
def load_predictor():
    """加载训练好的 AutoGluon 热容预测模型"""
    return TabularPredictor.load("./autogluon")  # ← 修改为你模型文件夹路径

# ---------------- 分子绘图函数 ----------------
def mol_to_image(mol, size=(300, 300)):
    d2d = MolDraw2DSVG(size[0], size[1])
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()
    svg = re.sub(r'<rect[^>]*>', '', svg, flags=re.DOTALL)
    return svg

# ---------------- RDKit 描述符 ----------------
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

# ---------------- Mordred 描述符 ----------------
def calc_mordred_descriptors(smiles_list):
    calc = Calculator(descriptors, ignore_3D=True)
    results = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        res = calc(mol)
        results.append(res.asdict())
    return pd.DataFrame(results)

# ---------------- 清洗描述符函数（关键修复） ----------------
def clean_descriptor_dataframe(df):
    """确保所有描述符都是单值数值（非列表或对象），防止 numpy 报错"""
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: x[0] if isinstance(x, (list, tuple, np.ndarray)) else x
        )
    df = df.apply(pd.to_numeric, errors='coerce')
    return df

# ---------------- 合并特征 ----------------
def merge_features_without_duplicates(original_df, *feature_dfs):
    merged = pd.concat([original_df] + list(feature_dfs), axis=1)
    merged = merged.loc[:, ~merged.columns.duplicated()]
    return merged

# ---------------- 主预测逻辑 ----------------
if submit_button:
    if not smiles:
        st.error("请输入有效的 SMILES 字符串。")
    else:
        with st.spinner("正在处理分子并预测热容，请稍候..."):
            try:
                mol = Chem.MolFromSmiles(smiles)
                if not mol:
                    st.error("SMILES 格式无效，请检查输入。")
                    st.stop()

                # 绘制分子结构
                mol = Chem.AddHs(mol)
                AllChem.Compute2DCoords(mol)
                svg = mol_to_image(mol)
                st.markdown(f'<div style="text-align:center;">{svg}</div>', unsafe_allow_html=True)

                # 分子量
                mol_weight = Descriptors.MolWt(mol)
                st.markdown(f"**分子量：** {mol_weight:.2f} g/mol")

                # 计算描述符
                smiles_list = [smiles]
                rdkit_features = calc_rdkit_descriptors(smiles_list)
                mordred_features = calc_mordred_descriptors(smiles_list)

                # 🔹 数据清洗，防止列表/对象型数据
                rdkit_features = clean_descriptor_dataframe(rdkit_features)
                mordred_features = clean_descriptor_dataframe(mordred_features)

                # 合并特征
                merged_features = merge_features_without_duplicates(rdkit_features, mordred_features)

                # 检查是否有序列型列（调试提示）
                for col in merged_features.columns:
                    types = merged_features[col].apply(lambda x: type(x)).unique()
                    if any(t in [list, tuple, np.ndarray] for t in types):
                        st.warning(f"⚠️ 特征列 {col} 含有序列数据，已自动清洗。")

                # 提取模型所需特征
                data = merged_features.loc[:, required_descriptors]

                # 构建预测输入
                predict_df = pd.DataFrame({
                    'ATS0se': [data.iloc[0]['ATS0se']], 
                    'EState_VSA5': [data.iloc[0]['EState_VSA5']], 
                    'ATSC0dv': [data.iloc[0]['ATSC0dv']]
                })

                # 加载模型并预测
                predictor = load_predictor()
                prediction = predictor.predict(predict_df)

                # 显示预测结果
                st.success(f"预测热容 Cp：{prediction.values[0]:.2f} J/(mol·K)")

                # 释放内存
                del predictor
                gc.collect()

            except Exception as e:
                st.error(f"出现错误：{str(e)}")
