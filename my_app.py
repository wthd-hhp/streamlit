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

# ========== 页面样式 ==========
st.set_page_config(layout="centered")
st.markdown(
    """
    <style>
    .stApp { border: 2px solid #808080; border-radius: 20px; margin: 30px auto; max-width: 720px;
            background-color: #f9f9f9; padding: 20px; box-sizing: border-box; }
    .rounded-container h2 { text-align: center; background-color: #eaeaea; padding: 10px; border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class='rounded-container'>
        <h2>Predict Heat Capacity (Cp)</h2>
        <p>Enter a SMILES string; the app computes descriptors and predicts heat capacity using a trained AutoGluon model.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ========== 用户输入 ==========
smiles = st.text_input("SMILES:", placeholder="e.g., C1=CC=CC=C1O")
submit = st.button("Predict Cp")

# ========== 需要的本地/显式特征名（若你知道，可放在这里备份） ==========
# required_descriptors = ["ATS0se", "EState_VSA5", "ATSC0dv"]
#（此版本将以模型的 feature list 为准，不直接依赖 required_descriptors）

# ========== 模型加载 ==========
@st.cache_resource(show_spinner=False)
def load_predictor():
    # 修改为你的模型路径（模型文件夹, 如 ./autogluon 或 ./ag-heatcapacity-gas 等）
    return TabularPredictor.load("./autogluon")

# ========== 工具函数 ==========
def mol_to_image(mol, size=(300, 300)):
    d2d = MolDraw2DSVG(size[0], size[1])
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    svg = d2d.GetDrawingText()
    svg = re.sub(r'<rect[^>]*>', '', svg, flags=re.DOTALL)
    return svg

def calc_rdkit_descriptors(smiles_list):
    desc_names = [name for name, _ in Descriptors._descList]
    calc = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        vals = calc.CalcDescriptors(mol)
        rows.append(vals)
    return pd.DataFrame(rows, columns=desc_names)

def calc_mordred_descriptors(smiles_list):
    calc = Calculator(descriptors, ignore_3D=True)
    rows = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        mol = Chem.AddHs(mol)
        res = calc(mol)
        d = {}
        for k, v in res.asdict().items():
            if isinstance(v, (list, tuple, np.ndarray, pd.Series)):
                d[k] = v[0] if len(v) > 0 else np.nan
            elif v is None or isinstance(v, complex):
                d[k] = np.nan
            elif hasattr(v, "__class__") and v.__class__.__name__ == "Missing":
                d[k] = np.nan
            else:
                d[k] = v
        rows.append(d)
    return pd.DataFrame(rows)

def clean_descriptor_dataframe(df):
    if df is None or df.shape[0] == 0:
        return pd.DataFrame()
    cleaned = df.copy()
    for col in cleaned.columns:
        cleaned[col] = cleaned[col].apply(
            lambda x: x[0] if isinstance(x, (list, tuple, np.ndarray, pd.Series)) and len(x) > 0 else x
        )
    cleaned = cleaned.apply(pd.to_numeric, errors="coerce")
    return cleaned

def build_model_input(merged_features_df, predictor):
    """
    根据 predictor 的
