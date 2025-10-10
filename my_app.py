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
    根据 predictor 的期望特征构造 final_input DataFrame。
    缺失特征用 0.0 填充，保证列顺序与训练时一致。
    """
    # 尽量通过 feature_metadata 获取特征名
    try:
        model_features = predictor.feature_metadata.get_features()
    except Exception:
        # fallback: 使用训练时保存的列（若可访问）
        try:
            model_features = predictor._learner.feature_metadata.get_features()
        except Exception:
            model_features = None

    if model_features is None or len(model_features) == 0:
        # 最后退回到 merged_features 的所有列（但这不理想）
        model_features = list(merged_features_df.columns)

    # 创建一个全零 dataframe（1 行）
    final_df = pd.DataFrame(0.0, index=[0], columns=model_features)

    # 将 merged_features 的已有列拷贝到 final_df（如果列名匹配）
    for col in merged_features_df.columns:
        if col in final_df.columns:
            try:
                val = merged_features_df.iloc[0][col]
                # 若是序列，取第一个标量
                if isinstance(val, (list, tuple, np.ndarray, pd.Series)):
                    val = val[0] if len(val) > 0 else np.nan
                final_df.at[0, col] = float(val) if pd.notna(val) else 0.0
            except Exception:
                # 如果转换失败，设置为 0.0
                final_df.at[0, col] = 0.0

    # 确保所有列为 float 类型
    final_df = final_df.astype(float)
    return final_df

# ========== 主流程 ==========
if submit:
    if not smiles:
        st.error("Please enter a SMILES string.")
    else:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                st.error("Invalid SMILES. Please check the input.")
            else:
                # 显示分子图与分子量
                mol = Chem.AddHs(mol)
                AllChem.Compute2DCoords(mol)
                st.markdown(mol_to_image(mol), unsafe_allow_html=True)
                st.write(f"Molecular Weight: {Descriptors.MolWt(mol):.2f} g/mol")

                # 计算描述符并清洗
                smiles_list = [smiles]
                rdkit_df = clean_descriptor_dataframe(calc_rdkit_descriptors(smiles_list))
                mordred_df = clean_descriptor_dataframe(calc_mordred_descriptors(smiles_list))
                merged = pd.concat([rdkit_df, mordred_df], axis=1)
                merged = merged.loc[:, ~merged.columns.duplicated()]
                merged = clean_descriptor_dataframe(merged)

                # 加载模型
                predictor = load_predictor()

                # 构建与模型期望匹配的输入向量（严格列对齐）
                final_input = build_model_input(merged, predictor)

                # 预测
                pred = predictor.predict(final_input)
                st.success(f"Predicted Heat Capacity (Cp): {pred.values[0]:.2f} J/(mol·K)")

                # 清理
                del predictor
                gc.collect()

        except Exception as e:
            st.error(f"Error during prediction: {e}")
