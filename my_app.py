import streamlit as st
from rdkit import Chem
from rdkit.Chem import Descriptors, Draw, AllChem
from rdkit.Chem.Draw import MolDraw2DSVG
from rdkit.ML.Descriptors import MoleculeDescriptors
from mordred import Calculator, descriptors
from mordred import Calculator, descriptors
import pandas as pd
from autogluon.tabular import TabularPredictor
import tempfile
import base64
from io import BytesIO
from autogluon.tabular import FeatureMetadata
import gc  # 添加垃圾回收模块
import re  # 添加正则表达式模块用于处理SVG
from tqdm import tqdm 
import numpy as np


# 添加 CSS 样式
st.markdown(
    """
    <style>
    .stApp {
        border: 2px solid #808080;
        border-radius: 20px;
        margin: 50px auto;
        max-width: 39%; /* 设置最大宽度 */
        background-color: #f9f9f9f9;
        padding: 20px; /* 增加内边距 */
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
    a {
        color: #0000EE;
        text-decoration: underline;
    }
    .process-text, .molecular-weight {
        font-family: Arial, sans-serif;
        font-size: 16px;
        color: #333;
    }
    .stDataFrame {
        margin-top: 10px;
        margin-bottom: 0px !important;
    }
    .molecule-container {
        display: block;
        margin: 20px auto;
        max-width: 300px;
        border: 1px solid #ddd;
        border-radius: 5px;
        padding: 5px;
        background-color: transparent; /* 透明背景 */
    }
     /* 针对小屏幕的优化 */
    @media (max-width: 768px) {
        .rounded-container {
            padding: 10px; /* 减少内边距 */
        }
        .rounded-container blockquote {
            font-size: 0.9em; /* 缩小字体 */
        }
        .rounded-container h2 {
            font-size: 1.2em; /* 调整标题字体大小 */
        }
        .stApp {
            padding: 1px !important; /* 减少内边距 */
            max-width: 99%; /* 设置最大宽度 */
        }
        .process-text, .molecular-weight {
            font-size: 0.9em; /* 缩小文本字体 */
        }
        .molecule-container {
            max-width: 200px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 页面标题和简介
st.markdown(
    """
    <div class='rounded-container'>
        <h2>Predict Organic Fluorescence <br>Emission Wavelengths</h2>
        <blockquote>
            1. This website aims to quickly predict the emission wavelength of organic molecules based on their structure (SMILES) and solvent using machine learning models.<br>
            2. Code and data are available at <a href='https://github.com/dqzs/Fluorescence-Emission-Wavelength-Prediction' target='_blank'>https://github.com/dqzs/Fluorescence-Emission-Wavelength-Prediction</a>.
        </blockquote>
    </div>
    """,
    unsafe_allow_html=True,
)



# SMILES 输入区域
smiles = st.text_input("Enter the SMILES representation of the molecule:", placeholder="e.g., [BH3-][P+]1(c2ccccc2)c2ccccc2-c2sc3ccccc3c21,Solvent:Cyclohexane")

# 提交按钮
submit_button = st.button("Submit and Predict", key="predict_button")

# 指定的描述符列表 - 已更新为所需的特征
required_descriptors = ["ATS0se", "EState_VSA5", "ATSC0dv"]

# 缓存模型加载器以避免重复加载
@st.cache_resource(show_spinner=False, max_entries=1)  # 限制只缓存一个实例
def load_predictor():
    """缓存模型加载，避免重复加载导致内存溢出"""
    return TabularPredictor.load("./ag-20250609_005753")

def mol_to_image(mol, size=(300, 300)):
    """将分子转换为背景颜色为 #f9f9f9f9 的SVG图像"""
    # 创建绘图对象
    d2d = MolDraw2DSVG(size[0], size[1])
    
    # 获取绘图选项
    draw_options = d2d.drawOptions()
    
    # 设置背景颜色为 #f9f9f9f9
    draw_options.background = '#f9f9f9'
    
    # 移除所有边框和填充
    draw_options.padding = 0.0
    draw_options.additionalBondPadding = 0.0
    
    # 移除原子标签的边框
    draw_options.annotationFontScale = 1.0
    draw_options.addAtomIndices = False
    draw_options.addStereoAnnotation = False
    draw_options.bondLineWidth = 1.5
    
    # 禁用所有边框
    draw_options.includeMetadata = False
    
    # 绘制分子
    d2d.DrawMolecule(mol)
    d2d.FinishDrawing()
    
    # 获取SVG内容
    svg = d2d.GetDrawingText()
    
    # 移除SVG中所有可能存在的边框元素
    # 1. 移除黑色边框矩形
    svg = re.sub(r'<rect [^>]*stroke:black[^>]*>', '', svg, flags=re.DOTALL)
    svg = re.sub(r'<rect [^>]*stroke:#000000[^>]*>', '', svg, flags=re.DOTALL)
    
    # 2. 移除所有空的rect元素
    svg = re.sub(r'<rect[^>]*/>', '', svg, flags=re.DOTALL)
    
    # 3. 确保viewBox正确设置
    if 'viewBox' in svg:
        # 设置新的viewBox以移除边距
        svg = re.sub(r'viewBox="[^"]+"', f'viewBox="0 0 {size[0]} {size[1]}"', svg)
    
    return svg
# RDKit 描述符计算函数
def calc_rdkit_descriptors(smiles_list):
    # 获取所有可用描述符名称
    desc_names = [desc_name for desc_name, _ in Descriptors._descList]
    calculator = MoleculeDescriptors.MolecularDescriptorCalculator(desc_names)
    
    results = []
    valid_indices = []
    skipped_molecules = []
    
    for idx, smi in tqdm(enumerate(smiles_list), total=len(smiles_list), desc="Calculating RDKit descriptors"):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smi}")
            
            # 添加氢原子以获得更准确的计算
            mol = Chem.AddHs(mol)
            
            # 计算所有描述符
            descriptors = calculator.CalcDescriptors(mol)
            
            # 检查并处理特殊值
            processed_descriptors = []
            for val in descriptors:
                # 处理所有不可用的值类型
                if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                    processed_descriptors.append(np.nan)
                elif val is None:  # 处理None值
                    processed_descriptors.append(np.nan)
                else:
                    processed_descriptors.append(val)
            
            results.append(processed_descriptors)
            valid_indices.append(idx)
        except Exception as e:
            skipped_molecules.append((smi, str(e)))
            print(f"Skipped SMILES: {smi}, reason: {str(e)}")
            continue
    
    # 转换为DataFrame并保留有效索引
    df_desc = pd.DataFrame(results, columns=desc_names, index=valid_indices)
    
    return df_desc

# Mordred 描述符计算函数
def calc_mordred_descriptors(smiles_list):
    # 创建仅包含2D描述符的计算器
    calc = Calculator(descriptors, ignore_3D=True)
    
    results = []
    valid_smiles = []
    skipped_molecules = []
    
    for smi in tqdm(smiles_list, desc="Calculating Mordred descriptors"):
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smi}")
            
            # 添加氢原子以获得更准确的计算
            mol = Chem.AddHs(mol)
            
            # 计算描述符
            res = calc(mol)
            
            # 处理结果，保留原始值或转换为NaN
            descriptor_dict = {}
            for key, val in res.asdict().items():
                # 处理NaN和无穷大
                if isinstance(val, float) and (np.isnan(val) or np.isinf(val)):
                    descriptor_dict[key] = np.nan
                # 处理None值
                elif val is None:
                    descriptor_dict[key] = np.nan
                # 处理Mordred的Missing值（无需特殊导入）
                elif hasattr(val, '__class__') and val.__class__.__name__ == 'Missing':  # 替代Missing检查
                    descriptor_dict[key] = np.nan
                else:
                    descriptor_dict[key] = val
            
            results.append(descriptor_dict)
            valid_smiles.append(smi)
        except Exception as e:
            skipped_molecules.append((smi, str(e)))
            print(f"Skipped SMILES: {smi}, reason: {str(e)}")
            continue
    
    # 创建DataFrame
    df_mordred = pd.DataFrame(results)
    df_mordred['SMILES'] = valid_smiles
    return df_mordred

# 改进的特征合并函数
def merge_features_without_duplicates(original_df, *feature_dfs):
    """合并多个特征DataFrame并去除重复列"""
    # 按顺序合并（后出现的重复列会被丢弃）
    merged = pd.concat([original_df] + list(feature_dfs), axis=1)
    
    # 保留第一个出现的列（根据合并顺序）
    merged = merged.loc[:, ~merged.columns.duplicated()]
    
    return merged


# 如果点击提交按钮
if submit_button:
    if not smiles:
        st.error("Please enter a valid SMILES string.")
    elif not solvent:
        st.error("Please select a solvent.")
    else:
        with st.spinner("Processing molecule and making predictions..."):
            try:
                # 处理SMILES输入
                mol = Chem.MolFromSmiles(smiles)
                if not mol:
                    st.error("Invalid SMILES input. Please check the format.")
                    st.stop()
                
                # 添加H原子并生成2D坐标
                mol = Chem.AddHs(mol)
                AllChem.Compute2DCoords(mol)

                # 显示分子结构
                svg = mol_to_image(mol)
                st.markdown(
                    f'<div class="molecule-container" style="background-color: #f9f9f9; padding: 0; border: none;">{svg}</div>', 
                    unsafe_allow_html=True
                )
                # 计算分子量
                mol_weight = Descriptors.MolWt(mol)
                st.markdown(f'<div class="molecular-weight">Molecular Weight: {mol_weight:.2f} g/mol</div>',
                            unsafe_allow_html=True)

               
                
                # 计算指定描述符 - 现在传递SMILES字符串
                smiles_list = [smiles]  # 将单个 SMILES 转换为列表
                rdkit_features = calc_rdkit_descriptors(smiles_list)
                mordred_features = calc_mordred_descriptors(smiles_list)
                
                # 合并特征并去除重复列
                merged_features = merge_features_without_duplicates(rdkit_features, mordred_features)
                data=merged_features.loc[:, ['ATS0se', 'EState_VSA5', 'ATSC0dv']]

                # 创建输入数据表 - 使用新的特征
                input_data = {
                    "SMILES": [smiles],
                    'ATS0se': [data.iloc[0]['ATS0se']], 
                    'EState_VSA5': [data.iloc[0]['EState_VSA5']], 
                    'ATSC0dv': [data.iloc[0]['ATSC0dv']]
                }
            
                input_df = pd.DataFrame(input_data)
                
                # 显示输入数据
                st.write("Input Data:")
                st.dataframe(input_df)

                # 创建预测用数据框 - 使用新的特征
                predict_df = pd.DataFrame({
                    'ATS0se': [data.iloc[0]['ATS0se']], 
                    'EState_VSA5': [data.iloc[0]['EState_VSA5']], 
                    'ATSC0dv': [data.iloc[0]['ATSC0dv']]
                })
                
                # 加载模型并预测
                try:
                    # 使用缓存的模型加载方式
                    predictor = load_predictor()
                    
                    # 只使用最关键的模型进行预测，减少内存占用
                    essential_models = ['CatBoost',
                                         'LightGBM',
                                         'LightGBMLarge',
                                         'RandomForestMSE',
                                         'WeightedEnsemble_L2',
                                         'XGBoost']
                    predict_df_1 = pd.concat([predict_df,predict_df],axis=0)
                    predictions_dict = {}
                    
                    for model in essential_models:
                        try:
                            predictions = predictor.predict(predict_df_1, model=model)
                            predictions_dict[model] = predictions.astype(int).apply(lambda x: f"{x} nm")
                        except Exception as model_error:
                            st.warning(f"Model {model} prediction failed: {str(model_error)}")
                            predictions_dict[model] = "Error"

                    # 显示预测结果
                    st.write("Prediction Results (Essential Models):")
                    st.markdown(
                        "**Note:** WeightedEnsemble_L2 is a meta-model combining predictions from other models.")
                    results_df = pd.DataFrame(predictions_dict)
                    st.dataframe(results_df.iloc[:1,:])
                    
                    # 主动释放内存
                    del predictor
                    gc.collect()

                except Exception as e:
                    st.error(f"Model loading failed: {str(e)}")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
