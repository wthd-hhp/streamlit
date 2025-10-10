import traceback  # 放在文件开头也可，这里示例内重复导入无妨

# --------- 在调用 predictor.predict 之前执行更严格的检查与清洗 ----------
# merged_features 已经存在（你原来的合并结果）
# 先从 merged_features 提取我们准备预测的列（或直接用 data 变量）
data = merged_features.loc[:, required_descriptors]

st.write("🔍 Columns:", list(data.columns))
st.write("🔍 dtypes:")
st.write(data.dtypes)

# 显示第一行每个单元格类型与 repr，便于定位哪个单元格是序列
first_row = data.iloc[0]
cell_info = {col: (type(first_row[col]).__name__, repr(first_row[col])) for col in data.columns}
st.write("🔍 First row cell types and repr:")
st.json(cell_info)

# 尝试更严格地把所有单元格展平成数值
def force_scalar_float(x):
    try:
        if isinstance(x, (list, tuple, np.ndarray, pd.Series)):
            # 如果是序列，取第一个元素，再尝试强转
            if len(x) == 0:
                return np.nan
            val = x[0]
        else:
            val = x
        # 处理 numpy scalars
        if isinstance(val, (np.generic,)):
            return float(val)
        # 处理 pandas NA / missing
        if val is None:
            return np.nan
        # 最后常规转换
        return float(val)
    except Exception:
        # 返回 NaN 并记录原始 repr（调试时可查看）
        return np.nan

# 应用强转
data_clean = data.applymap(force_scalar_float)
st.write("🔍 data_clean dtypes (after force):")
st.write(data_clean.dtypes)
st.write("🔍 data_clean values (first row):")
st.write(data_clean.iloc[0].to_dict())

# 打印 numpy 形式及 shape，检验是否为规则矩阵
try:
    arr = data_clean.to_numpy()
    st.write("🔍 numpy shape:", arr.shape)
    st.write("🔍 numpy dtype:", arr.dtype)
except Exception as e_arr:
    st.error(f"无法转换为 numpy 数组：{e_arr}")
    st.error(traceback.format_exc())

# 最后再将 NaN 填为 0（或按需填充），并确保 dtype=float
final_input = data_clean.fillna(0.0).astype(float)
st.write("🔍 final_input (ready for predict):")
st.dataframe(final_input)

# 加载模型并预测（用 try/except 捕获完整异常）
try:
    predictor = load_predictor()
    pred = predictor.predict(final_input)
    st.success(f"Predicted Heat Capacity (Cp): {pred.values[0]:.2f} J/(mol·K)")
    del predictor
    gc.collect()
except Exception as e:
    st.error("预测时报错（下面为 traceback）：")
    tb = traceback.format_exc()
    st.code(tb)
    # 另外打印一些额外信息，帮助诊断
    st.write("🔍 final_input info:")
    st.write("columns:", list(final_input.columns))
    st.write("dtypes:")
    st.write(final_input.dtypes)
    st.write("values (repr first row):")
    st.write({c: repr(final_input.iloc[0][c]) for c in final_input.columns})
