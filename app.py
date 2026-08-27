import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import os
from pathlib import Path
import statsmodels.api as sm

# 强制后台绘图，防止网页端多线程冲突
import matplotlib
matplotlib.use('Agg')

# ==========================================
# 0. 网页全局配置 & 终极 CSS 美化
# ==========================================
st.set_page_config(
    page_title="临床风险智能决策系统",
    page_icon="⚕️",
    layout="wide",
    initial_sidebar_state="expanded" # 默认展开侧边栏
)

# 💉 注入自定义高级 CSS
st.markdown("""
    <style>
    html, body, [class*="css"]  {
        font-family: 'Times New Roman', 'Microsoft YaHei', sans-serif;
    }
    
    /* 核心主按钮样式重写 */
    div.stButton > button:first-child {
        background-color: #2E86C1;
        color: white;
        border-radius: 8px;
        padding: 10px 24px;
        font-size: 18px;
        font-weight: bold;
        border: none;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 15px;
    }
    div.stButton > button:first-child:hover {
        background-color: #1B4F72;
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
        transform: translateY(-2px);
    }
    
    /* 侧边栏微调 */
    [data-testid="stSidebar"] {
        background-color: #F8F9F9;
        border-right: 1px solid #E5E7E9;
    }
    
    /* 预测结果超大数值的颜色强化 */
    div[data-testid="stMetricValue"] {
        font-size: 2.8rem;
        color: #C0392B; 
        font-weight: 900;
    }
    
    /* 输入框视觉强化 */
    input[type="number"] {
        font-weight: bold;
        color: #154360;
        background-color: #F4F6F7;
    }
    
    /* 隐藏原生菜单和 Deploy 按钮 */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 1. 顶部抬头设计
# ==========================================
col_logo, col_title = st.columns([1, 8])
with col_logo:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=80) 
with col_title:
    st.title("临床危险因素智能评估与决策平台")
    st.markdown("**(Clinical Risk Intelligent Assessment & Decision Support System)**")

st.markdown("""
<div style='background-color: #EBF5FB; padding: 15px; border-radius: 10px; border-left: 5px solid #2980B9; margin-bottom: 25px;'>
    <span style='color: #154360; font-size: 15px;'>
    <b>📊 系统简介：</b>本平台搭载最前沿的机器学习算法架构。支持<b>侧边栏滑动</b>与<b>主界面精确录入</b>双向同步，实时推演风险发生概率，并利用 <b>SHAP</b> 技术透视背后的危险致病因素。
    </span>
</div>
""", unsafe_allow_html=True)

# 🚨🚨🚨 【模型挂载配置区】 🚨🚨🚨
CHAMPION_MODEL_NAME = "XGBoost" 
SCENARIO = "非过采样结果" 
BASE_DIR = r"D:\MachineLearningpack\新代码"
# 🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨🚨

# ==========================================
# 2. 智能加载引擎 
# ==========================================
@st.cache_resource 
def load_assets(base_dir, scenario, model_name):
    folder_mapping = {
        "Logistic": "logistic", "RandomForest": "random_forest", "DecisionTree": "decision_tree",
        "XGBoost": "xgboost", "LightGBM": "lightgbm", "SVM": "svm", 
        "ANN": "ann", "CatBoost": "catboost", "TabNet": "tabnet"
    }
    actual_folder_name = folder_mapping.get(model_name, model_name.lower())
    model_folder = os.path.join(base_dir, "结果", scenario, actual_folder_name)
    
    if not os.path.exists(model_folder):
        raise FileNotFoundError(f"找不到模型文件夹: {model_folder}。请确保层级结构正确。")

    model_files = list(Path(model_folder).glob("*_model.pkl")) + list(Path(model_folder).glob("tabnet_model.zip"))
    if not model_files:
        raise FileNotFoundError(f"在 {model_folder} 中找不到模型文件！")
        
    scaler = None
    scaler_files = list(Path(model_folder).glob("*scaler.pkl"))
    if scaler_files:
        with open(scaler_files[0], 'rb') as f:
            scaler = pickle.load(f)

    if model_files[0].name.endswith('.pkl'):
        with open(model_files[0], 'rb') as f:
            model = pickle.load(f)
    else:
        from pytorch_tabnet.tab_model import TabNetClassifier
        model = TabNetClassifier()
        model.load_model(str(model_files[0]))
            
    return model, scaler

try:
    model, scaler = load_assets(BASE_DIR, SCENARIO, CHAMPION_MODEL_NAME)
except Exception as e:
    st.error("🚨 模型加载失败！请检查目录路径。")
    st.warning(f"底层报错: {e}")
    st.stop()


# ==========================================
# 3. 黑科技：双向状态同步 (Two-way Binding)
# ==========================================
# 基于实际数据分布设定的科学中位数作为默认值
default_values = {
    'ChE': 6664.0, 'Age': 816.0, 'PA': 197.5, 
    'Crea': 69.7, 'FDP': 1.5, 'Lymph_pct': 24.2, 
    'CEA': 3.57, 'GLO': 28.6, 'Lymph_count': 1.59
}

# 初始化 Session State 缓存
for key, val in default_values.items():
    if f"{key}_slider" not in st.session_state:
        st.session_state[f"{key}_slider"] = val
    if f"{key}_num" not in st.session_state:
        st.session_state[f"{key}_num"] = val

def sync_inputs(src_key, dest_key):
    st.session_state[dest_key] = st.session_state[src_key]


# ==========================================
# 4. 侧边栏：滑动控制台
# ==========================================
st.sidebar.markdown("### 🖥️ 系统运行状态")
st.sidebar.success(f"🟢 挂载模型: **{CHAMPION_MODEL_NAME}**")
st.sidebar.markdown("---")

st.sidebar.markdown("### 🎛️ 快速滑动控制台")
st.sidebar.markdown("*(滑动下方模块，右侧数值将自动同步)*")

with st.sidebar.expander("👤 基本信息与肝肾功能", expanded=True):
    st.slider("年龄 (Age, 月龄)", 200.0, 1300.0, step=1.0, key="Age_slider", on_change=sync_inputs, args=("Age_slider", "Age_num"))
    st.slider("肌酐 (Crea)", 10.0, 1200.0, step=0.1, key="Crea_slider", on_change=sync_inputs, args=("Crea_slider", "Crea_num"))
    st.slider("前白蛋白 (PA) mg/L", 10.0, 800.0, step=1.0, key="PA_slider", on_change=sync_inputs, args=("PA_slider", "PA_num"))
    st.slider("球蛋白 (GLO) g/L", 10.0, 120.0, step=0.1, key="GLO_slider", on_change=sync_inputs, args=("GLO_slider", "GLO_num"))

with st.sidebar.expander("🩸 血液学与凝血指标", expanded=True):
    st.slider("淋巴细胞百分比 (Lymph%)", 0.0, 100.0, step=0.1, key="Lymph_pct_slider", on_change=sync_inputs, args=("Lymph_pct_slider", "Lymph_pct_num"))
    st.slider("淋巴细胞绝对值 (Lymph count)", 0.0, 50.0, step=0.01, key="Lymph_count_slider", on_change=sync_inputs, args=("Lymph_count_slider", "Lymph_count_num"))
    st.slider("纤维蛋白降解产物 (FDP)", 0.0, 300.0, step=0.01, key="FDP_slider", on_change=sync_inputs, args=("FDP_slider", "FDP_num"))
    
with st.sidebar.expander("🔬 肿瘤与特种酶标志物", expanded=True):
    st.slider("胆碱酯酶 (ChE) U/L", 100.0, 25000.0, step=10.0, key="ChE_slider", on_change=sync_inputs, args=("ChE_slider", "ChE_num"))
    st.slider("癌胚抗原 (CEA)", 0.0, 5000.0, step=0.1, key="CEA_slider", on_change=sync_inputs, args=("CEA_slider", "CEA_num"))


# ==========================================
# 5. 主界面：精确录入矩阵
# ==========================================
st.markdown("### 👨‍⚕️ 患者生理指标实时录入矩阵")
st.markdown("*(直接在下方框内键入化验单数值，或使用左侧边栏拉动调节)*")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.number_input("Age (年龄, 月龄)", min_value=200.0, max_value=1300.0, step=1.0, format="%.0f", key="Age_num", on_change=sync_inputs, args=("Age_num", "Age_slider"))
    st.number_input("Crea (肌酐)", min_value=10.0, max_value=1200.0, step=0.1, format="%.1f", key="Crea_num", on_change=sync_inputs, args=("Crea_num", "Crea_slider"))
with col2:
    st.number_input("PA (前白蛋白)", min_value=10.0, max_value=800.0, step=1.0, format="%.1f", key="PA_num", on_change=sync_inputs, args=("PA_num", "PA_slider"))
    st.number_input("GLO (球蛋白)", min_value=10.0, max_value=120.0, step=0.1, format="%.1f", key="GLO_num", on_change=sync_inputs, args=("GLO_num", "GLO_slider"))
with col3:
    st.number_input("Lymph% (淋巴百分比)", min_value=0.0, max_value=100.0, step=0.1, format="%.1f", key="Lymph_pct_num", on_change=sync_inputs, args=("Lymph_pct_num", "Lymph_pct_slider"))
    st.number_input("Lymph count (淋巴绝对值)", min_value=0.0, max_value=50.0, step=0.01, format="%.2f", key="Lymph_count_num", on_change=sync_inputs, args=("Lymph_count_num", "Lymph_count_slider"))
with col4:
    st.number_input("ChE (胆碱酯酶)", min_value=100.0, max_value=25000.0, step=10.0, format="%.0f", key="ChE_num", on_change=sync_inputs, args=("ChE_num", "ChE_slider"))
    st.number_input("CEA (癌胚抗原)", min_value=0.0, max_value=5000.0, step=0.1, format="%.2f", key="CEA_num", on_change=sync_inputs, args=("CEA_num", "CEA_slider"))

col5, col6, col7, col8 = st.columns(4)
with col5:
    st.number_input("FDP (纤维蛋白降解产物)", min_value=0.0, max_value=300.0, step=0.01, format="%.2f", key="FDP_num", on_change=sync_inputs, args=("FDP_num", "FDP_slider"))


# 组装为 DataFrame 喂给底层引擎 (注意列名严格对应训练集 CSV 表头)
input_df = pd.DataFrame({
    'ChE': [st.session_state["ChE_num"]], 
    'Age': [st.session_state["Age_num"]], 
    'PA': [st.session_state["PA_num"]], 
    'Crea': [st.session_state["Crea_num"]],
    'FDP': [st.session_state["FDP_num"]], 
    'Lymph%': [st.session_state["Lymph_pct_num"]], 
    'CEA': [st.session_state["CEA_num"]], 
    'GLO': [st.session_state["GLO_num"]],
    'Lymphocyte count': [st.session_state["Lymph_count_num"]] 
})

# ==========================================
# 6. 核心预测与解释引擎
# ==========================================
if st.button("🚀 启动 AI 风险深度推演", type="primary"):
    with st.spinner('🧬 模型正在提取高维特征并进行反向求导计算，请稍候...'):
        
        # --- 1. 智能量纲剥离 ---
        if scaler is not None:
            input_calc = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)
        else:
            input_calc = input_df
            
        # --- 2. 智能预测 ---
        if CHAMPION_MODEL_NAME == "Logistic":
            X_const = sm.add_constant(input_calc, has_constant='add')
            if 'const' not in X_const.columns:
                X_const.insert(0, 'const', 1.0)
            risk_prob = model.predict(X_const).values[0]
        elif CHAMPION_MODEL_NAME == "TabNet":
            risk_prob = model.predict_proba(input_calc.values)[0][1]
        else:
            risk_prob = model.predict_proba(input_calc)[0][1] 
        
        # --- 3. 结果展示 ---
        st.markdown("---")
        st.markdown("### 🎯 临床诊断预估报告")
        
        res_col1, res_col2 = st.columns([1, 2])
        
        with res_col1:
            st.metric(label=f"综合致病风险概率", value=f"{risk_prob * 100:.2f} %")
            
        with res_col2:
            st.markdown("<br>", unsafe_allow_html=True) 
            if risk_prob > 0.5: 
                st.error("🚨 **【红色预警】该患者处于高风险区间！** 综合模型判定其极易发生不良结局，建议立即启动重点监护机制并制定干预预案。")
                st.toast('检测到高危风险！', icon='⚠️') 
            else:
                st.success("✅ **【安全评估】该患者处于低风险区间。** 暂未发现明显致病倾向，建议维持当前治疗方案及常规随访。")
                st.balloons() 

        # --- 4. SHAP 实时可解释性生成 ---
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 🧠 致病动因深度追踪 (SHAP 个体化靶点解析)")
        st.info("💡 **图表解读指南：** 红色条柱代表将患者风险推高的**危险因子**，蓝色条柱代表降低风险的**保护因子**。条柱越长，对该患者当前决策的影响权重越大。")
        
        try:
            tree_models = ["XGBoost", "RandomForest", "DecisionTree", "LightGBM", "CatBoost"]
            if CHAMPION_MODEL_NAME in tree_models:
                explainer = shap.TreeExplainer(model)
                shap_values_raw = explainer.shap_values(input_calc)
                
                # 处理不同树模型返回结构不一致的问题
                if isinstance(shap_values_raw, list):
                    shap_val_single = shap_values_raw[1][0]
                elif len(np.array(shap_values_raw).shape) == 3:
                    shap_val_single = shap_values_raw[0, :, 1]
                else:
                    shap_val_single = shap_values_raw[0]
                    
                ev = explainer.expected_value
                base_val = ev[1] if isinstance(ev, (list, np.ndarray)) and len(np.array(ev).flatten()) > 1 else (ev[0] if isinstance(ev, (list, np.ndarray)) else ev)
                
                # 画图 (显式声明fig避免重叠)
                fig, ax = plt.subplots(figsize=(10, 6))
                exp = shap.Explanation(values=shap_val_single, base_values=base_val, 
                                       data=input_df.iloc[0], feature_names=input_df.columns.tolist())
                shap.waterfall_plot(exp, max_display=10, show=False)
                st.pyplot(fig)
                plt.close(fig) # 释放内存
                
            else:
                st.warning(f"由于您当前采用的是非树形架构基座 [{CHAMPION_MODEL_NAME}]，无法直接渲染瀑布图，请参考后台生成的全局特征重要性矩阵。")
                
        except Exception as e:
            st.error(f"渲染瀑布图时遭遇底层异常: {e}")