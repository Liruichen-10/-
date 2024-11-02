import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型
model = joblib.load('SepsisModel.pkl')

# 定义预测变量的名称
feature_names = [
    "uWBC", "Double-J stent duration", "WBC", "ALT", "CR",
    "Albumin", "Stone burden", "import streamlit as st
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

# 加载模型
model = joblib.load('ET.pkl')

# 定义预测变量的名称
feature_names = [
    "uWBC", "Double-J_stent_duration", "WBC", "ALT", "CR",
    "Albumin", "Stone_burden", "Surgical_Duration"
]

# 使用Streamlit创建Web界面
st.title("Sepsis Risk Predictor")

# 收集用户输入
uWBC = st.number_input("Urinary WBC (uWBC):", min_value=0, max_value=500, value=100)
double_j_duration = st.number_input("Double-J stent duration (days):", min_value=0, max_value=365, value=30)
WBC = st.number_input("White Blood Cell Count (WBC):", min_value=0, max_value=50, value=10)
ALT = st.number_input("Alanine Aminotransferase (ALT):", min_value=0, max_value=500, value=35)
CR = st.number_input("Creatinine (CR):", min_value=0.0, max_value=10.0, value=1.0)
Albumin = st.number_input("Albumin (g/dL):", min_value=1.0, max_value=5.0, value=4.0)
stone_burden = st.number_input("Stone burden (mm^2):", min_value=0, max_value=1000, value=50)
surgical_duration = st.number_input("Surgical Duration (minutes):", min_value=0, max_value=600, value=90)

# 将输入的特征转化为数组形式以便模型处理
feature_values = [uWBC, double_j_duration, WBC, ALT, CR, Albumin, stone_burden, surgical_duration]
features = np.array([feature_values])

# 当用户点击“预测”按钮时，进行预测并显示结果
if st.button("Predict"):
    # 预测类别（是否发生脓毒症）
    predicted_class = model.predict(features)[0]

    # 预测概率
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**Predicted Class:** {'Sepsis' if predicted_class == 1 else 'No Sepsis'}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of sepsis. "
            f"The model predicts that your probability of having sepsis is {probability:.1f}%. "
            "Please consult your doctor for further evaluation and potential treatments."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of sepsis. "
            f"The model predicts that your probability of not having sepsis is {probability:.1f}%. "
            "Keep monitoring your health and consult a doctor if you have any concerns."
        )
    st.write(advice)

    # 计算并显示SHAP值，用于解释模型的预测
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    # 显示SHAP force plot，解释每个特征对预测的贡献
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names), matplotlib=True)

    # 保存SHAP plot为图片并展示
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
"
]

# 使用Streamlit创建Web界面
st.title("Sepsis Risk Predictor")

# 收集用户输入
uWBC = st.number_input("Urinary WBC (uWBC):", min_value=0, max_value=500, value=100)
double_j_duration = st.number_input("Double-J stent duration (days):", min_value=0, max_value=365, value=30)
WBC = st.number_input("White Blood Cell Count (WBC):", min_value=0, max_value=50, value=10)
ALT = st.number_input("Alanine Aminotransferase (ALT):", min_value=0, max_value=500, value=35)
CR = st.number_input("Creatinine (CR):", min_value=0.0, max_value=10.0, value=1.0)
Albumin = st.number_input("Albumin (g/dL):", min_value=1.0, max_value=5.0, value=4.0)
stone_burden = st.number_input("Stone burden (mm^2):", min_value=0, max_value=1000, value=50)
surgical_duration = st.number_input("Surgical Duration (minutes):", min_value=0, max_value=600, value=90)

# 将输入的特征转化为数组形式以便模型处理
feature_values = [uWBC, double_j_duration, WBC, ALT, CR, Albumin, stone_burden, surgical_duration]
features = np.array([feature_values])

# 当用户点击“预测”按钮时，进行预测并显示结果
if st.button("Predict"):
    # 预测类别（是否发生脓毒症）
    predicted_class = model.predict(features)[0]

    # 预测概率
    predicted_proba = model.predict_proba(features)[0]

    # 显示预测结果
    st.write(f"**Predicted Class:** {'Sepsis' if predicted_class == 1 else 'No Sepsis'}")
    st.write(f"**Prediction Probabilities:** {predicted_proba}")

    # 根据预测结果生成建议
    probability = predicted_proba[predicted_class] * 100
    if predicted_class == 1:
        advice = (
            f"According to our model, you have a high risk of sepsis. "
            f"The model predicts that your probability of having sepsis is {probability:.1f}%. "
            "Please consult your doctor for further evaluation and potential treatments."
        )
    else:
        advice = (
            f"According to our model, you have a low risk of sepsis. "
            f"The model predicts that your probability of not having sepsis is {probability:.1f}%. "
            "Keep monitoring your health and consult a doctor if you have any concerns."
        )
    st.write(advice)

    # 计算并显示SHAP值，用于解释模型的预测
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(pd.DataFrame([feature_values], columns=feature_names))

    # 显示SHAP force plot，解释每个特征对预测的贡献
    shap.force_plot(explainer.expected_value, shap_values[0], pd.DataFrame([feature_values], columns=feature_names),
                    matplotlib=True)

    # 保存SHAP plot为图片并展示
    plt.savefig("shap_force_plot.png", bbox_inches='tight', dpi=1200)
    st.image("shap_force_plot.png")
