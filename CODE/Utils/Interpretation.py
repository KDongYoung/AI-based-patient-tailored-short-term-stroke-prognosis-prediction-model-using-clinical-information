
import shap
import matplotlib.pyplot as plt
import os

def XAI(model, x_test, fold, feature_name, save_path):
    
    if not os.path.isdir(f"{save_path}/XAI"):
        os.makedirs(f"{save_path}/XAI")

    explainer = shap.TreeExplainer(model) # Shap Value 확인 객체 지정
    shap_values = explainer.shap_values(x_test) # Shap Values 계산
  
    # summary
    plt.figure(figsize=(7,6))
    shap.summary_plot(shap_values, x_test, show=False, feature_names=feature_name)
    plt.tight_layout()
    plt.savefig(f'{save_path}/XAI/summary_{fold}fold.png')
    
    # 각 변수에 대한 |Shap Values|을 통해 변수 importance 파악
    plt.figure(figsize=(7,6))
    shap.summary_plot(shap_values, x_test, plot_type = "bar", show=False, feature_names=feature_name)
    plt.tight_layout()
    plt.savefig(f'{save_path}/XAI/summary_{fold}fold2.png')

    