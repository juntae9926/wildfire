import os
from datetime import datetime
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

def make_dir(base_dir):
    now = datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_dir, timestamp)
    os.makedirs(save_dir, exist_ok=True)
    return save_dir

def seed_everything(random_seed):
    np.random.seed(random_seed)
    print(f"Seed locked: {random_seed}")


def compute_shap_values(model, X_test, save_dir):
    try:
        explainer = shap.TreeExplainer(model)
    except: 
        explainer = shap.DeepExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(os.path.join(save_dir, 'shap_summary_plot.png'))
    plt.clf()

def save_correlation_heatmap(data, output_file_path):
    data = data.drop(['label'], axis=1)
    corr_matrix = data.corr(method='pearson')
    f, ax = plt.subplots(figsize=(11, 9))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr_matrix, cmap=cmap, vmax=1.0, center=0,
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, annot=True)

    ax.set_xticklabels([])
    plt.savefig(output_file_path, bbox_inches='tight')
    plt.close()