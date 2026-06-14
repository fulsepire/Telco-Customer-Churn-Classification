import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


def print_metrics(model_name, y_test, y_pred, y_prob):
    print(f"=== {model_name} ===")
    print(classification_report(y_test, y_pred))
    print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")


def plot_confusion_matrix(model_name, y_test, y_pred, ax):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    ax.set_title(model_name)
    ax.set_ylabel('Stvarna vrednost')
    ax.set_xlabel('Predviđena vrednost')


def plot_feature_importance(feat_imp):
    plt.figure(figsize=(10, 8))
    sns.barplot(x=feat_imp.values, y=feat_imp.index,
                hue=feat_imp.index, palette='coolwarm', legend=False)
    plt.title('Feature Importance — XGBoost')
    plt.xlabel('Važnost')
    plt.tight_layout()
    plt.show()