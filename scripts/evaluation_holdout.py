import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# Evaluation Function
def evaluate_model(y_test, y_pred, title="Model Evaluation"):
    """
    Evaluates a model based on precomputed true labels (y_test) and predictions (y_pred).
    
    Parameters:
    - y_test: array-like of shape (n_samples,) True labels.
    - y_pred: array-like of shape (n_samples,) Predicted labels.
    - title: str, Title of the evaluation.
    """
    # Metrics calculation
    accuracy = np.mean(y_test == y_pred)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    recall = recall_score(y_test, y_pred, average='binary', zero_division=0)
    f1 = f1_score(y_test, y_pred, average='binary', zero_division=0)
    
    # Printing the evaluation metrics
    print(f"Evaluation Metrics for {title}:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle(f"{title} Evaluation", fontsize=16)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=["Class 0", "Class 1"], yticklabels=["Class 0", "Class 1"], ax=axs[0])
    axs[0].set_title("Confusion Matrix")
    axs[0].set_xlabel("Predicted Label")
    axs[0].set_ylabel("True Label")

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    roc_auc = auc(fpr, tpr)
    axs[1].plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC AUC = {roc_auc:.2f}")
    axs[1].plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random Guess")
    axs[1].set_xlim([0.0, 1.0])
    axs[1].set_ylim([0.0, 1.05])
    axs[1].set_xlabel("False Positive Rate")
    axs[1].set_ylabel("True Positive Rate")
    axs[1].set_title("ROC Curve")
    axs[1].legend(loc="lower right")

    # Displaying the plots
    plt.tight_layout()
    plt.subplots_adjust(top=0.85)
    plt.show()


