import matplotlib.pyplot as plt
import numpy as np
from pyod.models.knn import KNN
from pyod.utils.data import generate_data
from sklearn.metrics import confusion_matrix, roc_curve


def compute_balanced_acc(y_true, y_pred):
    """Matricea de confuzie"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    bal_acc = (tpr + tnr) / 2
    return bal_acc, tn, fp, fn, tp

contamination = 0.1

X_train_2d, X_test_2d, y_train_2d, y_test_2d = generate_data(
    n_train=400,
    n_test=100,
    n_features=2,
    contamination=contamination,
    random_state=42,
)

plt.figure(figsize=(7, 5))
plt.scatter(
    X_train_2d[y_train_2d == 0, 0],
    X_train_2d[y_train_2d == 0, 1],
    c="blue",
    label="Inliers",
    alpha=0.7,
)
plt.scatter(
    X_train_2d[y_train_2d == 1, 0],
    X_train_2d[y_train_2d == 1, 1],
    c="red",
    label="Outliers",
    alpha=0.8,
)
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.grid(True)
plt.show()

clf = KNN(contamination=contamination)
clf.fit(X_train_2d)

y_train_pred = clf.labels_
y_test_pred = clf.predict(X_test_2d)
y_test_scores = clf.decision_function(X_test_2d)

bal_acc_tr, tn_tr, fp_tr, fn_tr, tp_tr = compute_balanced_acc(
    y_train_2d, y_train_pred
)
bal_acc_te, tn_te, fp_te, fn_te, tp_te = compute_balanced_acc(
    y_test_2d, y_test_pred
)

print(
    f"Train -> TN: {tn_tr}, FP: {fp_tr}, FN: {fn_tr}, TP: {tp_tr} | Balanced Acc: {bal_acc_tr:.4f}"
)
print(
    f"Test  -> TN: {tn_te}, FP: {fp_te}, FN: {fn_te}, TP: {tp_te} | Balanced Acc: {bal_acc_te:.4f}"
)

#Curba ROC
fpr, tpr, _ = roc_curve(y_test_2d, y_test_scores)
plt.figure(figsize=(7, 5))
plt.plot(fpr, tpr, color="darkorange", lw=2, label="Curba ROC (KNN)")
plt.plot(
    [0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Sansa aleatoare"
)
plt.xlabel("False Positive Rate (FPR)")
plt.ylabel("True Positive Rate (TPR)")
plt.legend()
plt.grid(True)
plt.show()

X_train_1d, _, y_train_1d, _ = generate_data(
    n_train=1000,
    n_test=0,
    n_features=1,
    contamination=contamination,
    random_state=42,
)
X_flat = X_train_1d.flatten()

mu_1d = np.mean(X_flat)
sigma_1d = np.std(X_flat)
z_scores_1d = np.abs((X_flat - mu_1d) / sigma_1d)

thresh_1d = np.quantile(z_scores_1d, 1 - contamination)
y_pred_1d = (z_scores_1d > thresh_1d).astype(int)

bal_acc_1d, tn, fp, fn, tp = compute_balanced_acc(y_train_1d, y_pred_1d)
print(f"Prag Z-score (1D): {thresh_1d:.4f}")
print(f"Balanced Accuracy (1D): {bal_acc_1d:.4f}")

n_features = 3
X_std, _, y_train_md, _ = generate_data(
    n_train=1000,
    n_test=0,
    n_features=n_features,
    contamination=contamination,
    random_state=42,
)

# Definire medie μ si matrice de covarianta Σ
mu_target = np.array([1.5, -2.0, 3.0])
Sigma_target = np.array([[2.0, 0.5, 0.2], [0.5, 1.0, 0.3], [0.2, 0.3, 1.5]])

# Descompunerea Cholesky: Σ = L * L^T
L = np.linalg.cholesky(Sigma_target)

# Generare esantioane y = L * x + μ
Y_train = np.dot(X_std, L.T) + mu_target

# Z-scores Multidimensionale (Distanta Mahalanobis)
mu_hat = np.mean(Y_train, axis=0)
Sigma_hat = np.cov(Y_train, rowvar=False)
Sigma_inv = np.linalg.inv(Sigma_hat)

diff = Y_train - mu_hat
z_scores_md = np.sqrt(np.sum(np.dot(diff, Sigma_inv) * diff, axis=1))

thresh_md = np.quantile(z_scores_md, 1 - contamination)
y_pred_md = (z_scores_md > thresh_md).astype(int)

bal_acc_md, tn, fp, fn, tp = compute_balanced_acc(y_train_md, y_pred_md)
print(f"Prag Z-score Multidimensional: {thresh_md:.4f}")
print(f"Balanced Accuracy (3D): {bal_acc_md:.4f}")