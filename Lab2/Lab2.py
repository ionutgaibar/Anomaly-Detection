import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score

from pyod.utils.data import generate_data_clusters
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.utils.utility import standardizer
from pyod.models.combination import average, maximization

# Setare seed pentru reproductibilitate
np.random.seed(42)

def compute_leverage(X):
    """Calculeaza diagonala matricei de leverage H = X_design (X_design^T X_design)^(-1) X_design^T."""
    X_design = np.hstack([np.ones((X.shape[0], 1)), X])
    H = X_design @ np.linalg.inv(X_design.T @ X_design) @ X_design.T
    return np.diag(H)

# Model 1D: y = a*x1 + b + epsilon
a_1d, b_1d = 2.0, 1.0
n_regular = 100
n_outliers = 10

sigmas = [0.5, 2.0, 5.0]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, sigma in enumerate(sigmas):
    # Tip 1: Puncte regulate
    x_reg = np.random.uniform(-3, 3, n_regular)
    y_reg = a_1d * x_reg + b_1d + np.random.normal(0, sigma, n_regular)
    
    # Tip 2: Zgomot mare pe x (high variance pe x)
    x_high_x = np.random.uniform(8, 12, n_outliers)
    y_high_x = a_1d * x_high_x + b_1d + np.random.normal(0, sigma, n_outliers)
    
    # Tip 3: Zgomot mare pe y (high variance pe y)
    x_high_y = np.random.uniform(-3, 3, n_outliers)
    y_high_y = a_1d * x_high_y + b_1d + np.random.normal(0, sigma * 5, n_outliers)
    
    # Tip 4: Zgomot mare pe ambele (x si y)
    x_high_xy = np.random.uniform(8, 12, n_outliers)
    y_high_xy = a_1d * x_high_xy + b_1d + np.random.normal(0, sigma * 5, n_outliers)
    
    X_1d = np.concatenate([x_reg, x_high_x, x_high_y, x_high_xy]).reshape(-1, 1)
    y_1d = np.concatenate([y_reg, y_high_x, y_high_y, y_high_xy])
    
    h_ii = compute_leverage(X_1d)
    top_threshold = np.quantile(h_ii, 0.90)
    high_lev_mask = h_ii >= top_threshold
    
    ax = axes[idx]
    ax.scatter(X_1d[~high_lev_mask], y_1d[~high_lev_mask], c='blue', alpha=0.6, label='Leverage mic')
    ax.scatter(X_1d[high_lev_mask], y_1d[high_lev_mask], c='red', s=80, edgecolors='k', label='Leverage mare (Top 10%)')
    ax.set_title(f"1D (sigma^2 = {sigma**2:.1f})")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.legend()
    ax.grid(True)

plt.tight_layout()
plt.show()

# Extindere 2D: y = a*x1 + b*x2 + c + epsilon
a_2d, b_2d, c_2d = 2.0, -1.5, 0.5
X1_reg = np.random.uniform(-3, 3, (100, 1))
X2_reg = np.random.uniform(-3, 3, (100, 1))
X_reg_2d = np.hstack([X1_reg, X2_reg])
X_high_x_2d = np.random.uniform(8, 12, (20, 2))
X_2d_all = np.vstack([X_reg_2d, X_high_x_2d])

h_ii_2d = compute_leverage(X_2d_all)
top_mask_2d = h_ii_2d >= np.quantile(h_ii_2d, 0.90)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_2d_all[~top_mask_2d, 0], X_2d_all[~top_mask_2d, 1], h_ii_2d[~top_mask_2d], c='blue', label='Leverage mic')
ax.scatter(X_2d_all[top_mask_2d, 0], X_2d_all[top_mask_2d, 1], h_ii_2d[top_mask_2d], c='red', s=60, label='Leverage mare')
ax.set_xlabel("X1")
ax.set_ylabel("X2")
ax.set_zlabel("Leverage Score")
ax.set_title("Leverage Scores în 2D")
plt.legend()
plt.show()

X_tr_ex2, X_te_ex2, y_tr_ex2, y_te_ex2 = generate_data_clusters(
    n_train=400,
    n_test=200,
    n_features=2,
    n_clusters=2,
    contamination=0.1,
    random_state=42
)

neighbors_list = [3, 10, 30]

for n_neighbors in neighbors_list:
    clf_knn = KNN(n_neighbors=n_neighbors, contamination=0.1)
    clf_knn.fit(X_tr_ex2)
    
    y_tr_pred = clf_knn.labels_
    y_te_pred = clf_knn.predict(X_te_ex2)
    
    ba_tr = balanced_accuracy_score(y_tr_ex2, y_tr_pred)
    ba_te = balanced_accuracy_score(y_te_ex2, y_te_pred)
    
    print(f"n_neighbors = {n_neighbors:2d} | BA Train: {ba_tr:.4f} | BA Test: {ba_te:.4f}")
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    fig.suptitle(f"KNN Anomaly Detection (n_neighbors = {n_neighbors})", fontsize=14)
    
    axes[0, 0].scatter(X_tr_ex2[:, 0], X_tr_ex2[:, 1], c=y_tr_ex2, cmap='coolwarm', edgecolor='k')
    axes[0, 0].set_title("Ground Truth - Train")
    
    axes[0, 1].scatter(X_tr_ex2[:, 0], X_tr_ex2[:, 1], c=y_tr_pred, cmap='coolwarm', edgecolor='k')
    axes[0, 1].set_title(f"Predicted Train (BA: {ba_tr:.2f})")
    
    axes[1, 0].scatter(X_te_ex2[:, 0], X_te_ex2[:, 1], c=y_te_ex2, cmap='coolwarm', edgecolor='k')
    axes[1, 0].set_title("Ground Truth - Test")
    
    axes[1, 1].scatter(X_te_ex2[:, 0], X_te_ex2[:, 1], c=y_te_pred, cmap='coolwarm', edgecolor='k')
    axes[1, 1].set_title(f"Predicted Test (BA: {ba_te:.2f})")
    
    plt.tight_layout()
    plt.show()

c1, _ = make_blobs(n_samples=200, n_features=2, centers=[[-10, -10]], cluster_std=2.0, random_state=42)
c2, _ = make_blobs(n_samples=100, n_features=2, centers=[[10, 10]], cluster_std=6.0, random_state=42)
X_ex3 = np.vstack([c1, c2])

for k in [5, 20]:
    knn_mod = KNN(n_neighbors=k, contamination=0.07)
    knn_mod.fit(X_ex3)
    
    lof_mod = LOF(n_neighbors=k, contamination=0.07)
    lof_mod.fit(X_ex3)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].scatter(X_ex3[:, 0], X_ex3[:, 1], c=knn_mod.labels_, cmap='coolwarm', edgecolor='k', alpha=0.8)
    axes[0].set_title(f"KNN (n_neighbors = {k})")
    
    axes[1].scatter(X_ex3[:, 0], X_ex3[:, 1], c=lof_mod.labels_, cmap='coolwarm', edgecolor='k', alpha=0.8)
    axes[1].set_title(f"LOF (n_neighbors = {k})")
    
    plt.tight_layout()
    plt.show()

try:
    mat = loadmat('cardio.mat')
    X_cardio = mat['X']
    y_cardio = mat['y'].ravel()
    
    contamination_rate = np.mean(y_cardio)
    
    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(
        X_cardio, y_cardio, test_size=0.3, random_state=42, stratify=y_cardio
    )
    
    X_tr_c_norm, X_te_c_norm = standardizer(X_tr_c, X_te_c)
    
    k_range = np.linspace(30, 120, 10, dtype=int)
    n_models = len(k_range)
    
    tr_scores_mat = np.zeros((X_tr_c.shape[0], n_models))
    te_scores_mat = np.zeros((X_te_c.shape[0], n_models))
    
    for i, k in enumerate(k_range):
        model = KNN(n_neighbors=k, contamination=contamination_rate)
        model.fit(X_tr_c_norm)
        
        tr_scores_mat[:, i] = model.decision_scores_
        te_scores_mat[:, i] = model.decision_function(X_te_c_norm)
        
        ba_tr_ind = balanced_accuracy_score(y_tr_c, model.labels_)
        ba_te_ind = balanced_accuracy_score(y_te_c, model.predict(X_te_c_norm))
        print(f"Model Individual k={k:3d} | BA Train: {ba_tr_ind:.4f} | BA Test: {ba_te_ind:.4f}")
        
    tr_scores_norm, te_scores_norm = standardizer(tr_scores_mat, te_scores_mat)
    
    avg_tr = average(tr_scores_norm)
    avg_te = average(te_scores_norm)
    
    max_tr = maximization(tr_scores_norm)
    max_te = maximization(te_scores_norm)
    
    def evaluate_strategy(tr_scores, te_scores, name):
        thresh = np.quantile(tr_scores, 1 - contamination_rate)
        pred_tr = (tr_scores >= thresh).astype(int)
        pred_te = (te_scores >= thresh).astype(int)
        
        ba_tr_ens = balanced_accuracy_score(y_tr_c, pred_tr)
        ba_te_ens = balanced_accuracy_score(y_te_c, pred_te)
        
        print(f"\n--- Strategie Ansamblu: {name} ---")
        print(f"Balanced Accuracy Train: {ba_tr_ens:.4f}")
        print(f"Balanced Accuracy Test:  {ba_te_ens:.4f}")

    evaluate_strategy(avg_tr, avg_te, "AVERAGE")
    evaluate_strategy(max_tr, max_te, "MAXIMIZATION")

except FileNotFoundError:
    print("Fisierul 'cardio.mat' nu a fost gasit.")