# =============================================================================
# SCRIPT 2 — UNIVERSAL NEURALCAC V2 TRAINING + EVALUATION
# Save as: /home/Yuvaraj/Documents/SHIVIN_PHD_DEEPCAC/main/programs/universalcsc.py
#
# RUNS AFTER universalpreprocessing.py
# WORKS FOR ALL 5 DATASETS — change only DATASET_NAME below
#
# PTB-XL (primary) : full analysis — ablation, stability, statistical tests,
#                    ROC, PR, confusion matrix, t-SNE, cluster viz, grid search
# Others           : core analysis — comparison table, ROC, PR,
#                    confusion matrix, cluster metrics, best model checkpoint
#
# FIX APPLIED: NUM_WORKERS forced to 0. On Windows, multiprocessing uses
# "spawn" instead of "fork", which means every DataLoader worker process
# re-imports this entire script from the top. Because this script has no
# `if __name__ == "__main__":` guard around its top-level training/grid-search
# code, each spawned worker was re-running the whole script (including the
# grid search loop), which tried to spawn MORE workers recursively -> crash:
#   RuntimeError: An attempt has been made to start a new process before
#   the current process has finished its bootstrapping phase...
# Setting NUM_WORKERS = 0 disables DataLoader multiprocessing entirely, which
# is the safest fix for a script structured as top-level code. It also avoids
# problems with sharing CUDA tensors across worker processes (your data is
# already moved onto the GPU with .to(DEVICE) before the DataLoader is built,
# so worker processes would not have helped anyway).
# =============================================================================

# ── CHANGE ONLY THIS LINE ────────────────────────────────────────────────────
DATASET_NAME = "cpsc2018" #"cpsc2018", "cpsc2018",  "georgia", "incart", "ptbxl"
# ─────────────────────────────────────────────────────────────────────────────

import os, sys, json, warnings, random, logging, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import autocast, GradScaler

from sklearn.metrics import (
    roc_auc_score, accuracy_score, f1_score,
    precision_score, recall_score, average_precision_score,
    balanced_accuracy_score, matthews_corrcoef,
    cohen_kappa_score, confusion_matrix,
    roc_curve, precision_recall_curve,
    silhouette_score, davies_bouldin_score,
    calinski_harabasz_score, normalized_mutual_info_score,
    adjusted_rand_score, homogeneity_completeness_v_measure)
from sklearn.preprocessing import label_binarize
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from scipy import stats
from scipy.stats import bootstrap as scipy_bootstrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

try:    import xgboost as xgb;  HAS_XGB = True
except: HAS_XGB = False
try:    import lightgbm as lgb; HAS_LGB = True
except: HAS_LGB = False
try:    from catboost import CatBoostClassifier; HAS_CAT = True
except: HAS_CAT = False


# =============================================================================
# CONFIGURATION — auto-loaded from dataset_info.json
# =============================================================================
BASE_PATH = {
    "ptbxl"   : "/home/Yuvaraj/Documents/SHIVIN_PHD_DEEPCAC/main/programs/Multidataset/ptbxl_processed",
    "chapman" : "/home/Yuvaraj/Documents/SHIVIN_PHD_DEEPCAC/main/programs/Multidataset/chapman_processed",
    "cpsc2018":   r"C:\Users\uttha\OneDrive\Desktop\cac\cpsc2018_processed\cpsc2018_processed",
    "georgia" : "/home/Yuvaraj/Documents/SHIVIN_PHD_DEEPCAC/main/programs/Multidataset/georgia_processed",
    "incart": "/home/Yuvaraj/Documents/SHIVIN_PHD_DEEPCAC/main/programs/Multidataset/incart_processed",
}[DATASET_NAME]

SAVE_PATH  = fr"C:\Users\uttha\OneDrive\Desktop\cac/results/{DATASET_NAME}"
MODEL_PATH = fr"C:\Users\uttha\OneDrive\Desktop\cac/models/{DATASET_NAME}"
os.makedirs(SAVE_PATH, exist_ok=True)
os.makedirs(MODEL_PATH, exist_ok=True)

# Load dataset info written by Script 1
info_path = os.path.join(BASE_PATH, "dataset_info.json")
if not os.path.exists(info_path):
    print(f"ERROR: {info_path} not found. Run script1_load_preprocess.py first!")
    sys.exit(1)

with open(info_path) as f:
    DS_INFO = json.load(f)

N_CLASSES   = DS_INFO["n_classes"]
CLASS_NAMES = DS_INFO["class_names"]
N_LEADS     = DS_INFO["n_leads"]
INPUT_DIM   = DS_INFO["feature_dim"]    # always 245
IS_PRIMARY  = (DATASET_NAME == "ptbxl") # full analysis only for primary

# Training hyperparameters
EMBED_DIM   = 64
K_DEFAULT   = 5
GRAD_CLIP   = 1.0
SEED        = 42
BATCH_SIZE  = 64
EPOCHS      = 80
PATIENCE    = 10
LR          = 1e-3

# Experiment flags (full for PTB-XL, core for others)
RUN_GRID    = True
RUN_ABLATION= IS_PRIMARY
RUN_STABILITY=IS_PRIMARY
RUN_STATS   = IS_PRIMARY
ABL_SEEDS   = [42, 123, 456]
STAB_SEEDS  = [42, 123, 456, 789, 999]
K_LIST      = [2,3,4,5,6] if IS_PRIMARY else [2,3,4,5]
ALPHA_LIST  = [0.5, 1.0, 2.0]
BETA_LIST   = [0.3, 0.5, 1.0]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP= torch.cuda.is_available()

# --- FIX: force single-process DataLoader on Windows to avoid spawn crash ---
NUM_WORKERS = 0


# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(SAVE_PATH, f"{DATASET_NAME}_run.log"), mode="w")])
log = logging.getLogger(__name__)

log.info("="*70)
log.info(f"NEURALCAC V2 — SCRIPT 2: TRAINING & EVALUATION")
log.info(f"Dataset  : {DATASET_NAME.upper()}  (Primary={IS_PRIMARY})")
log.info(f"Classes  : {N_CLASSES}  {CLASS_NAMES}")
log.info(f"Device   : {DEVICE}  AMP={USE_AMP}")
log.info(f"XGB={HAS_XGB}  LGB={HAS_LGB}  CAT={HAS_CAT}")
log.info("="*70)


# =============================================================================
# REPRODUCIBILITY
# =============================================================================
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False

set_seed(SEED)


# =============================================================================
# SECTION 1 — LOAD DATA
# =============================================================================
log.info("\n[1] LOADING DATA")

def load_npy(name, dtype=np.float32):
    p = os.path.join(BASE_PATH, name)
    if not os.path.exists(p):
        log.error(f"Missing: {p}  — run script1 first!")
        sys.exit(1)
    return np.load(p).astype(dtype)

X_train = load_npy("X_train.npy")
X_val   = load_npy("X_val.npy")
X_test  = load_npy("X_test.npy")
y_train = load_npy("y_train.npy", np.int64)
y_val   = load_npy("y_val.npy",   np.int64)
y_test  = load_npy("y_test.npy",  np.int64)
S_train = load_npy("sig_train.npy")
S_val   = load_npy("sig_val.npy")
S_test  = load_npy("sig_test.npy")

# Sanity checks
assert not np.isnan(X_train).any(), "NaN in X_train"
assert set(np.unique(y_train)) <= set(range(N_CLASSES)), "Unknown classes"
log.info(f"  Train:{len(X_train)} Val:{len(X_val)} Test:{len(X_test)}")
log.info(f"  Feature dim:{INPUT_DIM}  Signal:{S_train.shape}")

def to_t(arr, dtype=torch.float32):
    return torch.tensor(arr, dtype=dtype).to(DEVICE)

Xtr_t = to_t(X_train); Xva_t = to_t(X_val); Xte_t = to_t(X_test)
ytr_t = to_t(y_train, torch.long)
"""Str_t = to_t(S_train.transpose(0,2,1))   # (N,12,1000) for CNN
Sva_t = to_t(S_val.transpose(0,2,1))
Ste_t = to_t(S_test.transpose(0,2,1))"""
Str_t = to_t(S_train)   # already (N,12,1000) from script1
Sva_t = to_t(S_val)
Ste_t = to_t(S_test)

results          = {}
all_metrics_dict = {}
PALETTE = ["#2ecc71","#e74c3c","#3498db","#e67e22","#9b59b6",
           "#1abc9c","#f39c12","#8e44ad","#16a085"]


# =============================================================================
# METRICS HELPER
# =============================================================================
def macro_auc(y_true, y_proba):
    yb   = label_binarize(y_true, classes=list(range(N_CLASSES)))
    aucs = [roc_auc_score(yb[:,c], y_proba[:,c])
             for c in range(N_CLASSES) if yb[:,c].sum() > 0]
    return float(np.mean(aucs)) if aucs else 0.5


def compute_metrics(y_true, y_proba):
    y_pred = y_proba.argmax(axis=1)
    yb     = label_binarize(y_true, classes=list(range(N_CLASSES)))
    cm     = confusion_matrix(y_true, y_pred, labels=list(range(N_CLASSES)))
    specs  = []
    for c in range(N_CLASSES):
        tn = cm.sum() - (cm[c,:].sum() + cm[:,c].sum() - cm[c,c])
        fp = cm[:,c].sum() - cm[c,c]
        specs.append(tn/(tn+fp) if (tn+fp)>0 else 0.0)
    pr_scores = [average_precision_score(yb[:,c], y_proba[:,c])
                  for c in range(N_CLASSES) if yb[:,c].sum() > 0]
    try:    mcc   = float(matthews_corrcoef(y_true, y_pred))
    except: mcc   = 0.0
    try:    kappa = float(cohen_kappa_score(y_true, y_pred))
    except: kappa = 0.0
    per_cls = {CLASS_NAMES[c]: round(float(cm[c,c]/max(cm[c,:].sum(),1)),4)
               for c in range(N_CLASSES) if c < len(CLASS_NAMES)}
    return {
        "AUC"          : round(macro_auc(y_true, y_proba), 4),
        "PR-AUC"       : round(float(np.mean(pr_scores)) if pr_scores else 0.0, 4),
        "Accuracy"     : round(float(accuracy_score(y_true, y_pred)), 4),
        "Balanced_Acc" : round(float(balanced_accuracy_score(y_true, y_pred)), 4),
        "F1"           : round(float(f1_score(y_true, y_pred,
                                               average="macro", zero_division=0)), 4),
        "Precision"    : round(float(precision_score(y_true, y_pred,
                                                      average="macro", zero_division=0)), 4),
        "Recall"       : round(float(recall_score(y_true, y_pred,
                                                   average="macro", zero_division=0)), 4),
        "Specificity"  : round(float(np.mean(specs)), 4),
        "MCC"          : round(mcc, 4),
        "Kappa"        : round(kappa, 4),
        "Per_class_acc": per_cls,
    }


# =============================================================================
# MODEL DEFINITIONS
# =============================================================================
class SEBlock(nn.Module):
    def __init__(self, ch, r=8):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc   = nn.Sequential(
            nn.Flatten(), nn.Linear(ch,ch//r), nn.ReLU(),
            nn.Linear(ch//r,ch), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(self.pool(x)).unsqueeze(-1)

class ResBlock1D(nn.Module):
    def __init__(self, in_ch, out_ch, k=7, stride=1, dropout=0.2):
        super().__init__()
        p = k//2
        self.conv = nn.Sequential(
            nn.Conv1d(in_ch,out_ch,k,stride=stride,padding=p,bias=False),
            nn.BatchNorm1d(out_ch), nn.GELU(),
            nn.Conv1d(out_ch,out_ch,k,padding=p,bias=False),
            nn.BatchNorm1d(out_ch))
        self.se   = SEBlock(out_ch)
        self.skip = (nn.Sequential(
            nn.Conv1d(in_ch,out_ch,1,stride=stride,bias=False),
            nn.BatchNorm1d(out_ch))
            if in_ch!=out_ch or stride!=1 else nn.Identity())
        self.act  = nn.GELU()
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        return self.drop(self.act(self.se(self.conv(x))+self.skip(x)))

class CNNEncoder(nn.Module):
    def __init__(self, n_leads=12, embed_dim=EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self.stem   = nn.Sequential(
            nn.Conv1d(n_leads,64,15,padding=7,stride=2,bias=False),
            nn.BatchNorm1d(64), nn.GELU(), nn.MaxPool1d(2))
        self.l1 = nn.Sequential(ResBlock1D(64,64,7), ResBlock1D(64,64,7))
        self.l2 = nn.Sequential(ResBlock1D(64,128,7,stride=2), ResBlock1D(128,128,7))
        self.l3 = nn.Sequential(ResBlock1D(128,256,5,stride=2), ResBlock1D(256,256,5))
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Sequential(
            nn.Flatten(), nn.Linear(256,embed_dim),
            nn.BatchNorm1d(embed_dim), nn.GELU())
    def forward(self, x):
        x=self.stem(x); x=self.l1(x); x=self.l2(x); x=self.l3(x)
        return self.proj(self.pool(x))

class FeatureEncoder(nn.Module):
    def __init__(self, in_dim=INPUT_DIM, embed_dim=EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim,256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256,256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256,embed_dim), nn.LayerNorm(embed_dim), nn.GELU())
    def forward(self, x): return self.net(x)

class Decoder(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, out_dim=INPUT_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim,128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128,256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256,out_dim))
    def forward(self, z): return self.net(z)

class SoftClustering(nn.Module):
    def __init__(self, K, embed_dim=EMBED_DIM, alpha=1.0):
        super().__init__()
        self.K=K; self.alpha=alpha
        self.centroids = nn.Parameter(torch.randn(K, embed_dim))
        nn.init.xavier_uniform_(self.centroids)
    def forward(self, z, temperature=1.0):
        d2 = torch.cdist(z, self.centroids)**2
        q  = (1+d2/self.alpha)**(-(self.alpha+1)/2)
        if temperature != 1.0:
            q = q**(1.0/temperature)
        return q/(q.sum(dim=1,keepdim=True)+1e-9)
    def target_distribution(self, q):
        w = q**2/(q.sum(dim=0,keepdim=True)+1e-9)
        return (w/(w.sum(dim=1,keepdim=True)+1e-9)).detach()
    def cluster_balance_loss(self, q):
        f = q.mean(dim=0)
        return F.kl_div(f.log(), torch.ones_like(f)/self.K, reduction="sum")
    def recover_empty(self, z, q, threshold=0.01):
        freqs = q.mean(dim=0)
        for j in range(self.K):
            if freqs[j].item() < threshold:
                dom  = int(freqs.argmax().item())
                dists= torch.cdist(z, self.centroids[dom:dom+1])[:,0]
                far  = dists.argmax().item()
                with torch.no_grad():
                    self.centroids[j] = z[far]+0.01*torch.randn_like(z[far])

class TopKGating(nn.Module):
    def __init__(self, embed_dim, K, top_k=2):
        super().__init__()
        self.K=K; self.top_k=min(top_k,K)
        self.net = nn.Sequential(
            nn.Linear(embed_dim,64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64,K))
    def forward(self, z):
        logits       = self.net(z)
        tv, ti       = logits.topk(self.top_k, dim=1)
        mask         = torch.full_like(logits, float("-inf"))
        mask.scatter_(1, ti, tv)
        gates        = F.softmax(mask, dim=1)
        load_loss    = self.K*(gates.mean(dim=0)*
                               F.softmax(logits,dim=1).mean(dim=0)).sum()
        return gates, load_loss

class Expert(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, n_classes=N_CLASSES):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim,64), nn.LayerNorm(64), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(64,32), nn.GELU(), nn.Linear(32,n_classes))
    def forward(self, z): return self.net(z)

class AMSoftmax(nn.Module):
    def __init__(self, embed_dim=EMBED_DIM, n_classes=N_CLASSES,
                 scale=30.0, margin=0.35):
        super().__init__()
        self.s=scale; self.m=margin
        self.W=nn.Parameter(torch.randn(n_classes,embed_dim))
        nn.init.xavier_uniform_(self.W)
    def forward(self, z, y):
        zn=F.normalize(z,dim=1); wn=F.normalize(self.W,dim=1)
        cos=zn@wn.T
        oh=torch.zeros_like(cos); oh.scatter_(1,y.long().unsqueeze(1),1.0)
        return F.cross_entropy(self.s*(cos-oh*self.m), y.long(), label_smoothing=0.05)

def cluster_reg(q, z, centroids):
    d2     = torch.cdist(z,centroids)**2
    L_comp = (q*d2).mean()
    K      = centroids.shape[0]
    cn     = F.normalize(centroids,dim=1)
    mask   = ~torch.eye(K,dtype=torch.bool,device=z.device)
    L_sep  = -(cn@cn.T)[mask].mean()
    qm     = q.mean(dim=0)
    L_ent  = (qm*torch.log(qm+1e-9)).sum()
    return 0.5*L_comp + 0.5*L_sep + 0.3*L_ent


# =============================================================================
# TRAINING FUNCTION — universal, works for any dataset
# =============================================================================
def warmup_cosine(opt, warmup_epochs, total_epochs):
    def lr_fn(ep):
        if ep < warmup_epochs:
            return (ep+1)/warmup_epochs
        return 0.5*(1+np.cos(
            np.pi*(ep-warmup_epochs)/max(1,total_epochs-warmup_epochs)))
    return optim.lr_scheduler.LambdaLR(opt, lr_fn)


def init_centroids(encoder, Xin_t, K, seed=42, use_cnn=True):
    encoder.eval()
    with torch.no_grad():
        Z0 = torch.cat([encoder(c) for c in
                         torch.split(Xin_t, 256)]).cpu().numpy()
    km = KMeans(n_clusters=K, n_init=15, random_state=seed).fit(Z0)
    return km.cluster_centers_


def train_neuralcac(use_cnn=True, K_=5,
                     alpha_am=1.0, beta_clust=0.5,
                     lam_recon=0.1, lam_load=0.01,
                     epochs=EPOCHS, patience=PATIENCE,
                     lr=LR, batch_size=BATCH_SIZE,
                     seed=SEED, verbose=True,
                     ckpt_prefix="nc_v2",
                     return_model=False):
    """
    Universal NeuralCAC V2 training.
    Works for any N_CLASSES, any dataset configuration.
    """
    set_seed(seed)

    # Build components
    encoder  = (CNNEncoder(N_LEADS, EMBED_DIM) if use_cnn
                 else FeatureEncoder(INPUT_DIM, EMBED_DIM)).to(DEVICE)
    decoder  = Decoder(EMBED_DIM, INPUT_DIM).to(DEVICE)
    am_fn    = AMSoftmax(EMBED_DIM, N_CLASSES).to(DEVICE)
    sc       = SoftClustering(K_, EMBED_DIM).to(DEVICE)
    gating   = TopKGating(EMBED_DIM, K_, min(2,K_)).to(DEVICE)
    experts  = nn.ModuleList([Expert(EMBED_DIM,N_CLASSES).to(DEVICE)
                               for _ in range(K_)])

    all_params = (list(encoder.parameters()) + list(decoder.parameters()) +
                  list(am_fn.parameters())   + list(sc.parameters()) +
                  list(gating.parameters())  +
                  [p for e in experts for p in e.parameters()])

    opt    = optim.AdamW(all_params, lr=lr, weight_decay=1e-4)
    sched  = warmup_cosine(opt, warmup_epochs=5, total_epochs=epochs)
    scaler = GradScaler() if USE_AMP else None

    Xin_tr = Str_t if use_cnn else Xtr_t
    Xin_va = Sva_t if use_cnn else Xva_t
    Xin_te = Ste_t if use_cnn else Xte_t

    loader = DataLoader(TensorDataset(Xin_tr, Xtr_t, ytr_t),
                        batch_size=batch_size, shuffle=True,
                        num_workers=NUM_WORKERS,
                        pin_memory=False,
                        drop_last=False)
    assert len(loader) > 0, "Empty DataLoader"

    # KMeans centroid init
    cents_np = init_centroids(encoder, Xin_tr, K_, seed, use_cnn)
    with torch.no_grad():
        sc.centroids.data = torch.tensor(cents_np, dtype=torch.float32).to(DEVICE)

    best_val   = 0.0
    best_state = None
    pat        = 0
    history    = {"epoch":[], "loss":[], "val_auc":[], "lr":[]}
    ckpt_path  = os.path.join(MODEL_PATH, f"{ckpt_prefix}_best.pt")

    for epoch in range(epochs):
        encoder.train(); decoder.train(); am_fn.train()
        sc.train(); gating.train()
        for e in experts: e.train()
        ep_loss = 0.0; nb = 0

        # Temperature annealing — starts at 2.0, decays to 1.0
        temperature = max(1.0, 2.0 - epoch/max(1, epochs/2))

        for sig_b, feat_b, y_b in loader:
            if sig_b.shape[0] < 2: continue

            def forward_pass():
                z           = encoder(sig_b)
                q           = sc(z, temperature)
                p           = sc.target_distribution(q)
                gates, L_ld = gating(z)
                w           = F.softmax(torch.log(q+1e-9)+torch.log(gates+1e-9), dim=1)
                el          = torch.stack([e(z) for e in experts], dim=1)
                logits      = (el*w.unsqueeze(-1)).sum(1)
                L_cls   = F.cross_entropy(logits, y_b, label_smoothing=0.05)
                L_am    = am_fn(z, y_b)
                L_kl    = F.kl_div(torch.log(q+1e-9), p, reduction="batchmean")
                L_reg   = cluster_reg(q, z, sc.centroids)
                L_bal   = sc.cluster_balance_loss(q)
                L_recon = F.mse_loss(decoder(z), feat_b) if lam_recon>0 else torch.tensor(0., device=DEVICE)
                L_exp   = torch.stack([F.cross_entropy(e(z),y_b,label_smoothing=0.05)
                                        for e in experts]).mean()
                total = (L_cls + alpha_am*L_am
                         + beta_clust*(L_kl+L_reg+0.1*L_bal)
                         + lam_recon*L_recon + lam_load*L_ld + 0.2*L_exp)
                return total

            if USE_AMP:
                with autocast():
                    total = forward_pass()
                if torch.isnan(total): continue
                opt.zero_grad()
                scaler.scale(total).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP)
                scaler.step(opt); scaler.update()
            else:
                total = forward_pass()
                if torch.isnan(total): continue
                opt.zero_grad(); total.backward()
                torch.nn.utils.clip_grad_norm_(all_params, GRAD_CLIP)
                opt.step()

            ep_loss += total.item(); nb += 1

        sched.step()

        # Empty cluster recovery every 10 epochs
        if (epoch+1) % 10 == 0:
            encoder.eval(); sc.eval()
            with torch.no_grad():
                zc = encoder(Xin_tr[:256])
                qc = sc(zc)
            sc.recover_empty(zc, qc)
            encoder.train(); sc.train()

        if (epoch+1) % 5 != 0: continue

        # Validation
        encoder.eval(); sc.eval(); gating.eval()
        for e in experts: e.eval()
        with torch.no_grad():
            zv = encoder(Xin_va); qv = sc(zv)
            gv, _ = gating(zv)
            wv = F.softmax(torch.log(qv+1e-9)+torch.log(gv+1e-9), dim=1)
            elv = torch.stack([e(zv) for e in experts], dim=1)
            lv  = (elv*wv.unsqueeze(-1)).sum(1)
            pv  = F.softmax(lv, dim=1).cpu().numpy()

        v_auc = macro_auc(y_val, pv)
        cur_lr= opt.param_groups[0]["lr"]
        history["epoch"].append(epoch+1)
        history["loss"].append(ep_loss/max(nb,1))
        history["val_auc"].append(v_auc)
        history["lr"].append(cur_lr)

        if verbose and (epoch+1) % 20 == 0:
            log.info(f"    Ep {epoch+1:3d} | loss:{ep_loss/max(nb,1):.4f} "
                     f"| val:{v_auc:.4f} | lr:{cur_lr:.6f}")

        if v_auc > best_val:
            best_val   = v_auc
            best_state = {
                "encoder" : {k:v.clone() for k,v in encoder.state_dict().items()},
                "sc"      : {k:v.clone() for k,v in sc.state_dict().items()},
                "gating"  : {k:v.clone() for k,v in gating.state_dict().items()},
                "experts" : [{k:v.clone() for k,v in e.state_dict().items()} for e in experts],
                "epoch"   : epoch+1, "val_auc": v_auc,
                "K"       : K_, "alpha_am": alpha_am, "beta_clust": beta_clust}
            torch.save(best_state, ckpt_path)
            pat = 0
        else:
            pat += 1
            if pat >= patience:
                if verbose: log.info(f"    Early stop @ ep {epoch+1}")
                break
        encoder.train(); sc.train(); gating.train()
        for e in experts: e.train()

    # Restore best
    if best_state:
        encoder.load_state_dict(best_state["encoder"])
        sc.load_state_dict(best_state["sc"])
        gating.load_state_dict(best_state["gating"])
        for e, s in zip(experts, best_state["experts"]):
            e.load_state_dict(s)

    # Test predictions
    encoder.eval(); sc.eval(); gating.eval()
    for e in experts: e.eval()
    with torch.no_grad():
        zt = encoder(Xin_te); qt = sc(zt)
        gt, _ = gating(zt)
        wt = F.softmax(torch.log(qt+1e-9)+torch.log(gt+1e-9), dim=1)
        elt = torch.stack([e(zt) for e in experts], dim=1)
        lt  = (elt*wt.unsqueeze(-1)).sum(1)
        pt  = F.softmax(lt, dim=1).cpu().numpy()

    t_metrics = compute_metrics(y_test, pt)
    if verbose:
        log.info(f"\n  Val AUC : {best_val:.4f}")
        log.info(f"  Test AUC: {t_metrics['AUC']:.4f}")

    # Extract embeddings + cluster assignments
    with torch.no_grad():
        all_Z, all_Q = [], []
        for (xb,) in DataLoader(TensorDataset(Xin_tr), batch_size=256):
            zb = encoder(xb)
            all_Z.append(zb.cpu().numpy())
            all_Q.append(sc(zb).cpu().numpy())
        Ztr_all = np.vstack(all_Z); Qtr_all = np.vstack(all_Q)

    hard = Qtr_all.argmax(axis=1)
    cluster_info = []
    for j in range(K_):
        mask = (hard==j)
        if mask.sum() < 2:
            cluster_info.append({"cluster":j,"N":0,"status":"collapsed"})
            continue
        yj  = y_train[mask]
        dom = int(pd.Series(yj).mode()[0])
        cluster_info.append({
            "cluster"   : j, "N": int(mask.sum()),
            "dominant"  : CLASS_NAMES[dom] if dom < len(CLASS_NAMES) else str(dom),
            "confidence": round(float(Qtr_all[mask,j].mean()),4),
            "class_dist": {CLASS_NAMES[c]: int((yj==c).sum())
                            for c in range(N_CLASSES) if c < len(CLASS_NAMES)}})

    if return_model:
        return (t_metrics["AUC"], best_val, t_metrics, history,
                cluster_info, encoder, sc, gating, experts,
                Ztr_all, Qtr_all, pt)
    return t_metrics["AUC"], best_val, t_metrics, history, cluster_info


# =============================================================================
# SECTION 2 — GRID SEARCH
# =============================================================================
log.info("\n[2] GRID SEARCH (K × alpha × beta — selected by val AUC)")

gs_rows    = []
best_v_auc = 0.0
best_cfg   = {"K":K_DEFAULT,"alpha":1.0,"beta":0.5}

for K_ in K_LIST:
    for alpha in ALPHA_LIST:
        for beta in BETA_LIST:
            t, v, _, _, _ = train_neuralcac(
                use_cnn=True, K_=K_, alpha_am=alpha, beta_clust=beta,
                epochs=EPOCHS, patience=PATIENCE, lr=LR, batch_size=BATCH_SIZE,
                seed=SEED, verbose=False,
                ckpt_prefix=f"gs_K{K_}_a{alpha}_b{beta}")
            gs_rows.append({"K":K_,"alpha":alpha,"beta":beta,
                            "val_auc":round(v,4),"test_auc":round(t,4)})
            log.info(f"  K={K_} α={alpha} β={beta} → val:{v:.4f} test:{t:.4f}")
            if v > best_v_auc:
                best_v_auc = v
                best_cfg   = {"K":K_,"alpha":alpha,"beta":beta,"test_auc":t,"val_auc":v}

pd.DataFrame(gs_rows).sort_values("val_auc",ascending=False).to_csv(
    os.path.join(SAVE_PATH,"grid_search.csv"), index=False)

K_BEST = best_cfg["K"]
log.info(f"\n  BEST: K={K_BEST} α={best_cfg['alpha']} β={best_cfg['beta']}")
log.info(f"  Val AUC : {best_cfg['val_auc']:.4f}")

# Full run with best config
(v2_auc, v2_val, v2_metrics, v2_history, v2_clusters,
 v2_enc, v2_sc, v2_gate, v2_experts,
 Ztr_all, Qtr_all, pt_v2) = train_neuralcac(
    use_cnn=True, K_=K_BEST,
    alpha_am=best_cfg["alpha"], beta_clust=best_cfg["beta"],
    lam_recon=0.1, lam_load=0.01,
    epochs=EPOCHS, patience=PATIENCE, lr=LR, batch_size=BATCH_SIZE,
    seed=SEED, verbose=True, ckpt_prefix="nc_v2_best", return_model=True)

results["NeuralCAC V2 (CNN+SE+MoE)"] = v2_auc
all_metrics_dict["NeuralCAC V2"]      = v2_metrics

log.info("\n  NeuralCAC V2 full metrics:")
for k, v in v2_metrics.items():
    if k != "Per_class_acc":
        log.info(f"    {k:<16}: {v:.4f}")
log.info(f"    Per-class: {v2_metrics['Per_class_acc']}")


# =============================================================================
# SECTION 3 — CLUSTER QUALITY METRICS (§17)
# =============================================================================
log.info("\n[3] CLUSTER QUALITY METRICS")

hard_assign = Qtr_all.argmax(axis=1)
sil=dbi=chi=nmi=ari=hom=com=vm=purity = 0.0
try:
    n   = min(5000, len(Ztr_all))
    idx = np.random.choice(len(Ztr_all), n, replace=False)
    sil = float(silhouette_score(Ztr_all[idx], hard_assign[idx]))
    dbi = float(davies_bouldin_score(Ztr_all[idx], hard_assign[idx]))
    chi = float(calinski_harabasz_score(Ztr_all[idx], hard_assign[idx]))
    nmi = float(normalized_mutual_info_score(y_train, hard_assign))
    ari = float(adjusted_rand_score(y_train, hard_assign))
    hom, com, vm = homogeneity_completeness_v_measure(y_train, hard_assign)
    purity = float(np.mean([
        np.bincount(y_train[hard_assign==j]).max() /
        max(int((hard_assign==j).sum()),1)
        for j in np.unique(hard_assign)]))
    log.info(f"  Silhouette : {sil:.4f}   DBI   : {dbi:.4f}   CHI : {chi:.2f}")
    log.info(f"  NMI        : {nmi:.4f}   ARI   : {ari:.4f}   Purity: {purity:.4f}")
    log.info(f"  Homogeneity: {hom:.4f}   Completeness: {com:.4f}   V-measure: {vm:.4f}")
except Exception as e:
    log.warning(f"  Cluster metrics error: {e}")

for ci in v2_clusters:
    if ci.get("N",0) == 0: continue
    cdist = " | ".join([f"{n}:{v}" for n,v in ci.get("class_dist",{}).items()])
    log.info(f"  C{ci['cluster']}: N={ci['N']:5d}  "
             f"dom={ci.get('dominant','?')}  "
             f"conf={ci.get('confidence',0):.3f}  [{cdist}]")

pd.DataFrame(v2_clusters).to_csv(os.path.join(SAVE_PATH,"clusters.csv"), index=False)
np.save(os.path.join(SAVE_PATH,"cluster_assignments.npy"), hard_assign)
np.save(os.path.join(SAVE_PATH,"embeddings_train.npy"),    Ztr_all)
np.save(os.path.join(SAVE_PATH,"test_preds_v2.npy"),       pt_v2)


# =============================================================================
# SECTION 4 — ALL BASELINES (§16)
# =============================================================================
log.info("\n[4] BASELINES")

def run_sklearn(name, clf):
    clf.fit(X_train, y_train)
    pt = clf.predict_proba(X_test)
    # Handle class mismatch (cluster may have fewer classes)
    if pt.shape[1] < N_CLASSES:
        full = np.zeros((len(X_test), N_CLASSES))
        for ci2, cl in enumerate(clf.classes_):
            full[:, int(cl)] = pt[:, ci2]
        pt = full
    m = compute_metrics(y_test, pt)
    results[name] = m["AUC"]
    all_metrics_dict[name] = m
    np.save(os.path.join(SAVE_PATH, f"pred_{name.replace(' ','_')}.npy"), pt)
    log.info(f"  {name:<30} AUC:{m['AUC']:.4f}  "
             f"Acc:{m['Accuracy']:.4f}  F1:{m['F1']:.4f}  MCC:{m['MCC']:.4f}")
    return pt

log.info("\n  [B1] Logistic Regression")
pt_lr = run_sklearn("Logistic Regression",
                     LogisticRegression(max_iter=2000,random_state=SEED,solver="lbfgs"))
log.info("\n  [B2] Random Forest")
run_sklearn("Random Forest",
             RandomForestClassifier(n_estimators=300,random_state=SEED,n_jobs=-1))
log.info("\n  [B3] MLP (static features)")
run_sklearn("MLP (static)",
             MLPClassifier(hidden_layer_sizes=(256,128,64),max_iter=500,
                            random_state=SEED,early_stopping=True,
                            learning_rate_init=1e-3))
if HAS_XGB:
    log.info("\n  [B4] XGBoost")
    try:
        clf = xgb.XGBClassifier(n_estimators=300,max_depth=6,learning_rate=0.1,
                                  random_state=SEED,eval_metric="mlogloss",verbosity=0)
        clf.fit(X_train,y_train,eval_set=[(X_val,y_val)],
                early_stopping_rounds=20,verbose=False)
        pt = clf.predict_proba(X_test)
        m  = compute_metrics(y_test, pt)
        results["XGBoost"] = m["AUC"]; all_metrics_dict["XGBoost"] = m
        log.info(f"  XGBoost AUC:{m['AUC']:.4f}  F1:{m['F1']:.4f}")
    except Exception as e: log.warning(f"  XGBoost: {e}")

if HAS_LGB:
    log.info("\n  [B5] LightGBM")
    try:
        clf = lgb.LGBMClassifier(n_estimators=300,max_depth=6,learning_rate=0.1,
                                   random_state=SEED,verbosity=-1)
        clf.fit(X_train,y_train,eval_set=[(X_val,y_val)],
                callbacks=[lgb.early_stopping(20),lgb.log_evaluation(-1)])
        pt = clf.predict_proba(X_test)
        m  = compute_metrics(y_test, pt)
        results["LightGBM"] = m["AUC"]; all_metrics_dict["LightGBM"] = m
        log.info(f"  LightGBM AUC:{m['AUC']:.4f}  F1:{m['F1']:.4f}")
    except Exception as e: log.warning(f"  LightGBM: {e}")

if HAS_CAT:
    log.info("\n  [B6] CatBoost")
    try:
        clf = CatBoostClassifier(iterations=300,depth=6,learning_rate=0.1,
                                  random_seed=SEED,verbose=0)
        clf.fit(X_train,y_train,eval_set=(X_val,y_val),early_stopping_rounds=20)
        pt = clf.predict_proba(X_test)
        m  = compute_metrics(y_test, pt)
        results["CatBoost"] = m["AUC"]; all_metrics_dict["CatBoost"] = m
        log.info(f"  CatBoost AUC:{m['AUC']:.4f}  F1:{m['F1']:.4f}")
    except Exception as e: log.warning(f"  CatBoost: {e}")

# Original CAC (KMeans + LR) proxy
log.info("\n  [B7] Original CAC (KMeans+LR)")
km_b = KMeans(n_clusters=K_BEST,n_init=10,random_state=SEED).fit(X_train)
pt_cac = pt_lr.copy()
lr_fb  = LogisticRegression(max_iter=1000,random_state=SEED,solver="lbfgs")
lr_fb.fit(X_train, y_train)
pt_cac = lr_fb.predict_proba(X_test).copy()
for j in range(K_BEST):
    mtr=(km_b.predict(X_train)==j); mte=(km_b.predict(X_test)==j)
    if mtr.sum()<10 or mte.sum()<2: continue
    try:
        c2=LogisticRegression(max_iter=1000,random_state=SEED,solver="lbfgs")
        c2.fit(X_train[mtr],y_train[mtr])
        pp=np.zeros((mte.sum(),N_CLASSES))
        for ci2,cl in enumerate(c2.classes_):
            pp[:,int(cl)]=c2.predict_proba(X_test[mte])[:,ci2]
        pt_cac[mte]=pp
    except: pass
m=compute_metrics(y_test,pt_cac)
results["Original CAC (KMeans+LR)"]=m["AUC"]; all_metrics_dict["Original CAC"]=m
log.info(f"  CAC AUC:{m['AUC']:.4f}  F1:{m['F1']:.4f}")

# DCN-Z + LR
log.info("\n  [B8] DCN-Z + LR")
dcn_enc=FeatureEncoder(INPUT_DIM,EMBED_DIM).to(DEVICE)
dcn_dec=Decoder(EMBED_DIM,INPUT_DIM).to(DEVICE)
dcn_opt=optim.AdamW(list(dcn_enc.parameters())+list(dcn_dec.parameters()),lr=2e-3)
dcn_ldr=DataLoader(TensorDataset(Xtr_t),batch_size=128,shuffle=True,num_workers=0)
for _ in range(20):
    dcn_enc.train(); dcn_dec.train()
    for (xb,) in dcn_ldr:
        z=dcn_enc(xb); l=F.mse_loss(dcn_dec(z),xb)
        dcn_opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(list(dcn_enc.parameters())+list(dcn_dec.parameters()),1.0)
        dcn_opt.step()
dcn_enc.eval()
with torch.no_grad():
    Ztr_d=dcn_enc(Xtr_t).cpu().numpy(); Zte_d=dcn_enc(Xte_t).cpu().numpy()
km_d=KMeans(n_clusters=K_BEST,n_init=10,random_state=SEED).fit(Ztr_d)
pt_dcn=lr_fb.predict_proba(X_test).copy()
for j in range(K_BEST):
    mtr=(km_d.predict(Ztr_d)==j); mte=(km_d.predict(Zte_d)==j)
    if mtr.sum()<10 or mte.sum()<2: continue
    try:
        c2=LogisticRegression(max_iter=1000,random_state=SEED,solver="lbfgs")
        c2.fit(Ztr_d[mtr],y_train[mtr])
        pp=np.zeros((mte.sum(),N_CLASSES))
        for ci2,cl in enumerate(c2.classes_):
            pp[:,int(cl)]=c2.predict_proba(Zte_d[mte])[:,ci2]
        pt_dcn[mte]=pp
    except: pass
m=compute_metrics(y_test,pt_dcn)
results["DCN-Z + LR"]=m["AUC"]; all_metrics_dict["DCN-Z + LR"]=m
log.info(f"  DCN AUC:{m['AUC']:.4f}  F1:{m['F1']:.4f}")

# NeuralCAC V1 (feature encoder)
log.info("\n  [B9] NeuralCAC V1 (feature encoder)")
nc1_auc,_,nc1_m,_,_ = train_neuralcac(
    use_cnn=False,K_=K_BEST,alpha_am=1.0,beta_clust=0.5,
    epochs=40,patience=6,lr=2e-3,batch_size=128,
    seed=SEED,verbose=False,ckpt_prefix="nc_v1")
results["NeuralCAC V1 (features)"]=nc1_auc; all_metrics_dict["NeuralCAC V1"]=nc1_m
log.info(f"  NC V1 AUC:{nc1_auc:.4f}  F1:{nc1_m['F1']:.4f}")

# ResNet1D baseline
log.info("\n  [B10] ResNet1D")
class ResBlockSimple(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(ch,ch,7,padding=3),nn.BatchNorm1d(ch),nn.ReLU(),
            nn.Conv1d(ch,ch,7,padding=3),nn.BatchNorm1d(ch))
    def forward(self,x): return F.relu(self.net(x)+x)

class ResNet1DBaseline(nn.Module):
    def __init__(self):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv1d(N_LEADS,64,15,padding=7),nn.BatchNorm1d(64),nn.ReLU(),
            nn.MaxPool1d(2),ResBlockSimple(64),ResBlockSimple(64),nn.MaxPool1d(2),
            nn.Conv1d(64,128,1),ResBlockSimple(128),ResBlockSimple(128),nn.MaxPool1d(2),
            nn.Conv1d(128,256,1),ResBlockSimple(256),nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),nn.Linear(256,128),nn.ReLU(),nn.Dropout(0.5),
            nn.Linear(128,N_CLASSES))
    def forward(self,x): return self.net(x)

rn=ResNet1DBaseline().to(DEVICE)
rn_opt=optim.AdamW(rn.parameters(),lr=LR,weight_decay=1e-4)
rn_sched=warmup_cosine(rn_opt,5,50)
rn_ldr=DataLoader(TensorDataset(Str_t,ytr_t),batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
rn_best=0.; rn_state=None; rn_pat=0
for epoch in range(50):
    rn.train()
    for xb,yb in rn_ldr:
        l=F.cross_entropy(rn(xb),yb,label_smoothing=0.05)
        rn_opt.zero_grad(); l.backward()
        torch.nn.utils.clip_grad_norm_(rn.parameters(),GRAD_CLIP); rn_opt.step()
    rn_sched.step()
    if (epoch+1)%5!=0: continue
    rn.eval()
    with torch.no_grad():
        pv=F.softmax(rn(Sva_t),dim=1).cpu().numpy()
    v_a=macro_auc(y_val,pv)
    if v_a>rn_best:
        rn_best=v_a; rn_state={k:v.clone() for k,v in rn.state_dict().items()}; rn_pat=0
    else:
        rn_pat+=1
        if rn_pat>=8: break
    rn.train()
if rn_state: rn.load_state_dict(rn_state)
rn.eval()
with torch.no_grad():
    pt_rn=F.softmax(rn(Ste_t),dim=1).cpu().numpy()
m=compute_metrics(y_test,pt_rn)
results["ResNet1D (raw ECG)"]=m["AUC"]; all_metrics_dict["ResNet1D"]=m
torch.save(rn.state_dict(),os.path.join(MODEL_PATH,"resnet1d_best.pt"))
log.info(f"  ResNet1D AUC:{m['AUC']:.4f}  F1:{m['F1']:.4f}")


# =============================================================================
# SECTION 5 — FINAL TABLE
# =============================================================================
best_auc = max(results.values())
log.info(f"\n{'='*72}")
log.info(f"FINAL TABLE — {DATASET_NAME.upper()} 5-Class ECG")
log.info(f"{'='*72}")
log.info(f"  {'Method':<35} {'AUC':>7} {'Acc':>7} {'F1':>7} {'MCC':>7} {'PR-AUC':>8}")
log.info("  "+"-"*72)
for method, auc in sorted(results.items(), key=lambda x: x[1]):
    m2  = all_metrics_dict.get(method, all_metrics_dict.get(
          method.split("(")[0].strip(), {}))
    bm  = " ◄" if auc==best_auc else ""
    log.info(f"  {method:<35} {auc:>7.4f} "
             f"{m2.get('Accuracy',0):>7.4f} "
             f"{m2.get('F1',0):>7.4f} "
             f"{m2.get('MCC',0):>7.4f} "
             f"{m2.get('PR-AUC',0):>8.4f}{bm}")
log.info("="*72)


# =============================================================================
# SECTION 6 — PTB-XL ONLY: ABLATION + STABILITY + STATISTICAL TESTS
# =============================================================================
ablation_results = {}
stab_aucs        = []
stat_rows        = []

if RUN_ABLATION:
    log.info("\n[5] ABLATION STUDY (PTB-XL primary)")
    ablation_cfgs = {
        "A1: Baseline LR"           : None,
        "A2: +Feature Encoder"      : {"use_cnn":False,"alpha_am":0.0,"beta_clust":0.0},
        "A3: +AM-Softmax"           : {"use_cnn":False,"alpha_am":1.0,"beta_clust":0.0},
        "A4: +Soft Clustering"      : {"use_cnn":False,"alpha_am":1.0,"beta_clust":0.5},
        "A5: +CNN (SE+Residual)"    : {"use_cnn":True, "alpha_am":1.0,"beta_clust":0.5},
        "A6: Full V2 (all)"         : {"use_cnn":True, "alpha_am":best_cfg["alpha"],
                                        "beta_clust":best_cfg["beta"]},
    }
    for name, cfg in ablation_cfgs.items():
        aucs = []
        for s in ABL_SEEDS:
            try:
                if cfg is None:
                    set_seed(s)
                    lr_s=LogisticRegression(max_iter=1000,random_state=s,solver="lbfgs")
                    lr_s.fit(X_train,y_train)
                    aucs.append(macro_auc(y_test,lr_s.predict_proba(X_test)))
                else:
                    auc_,_,_,_,_ = train_neuralcac(
                        K_=K_BEST, lam_recon=0.1, lam_load=0.01,
                        epochs=30, patience=5,
                        lr=1e-3 if cfg.get("use_cnn",False) else 2e-3,
                        batch_size=64 if cfg.get("use_cnn",False) else 128,
                        seed=s, verbose=False,
                        ckpt_prefix=f"abl_{name[:4]}_{s}", **cfg)
                    aucs.append(auc_)
            except Exception as e:
                log.warning(f"  {name} s={s}: {e}"); aucs.append(0.5)
        ablation_results[name] = aucs
        log.info(f"  {name:<40} {np.mean(aucs):.4f} ± {np.std(aucs):.4f}")

if RUN_STABILITY:
    log.info("\n[6] STABILITY ANALYSIS (PTB-XL primary)")
    use_fast = not torch.cuda.is_available()
    for s in STAB_SEEDS:
        try:
            auc,val,_,_,_ = train_neuralcac(
                use_cnn=(not use_fast), K_=K_BEST,
                alpha_am=best_cfg["alpha"], beta_clust=best_cfg["beta"],
                lam_recon=0.1, lam_load=0.01,
                epochs=50, patience=8,
                lr=1e-3 if not use_fast else 2e-3, batch_size=64,
                seed=s, verbose=False, ckpt_prefix=f"stab_{s}")
            stab_aucs.append(auc)
            log.info(f"  Seed {s}: AUC={auc:.4f}  val={val:.4f}")
        except Exception as e:
            log.warning(f"  Seed {s}: {e}"); stab_aucs.append(0.5)
    log.info(f"\n  Stability: {np.mean(stab_aucs):.4f} ± {np.std(stab_aucs):.4f}")

if RUN_STATS and stab_aucs:
    log.info("\n[7] STATISTICAL TESTING (PTB-XL primary)")
    bl_stat={"LR":[],"RF":[],"MLP":[]}
    for s in STAB_SEEDS:
        set_seed(s)
        for bn, bfn in [("LR", lambda s_: LogisticRegression(max_iter=1000,random_state=s_,solver="lbfgs")),
                         ("RF", lambda s_: RandomForestClassifier(n_estimators=100,random_state=s_)),
                         ("MLP",lambda s_: MLPClassifier(hidden_layer_sizes=(128,64),max_iter=200,
                                                          random_state=s_,early_stopping=True))]:
            try:
                clf=bfn(s); clf.fit(X_train,y_train)
                bl_stat[bn].append(macro_auc(y_test,clf.predict_proba(X_test)))
            except: bl_stat[bn].append(0.5)

    nc_arr=np.array(stab_aucs)
    for method,bl in bl_stat.items():
        bl_arr=np.array(bl)
        try:    t,p=stats.ttest_rel(nc_arr,bl_arr)
        except: t,p=0,1.0
        try:
            diff=nc_arr-bl_arr
            ci_res=scipy_bootstrap((diff,),np.mean,n_resamples=999,
                                    confidence_level=0.95,random_state=SEED)
            ci_str=f"[{ci_res.confidence_interval.low:+.4f},{ci_res.confidence_interval.high:+.4f}]"
        except: ci_str="N/A"
        sig="✓" if p<0.05 else "✗"
        log.info(f"  vs {method:<6}: NC={nc_arr.mean():.4f} BL={bl_arr.mean():.4f} "
                 f"t={t:.3f} p={p:.4f} CI={ci_str} {sig}")
        stat_rows.append({"method":method,
                           "nc_mean":round(float(nc_arr.mean()),4),
                           "bl_mean":round(float(bl_arr.mean()),4),
                           "t_stat":round(float(t),4),"p_val":round(float(p),4),
                           "ci_95":ci_str,"significant":p<0.05})
    pd.DataFrame(stat_rows).to_csv(os.path.join(SAVE_PATH,"stat_tests.csv"),index=False)


# =============================================================================
# SECTION 7 — ALL VISUALISATIONS (§13)
# =============================================================================
log.info("\n[8] GENERATING ALL VISUALISATIONS")

CMAP5=["#2ecc71","#e74c3c","#3498db","#e67e22","#9b59b6",
       "#1abc9c","#f39c12","#8e44ad","#16a085"]

def mcolor(name):
    n=name.lower()
    if "v2" in n:      return "#2ecc71"
    if "resnet" in n:  return "#3498db"
    if "xgb" in n:     return "#e67e22"
    if "light" in n:   return "#e74c3c"
    if "cat" in n:     return "#f1c40f"
    if "forest" in n:  return "#8e44ad"
    if "v1" in n:      return "#27ae60"
    if "dcn" in n:     return "#1abc9c"
    if "cac" in n and "neural" not in n: return "#c0392b"
    return "#95a5a6"

def save_fig(fig, name):
    p=os.path.join(SAVE_PATH,name)
    fig.savefig(p,dpi=150,bbox_inches="tight"); plt.close(fig)
    log.info(f"  Saved: {name}")

yb_test = label_binarize(y_test, classes=list(range(N_CLASSES)))

# ── Fig 1: Main comparison bar chart ─────────────────────────────────────────
sm  = sorted(results.items(), key=lambda x: x[1])
fig,ax = plt.subplots(figsize=(12, max(5,len(sm)*0.55+2)))
bars   = ax.barh([x[0] for x in sm],[x[1] for x in sm],
                  color=[mcolor(x[0]) for x in sm],edgecolor="white",height=0.65)
for bar,auc in zip(bars,[x[1] for x in sm]):
    ax.text(auc+0.001,bar.get_y()+bar.get_height()/2,
            f"{auc:.4f}",va="center",fontsize=9,fontweight="bold")
ax.axvline(x=v2_auc,color="#2ecc71",linestyle="--",linewidth=2,alpha=0.7,
            label=f"NeuralCAC V2 ({v2_auc:.4f})")
span=max([x[1] for x in sm])-min([x[1] for x in sm])
ax.set_xlim(max(0.3,min([x[1] for x in sm])-span*0.15),
             min(1.0,max([x[1] for x in sm])+span*0.1))
ax.set_xlabel("Macro OvR AUC",fontsize=11)
ax.set_title(f"Method Comparison — {DATASET_NAME.upper()}",fontsize=12,fontweight="bold")
ax.legend(fontsize=9); ax.grid(axis="x",alpha=0.3)
ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
plt.tight_layout(); save_fig(fig,"fig_comparison.png")

# ── Fig 2: ROC Curves ────────────────────────────────────────────────────────
fig,axes=plt.subplots(1,2,figsize=(16,7))
pred_items=[("NeuralCAC V2",pt_v2),("ResNet1D",pt_rn),("Logistic Reg",pt_lr)]
for method,preds in pred_items:
    clr=mcolor(method); fprs,tprs=[],[]
    for c in range(N_CLASSES):
        if yb_test[:,c].sum()==0: continue
        fpr,tpr,_=roc_curve(yb_test[:,c],preds[:,c])
        fprs.append(fpr); tprs.append(tpr)
    all_fpr=np.unique(np.concatenate(fprs))
    mean_tpr=np.mean([np.interp(all_fpr,f,t) for f,t in zip(fprs,tprs)],axis=0)
    auc_val=macro_auc(y_test,preds)
    axes[0].plot(all_fpr,mean_tpr,color=clr,linewidth=2,
                  label=f"{method} (AUC={auc_val:.4f})")
axes[0].plot([0,1],[0,1],"k--",alpha=0.4)
axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
axes[0].set_title("Macro-Average ROC Curve"); axes[0].legend(fontsize=8)
axes[0].grid(alpha=0.3)
# PR Curves
for method,preds in pred_items:
    clr=mcolor(method); aps=[]
    for c in range(N_CLASSES):
        if yb_test[:,c].sum()==0: continue
        aps.append(average_precision_score(yb_test[:,c],preds[:,c]))
    axes[1].plot([],[],color=clr,linewidth=2,
                  label=f"{method} (AP={np.mean(aps):.4f})")
axes[1].set_xlabel("Recall"); axes[1].set_ylabel("Precision")
axes[1].set_title("Precision-Recall Curve"); axes[1].legend(fontsize=8)
axes[1].grid(alpha=0.3)
plt.tight_layout(); save_fig(fig,"fig_roc_pr.png")

# ── Fig 3: Confusion Matrix ───────────────────────────────────────────────────
fig,axes=plt.subplots(1,2,figsize=(14,6))
for ax_idx,(method,preds) in enumerate([("NeuralCAC V2",pt_v2),("ResNet1D",pt_rn)]):
    ax=axes[ax_idx]
    cm_raw=confusion_matrix(y_test,preds.argmax(axis=1),
                             labels=list(range(N_CLASSES)),normalize="true")
    im=ax.imshow(cm_raw,cmap="Blues",vmin=0,vmax=1)
    ax.set_xticks(range(N_CLASSES)); ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASS_NAMES,rotation=45,fontsize=9)
    ax.set_yticklabels(CLASS_NAMES,fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {method}",fontweight="bold")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j,i,f"{cm_raw[i,j]:.2f}",ha="center",va="center",
                    fontsize=8,color="white" if cm_raw[i,j]>0.5 else "black")
    plt.colorbar(im,ax=ax,shrink=0.8)
plt.tight_layout(); save_fig(fig,"fig_confusion.png")

# ── Fig 4: Training history ───────────────────────────────────────────────────
if v2_history["epoch"]:
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    ep=v2_history["epoch"]
    axes[0].plot(ep,v2_history["val_auc"],color="#2ecc71",linewidth=2.5,
                  marker="o",ms=4)
    axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("Val AUC")
    axes[0].set_title("Validation AUC — NeuralCAC V2"); axes[0].grid(alpha=0.3)
    axes[1].plot(ep,v2_history["loss"],color="#e74c3c",linewidth=2.5,label="Loss")
    if v2_history.get("lr"):
        ax2=axes[1].twinx()
        ax2.plot(ep,v2_history["lr"],color="#3498db",linestyle="--",label="LR")
        ax2.set_ylabel("LR",color="#3498db")
    axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("Loss")
    axes[1].set_title("Loss + LR (Warmup → Cosine)"); axes[1].legend()
    plt.tight_layout(); save_fig(fig,"fig_training.png")

# ── Fig 5: Cluster visualisation ─────────────────────────────────────────────
valid_cd=[c for c in v2_clusters if c.get("N",0)>0 and "class_dist" in c]
fig,axes=plt.subplots(1,3,figsize=(18,5))
if valid_cd:
    ax=axes[0]; x4=np.arange(len(valid_cd)); w4=0.15
    for ci2,cn in enumerate(CLASS_NAMES):
        vals=[c["class_dist"].get(cn,0)/max(c["N"],1) for c in valid_cd]
        ax.bar(x4+ci2*w4,vals,w4,label=cn,color=CMAP5[ci2%len(CMAP5)],alpha=0.85)
    ax.set_xticks(x4+len(CLASS_NAMES)*w4/2)
    ax.set_xticklabels([f"C{c['cluster']}\nN={c['N']}" for c in valid_cd],fontsize=8)
    ax.set_ylabel("Proportion"); ax.legend(fontsize=8)
    ax.set_title("Cluster Class Distribution"); ax.grid(axis="y",alpha=0.3)
ax=axes[1]
n2=min(300,len(Qtr_all)); idx2=np.random.choice(len(Qtr_all),n2,replace=False)
srt2=np.argsort(y_train[idx2])
im=ax.imshow(Qtr_all[idx2][srt2].T,aspect="auto",cmap="YlOrRd",vmin=0,vmax=1)
ax.set_ylabel("Cluster"); ax.set_xlabel("Patient (sorted by class)")
ax.set_title("Soft Assignment Heatmap")
ax.set_yticks(range(K_BEST)); ax.set_yticklabels([f"C{j}" for j in range(K_BEST)])
plt.colorbar(im,ax=ax,shrink=0.8)
ax=axes[2]
if valid_cd:
    ccs=[c.get("confidence",0) for c in valid_cd]
    ax.bar(range(len(valid_cd)),ccs,
            color=[CMAP5[i%len(CMAP5)] for i in range(len(valid_cd))],edgecolor="white")
    ax.set_xticks(range(len(valid_cd)))
    ax.set_xticklabels([f"C{c['cluster']}" for c in valid_cd])
    ax.set_ylabel("Avg Confidence"); ax.set_title("Per-cluster Confidence")
    ax.grid(axis="y",alpha=0.3)
plt.tight_layout(); save_fig(fig,"fig_clusters.png")

# ── Fig 6: Metrics heatmap ────────────────────────────────────────────────────
key_m_list=[k for k in all_metrics_dict if k in all_metrics_dict]
met_cols=["AUC","Accuracy","Balanced_Acc","F1","MCC","Kappa","PR-AUC"]
hm=np.array([[all_metrics_dict[m].get(c,0) for c in met_cols] for m in key_m_list])
fig,ax=plt.subplots(figsize=(max(10,len(met_cols)*1.5),max(5,len(key_m_list)*0.65+1)))
im=ax.imshow(hm,cmap="RdYlGn",aspect="auto",vmin=0.3,vmax=1.0)
ax.set_xticks(range(len(met_cols))); ax.set_yticks(range(len(key_m_list)))
ax.set_xticklabels(met_cols,fontsize=9,fontweight="bold")
ax.set_yticklabels(key_m_list,fontsize=8)
for i in range(len(key_m_list)):
    for j in range(len(met_cols)):
        ax.text(j,i,f"{hm[i,j]:.3f}",ha="center",va="center",fontsize=7.5,
                color="black" if 0.3<hm[i,j]<0.85 else "white")
plt.colorbar(im,ax=ax,shrink=0.6)
ax.set_title(f"Metrics Heatmap — {DATASET_NAME.upper()}")
plt.tight_layout(); save_fig(fig,"fig_metrics_heatmap.png")

# ── Fig 7: t-SNE ─────────────────────────────────────────────────────────────
try:
    from sklearn.manifold import TSNE
    n3=min(2000,len(Ztr_all)); idx3=np.random.choice(len(Ztr_all),n3,replace=False)
    ts=TSNE(n_components=2,perplexity=30,random_state=SEED).fit_transform(Ztr_all[idx3])
    fig,axes=plt.subplots(1,2,figsize=(14,6))
    for c in range(N_CLASSES):
        m3=(y_train[idx3]==c)
        axes[0].scatter(ts[m3,0],ts[m3,1],c=CMAP5[c%len(CMAP5)],
                         alpha=0.4,s=8,label=CLASS_NAMES[c] if c<len(CLASS_NAMES) else str(c))
    axes[0].set_title("t-SNE — True Classes"); axes[0].legend(fontsize=8)
    hard3=Qtr_all[idx3].argmax(axis=1)
    for j in range(K_BEST):
        m3=(hard3==j)
        axes[1].scatter(ts[m3,0],ts[m3,1],c=CMAP5[j%len(CMAP5)],
                         alpha=0.4,s=8,label=f"C{j}")
    axes[1].set_title(f"t-SNE — Soft Clusters (K={K_BEST})")
    axes[1].legend(fontsize=8)
    plt.tight_layout(); save_fig(fig,"fig_tsne.png")
except Exception as e:
    log.warning(f"  t-SNE skipped: {e}")

# ── Fig 8: PTB-XL extras (ablation + stability + stats) ──────────────────────
if IS_PRIMARY and ablation_results:
    fig=plt.figure(figsize=(20,12))
    gs4=gridspec.GridSpec(2,3,figure=fig,hspace=0.5,wspace=0.38)

    # Ablation
    ax=fig.add_subplot(gs4[0,0])
    an=list(ablation_results.keys()); am_=[np.mean(v) for v in ablation_results.values()]
    as_=[np.std(v) for v in ablation_results.values()]
    ac_=["#bdc3c7","#85c1e9","#5dade2","#82e0aa","#27ae60","#2ecc71"]
    ax.barh(an,am_,xerr=as_,color=ac_[:len(an)],edgecolor="white",height=0.5,capsize=4)
    for i,(m_,s_) in enumerate(zip(am_,as_)):
        ax.text(m_+s_+0.001,i,f"{m_:.4f}",va="center",fontsize=7.5)
    ax.set_xlabel("Test AUC"); ax.set_title("(a) Ablation Study")
    ax.grid(axis="x",alpha=0.3); ax.spines["top"].set_visible(False)

    # Stability
    ax=fig.add_subplot(gs4[0,1])
    if stab_aucs:
        ax.bar(range(len(STAB_SEEDS)),stab_aucs,color="#2ecc71",alpha=0.85,edgecolor="white")
        mn=np.mean(stab_aucs); sd=np.std(stab_aucs)
        ax.axhline(y=mn,color="black",linestyle="--",linewidth=2,
                    label=f"Mean={mn:.4f}±{sd:.4f}")
        ax.set_xticks(range(len(STAB_SEEDS)))
        ax.set_xticklabels([f"S{s}" for s in STAB_SEEDS])
        ax.set_ylabel("Test AUC"); ax.set_title("(b) Stability — 5 Seeds")
        ax.legend(fontsize=8); ax.grid(axis="y",alpha=0.3)
        ax.spines["top"].set_visible(False)

    # Statistical tests
    ax=fig.add_subplot(gs4[0,2])
    if stat_rows and stab_aucs:
        nc_arr2=np.array(stab_aucs)
        meths=[r["method"] for r in stat_rows]
        bl_m_=[r["bl_mean"] for r in stat_rows]
        x5=np.arange(len(meths)); w5=0.35
        ax.bar(x5-w5/2,[nc_arr2.mean()]*len(meths),w5,yerr=[nc_arr2.std()]*len(meths),
                color="#2ecc71",alpha=0.85,label="NeuralCAC V2",capsize=5)
        ax.bar(x5+w5/2,bl_m_,w5,color="#95a5a6",alpha=0.85,label="Baseline",capsize=5)
        for i,row in enumerate(stat_rows):
            ymax=max(nc_arr2.mean()+nc_arr2.std(),bl_m_[i])+0.005
            star=("***" if row["p_val"]<0.001 else "**" if row["p_val"]<0.01
                  else "*" if row["p_val"]<0.05 else "ns")
            ax.text(i,ymax,star,ha="center",fontsize=13,fontweight="bold",
                     color="red" if star!="ns" else "gray")
        ax.set_xticks(x5); ax.set_xticklabels(meths)
        ax.set_ylabel("Test AUC"); ax.legend(fontsize=9)
        ax.set_title("(c) Statistical Tests  (* p<0.05)")
        ax.grid(axis="y",alpha=0.3); ax.spines["top"].set_visible(False)

    # Training history
    ax=fig.add_subplot(gs4[1,0])
    if v2_history["epoch"]:
        ax.plot(v2_history["epoch"],v2_history["val_auc"],
                 color="#2ecc71",linewidth=2.5,marker="o",ms=4)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Val AUC")
        ax.set_title("(d) Training Curve"); ax.grid(alpha=0.3)

    # Soft assignments heatmap (smaller)
    ax=fig.add_subplot(gs4[1,1])
    n4=min(200,len(Qtr_all)); idx4=np.random.choice(len(Qtr_all),n4,replace=False)
    srt4=np.argsort(y_train[idx4])
    im4=ax.imshow(Qtr_all[idx4][srt4].T,aspect="auto",cmap="YlOrRd",vmin=0,vmax=1)
    ax.set_ylabel("Cluster"); ax.set_xlabel("Patient")
    ax.set_title("(e) Soft Assignments"); ax.set_yticks(range(K_BEST))
    ax.set_yticklabels([f"C{j}" for j in range(K_BEST)])
    plt.colorbar(im4,ax=ax,shrink=0.8)

    # Summary text
    ax=fig.add_subplot(gs4[1,2]); ax.axis("off")
    lr_auc  = results.get("Logistic Regression",0)
    xgb_auc = results.get("XGBoost",0)
    summary=(f"NEURALCAC V2 — {DATASET_NAME.upper()}\n"
             f"─────────────────────────────────\n\n"
             f"NeuralCAC V2 : {v2_auc:.4f}\n"
             f"ResNet1D     : {results.get('ResNet1D (raw ECG)',0):.4f}\n"
             f"XGBoost      : {xgb_auc:.4f}\n"
             f"LR           : {lr_auc:.4f}\n\n"
             f"Gain vs LR   : {v2_auc-lr_auc:+.4f}\n"
             f"Gain vs XGB  : {v2_auc-xgb_auc:+.4f}\n\n"
             f"Stability    : {np.mean(stab_aucs) if stab_aucs else 0:.4f}"
             f"±{np.std(stab_aucs) if stab_aucs else 0:.4f}\n"
             f"Best K       : {K_BEST}\n"
             f"Silhouette   : {sil:.4f}\n"
             f"NMI          : {nmi:.4f}\n"
             f"ARI          : {ari:.4f}\n\n"
             f"ALL STAGES COMPLETE:\n"
             f"  Grid search    ✓\n"
             f"  Ablation       ✓\n"
             f"  Stability      ✓\n"
             f"  Stat tests     ✓\n"
             f"  ROC/PR/CM      ✓\n"
             f"  t-SNE          ✓\n"
             f"  Cluster viz    ✓\n"
             f"  Metrics heatmap✓\n"
             f"  All CSVs saved ✓")
    ax.text(0.03,0.97,summary,transform=ax.transAxes,fontsize=7.5,
             va="top",fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4",facecolor="#eafaf1",
                       edgecolor="#27ae60",linewidth=2))
    ax.set_title("(f) Summary")

    fig.suptitle(f"NeuralCAC V2 — {DATASET_NAME.upper()} — Primary Analysis",
                  fontsize=12,fontweight="bold",y=1.01)
    save_fig(fig,"fig_primary_analysis.png")


# =============================================================================
# SECTION 8 — SAVE ALL OUTPUTS (§19)
# =============================================================================
log.info("\n[9] SAVING ALL OUTPUTS")

# Results CSV
pd.DataFrame([(m,a) for m,a in results.items()],
              columns=["Method","AUC"]).sort_values("AUC",ascending=False).to_csv(
    os.path.join(SAVE_PATH,"final_results.csv"), index=False)

# Full metrics CSV
pd.DataFrame([{"Method":k,
               **{mk:mv for mk,mv in v.items() if mk!="Per_class_acc"}}
               for k,v in all_metrics_dict.items()
               ]).sort_values("AUC",ascending=False).to_csv(
    os.path.join(SAVE_PATH,"full_metrics.csv"), index=False)

# Cluster quality
pd.DataFrame([{
    "Silhouette":sil,"DBI":dbi,"CHI":chi,
    "NMI":nmi,"ARI":ari,"Homogeneity":hom,
    "Completeness":com,"V_measure":vm,"Purity":purity
}]).to_csv(os.path.join(SAVE_PATH,"cluster_quality.csv"), index=False)

# Training history
pd.DataFrame(v2_history).to_csv(
    os.path.join(SAVE_PATH,"training_history.csv"), index=False)

# Ablation (PTB-XL only)
if ablation_results:
    pd.DataFrame([{
        "variant":k,"mean_auc":round(np.mean(v),4),
        "std_auc":round(np.std(v),4)}
        for k,v in ablation_results.items()]).to_csv(
        os.path.join(SAVE_PATH,"ablation.csv"), index=False)

# Cross-dataset summary row (for multi-dataset table)
summary_row = {
    "Dataset"     : DATASET_NAME,
    "N_Classes"   : N_CLASSES,
    "Best_K"      : K_BEST,
    "NC_V2_AUC"   : round(v2_auc, 4),
    "ResNet_AUC"  : round(results.get("ResNet1D (raw ECG)",0), 4),
    "XGB_AUC"     : round(results.get("XGBoost",0), 4),
    "LR_AUC"      : round(results.get("Logistic Regression",0), 4),
    "Gain_vs_LR"  : round(v2_auc - results.get("Logistic Regression",0), 4),
    "Silhouette"  : round(sil, 4),
    "NMI"         : round(nmi, 4),
    "NC_V2_F1"    : round(v2_metrics["F1"], 4),
    "NC_V2_MCC"   : round(v2_metrics["MCC"], 4),
}
summary_path = "/home/Yuvaraj/Documents/cross_dataset_summary.csv"
if os.path.exists(summary_path):
    df_sum = pd.read_csv(summary_path)
    df_sum = df_sum[df_sum["Dataset"]!=DATASET_NAME]
    df_sum = pd.concat([df_sum, pd.DataFrame([summary_row])], ignore_index=True)
else:
    df_sum = pd.DataFrame([summary_row])
df_sum.to_csv(summary_path, index=False)
log.info(f"  Cross-dataset summary updated: {summary_path}")

log.info(f"\n  All outputs → {SAVE_PATH}/")
for f in ["final_results.csv","full_metrics.csv","cluster_quality.csv",
          "training_history.csv","clusters.csv","grid_search.csv",
          "stat_tests.csv","ablation.csv"]:
    p = os.path.join(SAVE_PATH, f)
    if os.path.exists(p):
        log.info(f"    {f}")

log.info(f"\n  Models → {MODEL_PATH}/")
log.info(f"  Figures → {SAVE_PATH}/  (fig_*.png)")


# =============================================================================
# FINAL SUMMARY
# =============================================================================
log.info(f"\n{'='*70}")
log.info(f"NEURALCAC V2 — {DATASET_NAME.upper()} — COMPLETE")
log.info(f"{'='*70}")
log.info(f"  NeuralCAC V2 AUC : {v2_auc:.4f}")
log.info(f"  Best method      : {max(results, key=results.get)}")
log.info(f"  Best overall AUC : {best_auc:.4f}")
log.info(f"  Gain vs LR       : {v2_auc-results.get('Logistic Regression',0):+.4f}")
log.info(f"  Best K           : {K_BEST}")
log.info(f"  Silhouette       : {sil:.4f}  NMI: {nmi:.4f}  ARI: {ari:.4f}")
if stab_aucs:
    log.info(f"  Stability        : {np.mean(stab_aucs):.4f} ± {np.std(stab_aucs):.4f}")
log.info(f"\n  To run next dataset:")
log.info(f"    1. Set DATASET_NAME at top of script1_load_preprocess.py")
log.info(f"    2. python script1_load_preprocess.py")
log.info(f"    3. Set DATASET_NAME at top of script2_train_evaluate.py")
log.info(f"    4. python script2_train_evaluate.py")
log.info(f"{'='*70}")
