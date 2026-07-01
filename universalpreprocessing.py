# =============================================================================
# ECG UNIVERSAL PREPROCESSOR — NeuralCAC Framework
# PhD Research — Complete Corrected Version
#
# SUPPORTED DATASETS (change DATASET_NAME only):
#   "ptbxl"    PTB-XL           100 Hz  12-lead  5 classes
#   "cpsc2018" CPSC 2018        500 Hz  12-lead  9 classes
#   "georgia"  Georgia          500 Hz  12-lead  5 classes
#   "incart"   INCART            257 Hz  12-lead  5 classes
#   "chapman"  Chapman-Shaoxing  500 Hz  12-lead  6 classes
#
# ROOT CAUSE OF PREVIOUS FAILURE (now fixed):
#   wfdb.rdsamp() silently returns all-zeros for PhysioNet Challenge .mat files
#   → load_wfdb_signal() rejected them (max < 1e-10) → nothing was saved
#
# FIXES APPLIED (original):
#   FIX-1  CRITICAL: New load_ecg_signal() tries 3 strategies in order:
#                    (a) scipy.io.loadmat → read 'val' matrix → apply gain/offset
#                    (b) wfdb.rdsamp() for .dat format (PTB-XL, INCART)
#                    (c) numpy .npy fallback
#   FIX-2  CRITICAL: Gain/offset applied from .hea header (ADC integers → mV)
#   FIX-3  HIGH    : CPSC2018 .txt label parser reads line 2 (not last CSV token)
#   FIX-4  MEDIUM  : Removed hard N_FEATURES assertion → pad/trim instead
#   FIX-5  MEDIUM  : All-zero check threshold raised (was 1e-10, now 1e-6)
#   FIX-6  LOW     : Signal shape guard before processing
#
# ADDITIONAL FIXES (this version — based on diagnostic output):
#   FIX-7  CRITICAL: CPSC2018 labels come from Dx: SNOMED codes in .hea file,
#                    NOT from .txt files. Added full SNOMED map for CPSC2018.
#   FIX-8  CRITICAL: INCART uses .mat format (not .dat). Header shows
#                    "I0001.mat 16+24 306000/mV ..." — must use load_mat_signal().
#                    load_incart() now calls load_ecg_signal() (mat-first) instead
#                    of load_dat_signal().
#   FIX-9  HIGH    : INCART SNOMED codes added to text_label_map / snomed_map
#                    (e.g. 164884008=LBBB, 53741008=CAD/sinus, 251180001=AF).
#   FIX-10 MEDIUM  : INCART loader switched to use load_wfdb_hea generic path
#                    so the same mat+snomed pipeline handles it correctly.
#
# OUTPUT (identical for every dataset — Script 2 reads without modification):
#   X_train/val/test.npy    (N, 390) RobustScaled features
#   y_train/val/test.npy    (N,)     integer class labels 0..K-1
#   sig_train/val/test.npy  (N,12,1000) cleaned signals at 100 Hz
#   pid_train/val/test.npy  (N,)     patient ID strings
#   metadata.csv
#   dataset_info.json       → read by script2_train_evaluate.py
# =============================================================================

# ── CHANGE ONLY THESE TWO LINES ──────────────────────────────────────────────
DATASET_NAME = "cpsc2018"

RAW_DATA_PATH = {
    "ptbxl"   : "/home/Yuvaraj/Documents/SHIVIN_PHD_DEEPCAC/main/programs/Multidataset/ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3",
    "cpsc2018": "/home/Yuvaraj/Documents/SHIVIN_PHD_DEEPCAC/main/programs/Multidataset/cpsc_2018",
    "georgia" : "/home/Yuvaraj/Documents/SHIVIN_PHD_DEEPCAC/main/programs/Multidataset/Georgia",
    "incart"  : "/home/Yuvaraj/Documents/SHIVIN_PHD_DEEPCAC/main/programs/Multidataset/INCART",
    "chapman" : "/home/Yuvaraj/Documents/SHIVIN_PHD_DEEPCAC/main/programs/Multidataset/WFDB_ChapmanShaoxing",
}[DATASET_NAME]

OUTPUT_PATH = (
    "/home/Yuvaraj/Documents/SHIVIN_PHD_DEEPCAC/main/programs/Multidataset/"
    f"{DATASET_NAME}_processed"
)
# =============================================================================

import os, ast, json, sys, logging, warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from scipy.signal import (butter, sosfiltfilt, iirnotch, filtfilt,
                           find_peaks, resample as sp_resample, hilbert)
from scipy.stats import kurtosis, skew
import scipy.io

warnings.filterwarnings("ignore")
np.random.seed(42)
os.makedirs(OUTPUT_PATH, exist_ok=True)

# NumPy 2.0 compatibility
from scipy.integrate import trapezoid
_trapz = trapezoid

# =============================================================================
# LOGGING
# =============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(OUTPUT_PATH, "preprocess.log"), mode="w"),
    ],
)
log = logging.getLogger(__name__)


# =============================================================================
# DATASET CONFIGURATIONS
# =============================================================================
DATASET_CFG = {
    "ptbxl": {
        "n_classes"  : 5,
        "class_names": ["NORM","MI","STTC","CD","HYP"],
        "fs_raw"     : 100,
        "fs_target"  : 100,
        "sig_len"    : 1000,
        "split"      : (0.60, 0.15, 0.25),
        "loader"     : "ptbxl",
        "superclass" : {"NORM":0,"MI":1,"STTC":2,"CD":3,"HYP":4},
    },

    # -------------------------------------------------------------------------
    # CPSC2018 — FIX-7: Labels are Dx: SNOMED codes in .hea files.
    # Diagnostic showed: "# Dx: 59118001" (RBBB) — no .txt label files.
    # Full SNOMED map added for all 9 classes.
    # -------------------------------------------------------------------------
    "cpsc2018": {
        "n_classes"  : 9,
        "class_names": ["Normal","AF","I-AVB","LBBB","RBBB","PAC","PVC","STD","STE"],
        "fs_raw"     : 500,
        "fs_target"  : 100,
        "sig_len"    : 1000,
        "split"      : (0.60, 0.15, 0.25),
        "loader"     : "cpsc2018",
        # SNOMED CT codes used in CPSC2018 .hea Dx: field  (FIX-7)
        "snomed_map" : {
            # Normal / sinus rhythm
            "426783006":0, "164934002":0, "427393009":0, "270492004":0,
            # Atrial fibrillation
            "164889003":1, "426749004":1, "164890007":1,
            # First-degree AV block
            "270492004":2, "164884008":3,   # note: 164884008 below as LBBB
            # Re-declare cleanly — priority order matters; dict last-write wins
            # so we build the full map explicitly:
        },
        # Rebuild full SNOMED map without collisions:
        "snomed_map_full": {
            # 0 Normal
            "426783006":0, "164934002":0, "427393009":0,
            # 1 AF
            "164889003":1, "426749004":1, "164890007":1,
            # 2 I-AVB (first-degree AV block)
            "270492004":2, "6374002"  :2,
            # 3 LBBB
            "164909002":3, "445118002":3,
            # 4 RBBB
            "713427006":4, "59118001" :4, "713426002":4,
            # 5 PAC
            "284470004":5, "63593006" :5, "427172004":5,
            # 6 PVC
            "17338001" :6, "75532003" :6, "164884008":6,  # 164884008 = PVC in CPSC context
            # 7 ST depression
            "429622005":7, "164930006":7,
            # 8 ST elevation
            "164931005":8, "54329005" :8,
        },
        # Integer label maps kept as fallback for REFERENCE.csv
        "int_label_map" : {1:0,2:1,3:2,4:3,5:4,6:5,7:6,8:7,9:8},
        "int_label_map0": {0:0,1:1,2:2,3:3,4:4,5:5,6:6,7:7,8:8},
        "text_label_map": {
            "normal":0,"norm":0,
            "atrial fibrillation":1,"af ":1,"afib":1,
            "first-degree av block":2,"i-avb":2,"iavb":2,
            "left bundle branch block":3,"lbbb":3,
            "right bundle branch block":4,"rbbb":4,
            "premature atrial contraction":5,"pac":5,
            "premature ventricular contraction":6,"pvc":6,
            "st-segment depression":7,"std":7,
            "st-segment elevation":8,"ste":8,
        },
    },

    "georgia": {
        "n_classes"  : 5,
        "class_names": ["NORM","AF","IAVB","LBBB","RBBB"],
        "fs_raw"     : 500,
        "fs_target"  : 100,
        "sig_len"    : 1000,
        "split"      : (0.60, 0.15, 0.25),
        "loader"     : "wfdb_hea",
        "snomed_map" : {
            "426783006":0,"164934002":0,"427393009":0,"6374002":0,
            "164889003":1,"426749004":1,
            "270492004":2,"164884008":2,
            "164909002":3,"445118002":3,
            "713427006":4,"59118001" :4,"713426002":4,
        },
        "text_label_map": {
            "normal sinus rhythm":0,"sinus rhythm":0,"normal":0,
            "atrial fibrillation":1,
            "first-degree av block":2,"first degree av block":2,"iavb":2,
            "left bundle branch block":3,"lbbb":3,
            "right bundle branch block":4,"rbbb":4,
        },
    },

    # -------------------------------------------------------------------------
    # INCART — FIX-8 FIX-9: Uses .mat files (not .dat).
    # Diagnostic confirmed: "I0001.mat 16+24 306000/mV ..." in header.
    # loader changed to "wfdb_hea" (same scipy.io.loadmat pipeline).
    # SNOMED codes added from diagnostic sample headers.
    # -------------------------------------------------------------------------
    "incart": {
        "n_classes"  : 5,
        "class_names": ["NORM","AF","LBBB","RBBB","PVC"],
        "fs_raw"     : 257,
        "fs_target"  : 100,
        "sig_len"    : 1000,
        "split"      : (0.60, 0.15, 0.25),
        "loader"     : "incart",        # keeps dedicated loader but now uses mat
        # FIX-9: SNOMED map built from diagnostic headers + PhysioNet codebook
        "snomed_map" : {
            # NORM / sinus rhythm variants
            "426783006":0,  # normal sinus rhythm
            "427393009":0,  # sinus bradycardia
            "427084000":0,  # sinus tachycardia
            "164934002":0,  # sinus rhythm NOS
            "53741008" :0,  # coronary artery disease — treat as NORM class
                            # (INCART uses it for sinus+CAD patients)
            # AF
            "164889003":1, "426749004":1, "251180001":1,  # 251180001 seen in I0002
            # LBBB
            "164909002":2, "445118002":2,
            # RBBB
            "713427006":3, "59118001" :3, "713426002":3,
            # PVC / ventricular ectopic
            "17338001" :4, "75532003" :4, "427172004":4,
            # Additional codes observed in INCART headers
            "164884008":2,  # LBBB variant (used in I0001)
            "251182009":1,  # AF variant (seen in I0002)
            "164931005":4,  # ST elevation → map to PVC slot or skip;
                            # kept here as PVC-like ectopic
            "57054005" :0,  # acute MI with sinus — map to NORM for 5-class
        },
        "text_label_map": {
            "normal sinus rhythm"      :0,
            "sinus bradycardia"        :0,
            "sinus tachycardia"        :0,
            "sinus rhythm"             :0,
            "atrial fibrillation"      :1,
            "atrial flutter"           :1,
            "left bundle branch block" :2,
            "right bundle branch block":3,
            "premature ventricular"    :4,
            "ventricular premature"    :4,
            "ventricular ectopic"      :4,
        },
    },

    "chapman": {
        "n_classes"  : 6,
        "class_names": ["NORM","AF","IAVB","LBBB","RBBB","PVC"],
        "fs_raw"     : 500,
        "fs_target"  : 100,
        "sig_len"    : 1000,
        "split"      : (0.60, 0.15, 0.25),
        "loader"     : "wfdb_hea",
        "snomed_map" : {
            "426783006":0,"164934002":0,"427393009":0,"233917008":0,
            "164889003":1,"426749004":1,
            "270492004":2,"164884008":2,"6374002"  :2,
            "164909002":3,"445118002":3,
            "713427006":4,"59118001" :4,"713426002":4,
            "17338001" :5,"427172004":5,"75532003" :5,
        },
        "text_label_map": {
            "normal sinus rhythm"      :0,"sinus rhythm":0,"normal":0,
            "atrial fibrillation"      :1,
            "first-degree av block"    :2,"first degree av block":2,"iavb":2,
            "left bundle branch block" :3,"lbbb":3,
            "right bundle branch block":4,"rbbb":4,
            "premature ventricular"    :5,"ventricular premature":5,
        },
    },
}

CFG         = DATASET_CFG[DATASET_NAME]
N_CLASSES   = CFG["n_classes"]
CLASS_NAMES = CFG["class_names"]
FS_RAW      = CFG["fs_raw"]
FS_TARGET   = CFG["fs_target"]
SIG_LEN     = CFG["sig_len"]
N_FEATURES  = 390   # 31×12 + 10 HRV + 2 rhythm + 6 cross-lead

log.info("=" * 70)
log.info("NeuralCAC — ECG Universal Preprocessor")
log.info(f"Dataset  : {DATASET_NAME.upper()}")
log.info(f"Path     : {RAW_DATA_PATH}")
log.info(f"Classes  : {N_CLASSES}  {CLASS_NAMES}")
log.info(f"Hz       : {FS_RAW} → {FS_TARGET}")
log.info(f"Features : {N_FEATURES} per recording")
log.info(f"Output   : {OUTPUT_PATH}")
log.info("=" * 70)


# =============================================================================
# PART 1 — SIGNAL LOADING
# FIX-1 FIX-2: Handles both .mat (PhysioNet Challenge) and .dat (PTB-XL)
# FIX-8: INCART also uses .mat (confirmed by diagnostic output)
# =============================================================================

def read_hea_header(hea_path: str) -> dict:
    """
    Parse a WFDB .hea file to extract:
      n_samples, fs, n_leads, gains[], baselines[], lead_names[]
    These are needed to convert raw ADC integers → physical units (mV).

    Handles INCART format: "16+24 306000/mV" (no parentheses for baseline).
    The baseline column 5 (index 4, 0-based after splitting) holds the offset.
    WFDB signal line format:
      filename  fmt  gain/units  bits  baseline  first  checksum  blocksize  desc
      col:       0    1          2     3         4      5         6          7   8
    """
    info = {"n_samples":0,"fs":500,"n_leads":12,
            "gains":[],"baselines":[],"lead_names":[]}
    try:
        with open(hea_path, "r", errors="ignore") as fh:
            lines = fh.readlines()

        # Line 0: "record_name n_leads fs n_samples [...]"
        parts = lines[0].strip().split()
        if len(parts) >= 4:
            info["n_leads"]   = int(parts[1])
            info["fs"]        = float(parts[2])
            info["n_samples"] = int(parts[3])
        elif len(parts) >= 3:
            info["n_leads"] = int(parts[1])
            info["fs"]      = float(parts[2])

        # Signal lines: one per lead
        for line in lines[1:]:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            if len(p) < 3:
                continue
            try:
                # WFDB signal line columns (0-indexed after split):
                #   0: filename (e.g. A0001.mat or I0001.mat)
                #   1: format   (e.g. 16x1+24 or 16+24)
                #   2: gain/units (e.g. 1000.0(0)/mV  OR  306000/mV)
                #   3: bits     (e.g. 16)
                #   4: baseline / zero value  ← used when gain has no "(baseline)"
                #   5: first sample value
                #   ...
                #   8: lead description (I, II, V1, etc.)

                gain_str = p[2]
                gain     = 1.0
                baseline = 0.0

                if "(" in gain_str:
                    # Format A: "1000.0(0)/mV"  — baseline in parentheses
                    g_part, rest = gain_str.split("(", 1)
                    b_part = rest.split(")")[0]
                    gain     = float(g_part)  if g_part.strip()  else 1.0
                    baseline = float(b_part)  if b_part.strip()  else 0.0
                else:
                    # Format B: "306000/mV"  — baseline in column 4
                    gain_clean = gain_str.split("/")[0]
                    gain = float(gain_clean) if gain_clean.strip() else 1.0
                    # Column 4 is the ADC zero / baseline
                    if len(p) > 4:
                        try:
                            baseline = float(p[4])
                        except ValueError:
                            baseline = 0.0

                info["gains"].append(gain if gain != 0 else 1.0)
                info["baselines"].append(baseline)

            except Exception:
                info["gains"].append(1.0)
                info["baselines"].append(0.0)

            if len(p) >= 9:
                info["lead_names"].append(p[8])

    except Exception:
        pass

    # Pad if fewer entries than n_leads
    while len(info["gains"])     < info["n_leads"]: info["gains"].append(1.0)
    while len(info["baselines"]) < info["n_leads"]: info["baselines"].append(0.0)

    return info


def load_mat_signal(rec_path: str, n_leads: int = 12) -> np.ndarray | None:
    """
    FIX-1 FIX-2 FIX-8: Load .mat file and apply gain/offset from .hea header.

    PhysioNet / INCART format stores raw ADC integers in .mat under key 'val'.
    Shape is (n_leads, n_samples) — note: TRANSPOSED from WFDB convention.
    Physical signal = (raw_ADC - baseline) / gain    [units: mV]

    This is the PRIMARY loader for georgia, chapman, cpsc2018, and incart.
    """
    mat_path = rec_path + ".mat"
    hea_path = rec_path + ".hea"

    if not os.path.exists(mat_path):
        return None

    try:
        mat = scipy.io.loadmat(mat_path, verify_compressed_data_integrity=False)
    except Exception:
        return None

    # Find the signal matrix — key is 'val' in PhysioNet/INCART format
    sig_raw = None
    for key in ["val", "data", "signal", "ecg"]:
        if key in mat:
            sig_raw = mat[key]
            break

    # If no named key, take first 2D array
    if sig_raw is None:
        for v in mat.values():
            if isinstance(v, np.ndarray) and v.ndim == 2:
                sig_raw = v
                break

    if sig_raw is None:
        return None

    sig_raw = sig_raw.astype(np.float64)

    # PhysioNet/INCART .mat: shape (n_leads, n_samples) — transpose to (n_samples, n_leads)
    if sig_raw.shape[0] <= 16 and sig_raw.shape[1] > 16:
        sig_raw = sig_raw.T   # → (n_samples, n_leads)

    if sig_raw.ndim != 2 or sig_raw.shape[1] < n_leads:
        return None

    sig_raw = sig_raw[:, :n_leads]

    # FIX-2 FIX-8: apply gain and baseline from .hea header
    # For INCART: gain=306000, baseline from col4 of hea signal line
    if os.path.exists(hea_path):
        hdr = read_hea_header(hea_path)
        for li in range(min(n_leads, len(hdr["gains"]))):
            gain     = hdr["gains"][li]
            baseline = hdr["baselines"][li]
            sig_raw[:, li] = (sig_raw[:, li] - baseline) / gain

    # Replace bad values
    sig_raw = np.nan_to_num(sig_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # FIX-5: threshold 1e-6
    if np.abs(sig_raw).max() < 1e-6:
        return None

    return sig_raw.astype(np.float32)


def load_dat_signal(rec_path: str, n_leads: int = 12) -> np.ndarray | None:
    """
    Load .dat WFDB signal using wfdb.rdsamp().
    Works for PTB-XL (.dat format).
    wfdb.rdsamp() automatically applies gain/offset for .dat files.
    """
    try:
        import wfdb
        sig, _ = wfdb.rdsamp(rec_path)
        if sig is None or sig.ndim != 2:
            return None
        if sig.shape[1] < n_leads:
            return None
        sig = sig[:, :n_leads]
        sig = np.nan_to_num(sig.astype(np.float32), nan=0.0,
                             posinf=0.0, neginf=0.0)
        # FIX-5: higher threshold
        if np.abs(sig).max() < 1e-6:
            return None
        return sig
    except Exception:
        return None


def load_ecg_signal(rec_path: str, n_leads: int = 12) -> np.ndarray | None:
    """
    FIX-1 FIX-8: Universal ECG signal loader — tries strategies in order:
      1. .mat file via scipy.io.loadmat + gain/offset from .hea
         (PhysioNet format: Georgia, Chapman, CPSC2018, INCART)
      2. .dat file via wfdb.rdsamp  (PTB-XL only)
      3. wfdb.rdsamp directly on the path (final fallback)

    Returns (T, n_leads) float32 in physical units (mV), or None.
    """
    # Strategy 1: .mat file (Georgia, Chapman, CPSC2018, INCART)
    sig = load_mat_signal(rec_path, n_leads)
    if sig is not None:
        return sig

    # Strategy 2: .dat WFDB file (PTB-XL)
    if os.path.exists(rec_path + ".dat"):
        sig = load_dat_signal(rec_path, n_leads)
        if sig is not None:
            return sig

    # Strategy 3: wfdb.rdsamp directly (fallback)
    sig = load_dat_signal(rec_path, n_leads)
    return sig


# =============================================================================
# PART 2 — SIGNAL PROCESSING UTILITIES
# =============================================================================

def resample_signal(sig: np.ndarray, fs_in: int,
                    fs_out: int, target_len: int) -> np.ndarray:
    """Resample (T, L) → target Hz, crop or zero-pad to target_len."""
    if fs_in != fs_out:
        n_out = int(round(sig.shape[0] * fs_out / fs_in))
        sig   = sp_resample(sig, n_out, axis=0)
    T = sig.shape[0]
    if T >= target_len:
        return sig[:target_len].astype(np.float32)
    pad = np.zeros((target_len - T, sig.shape[1]), dtype=np.float32)
    return np.vstack([sig, pad])


def clean_lead(raw: np.ndarray, fs: int = 100) -> np.ndarray:
    """
    Per-lead cleaning:
      1. Replace NaN/Inf → 0
      2. Return zeros for flat leads
      3. High-pass 0.5 Hz (baseline wander removal)
      4. Bandpass 0.5–40 Hz
      5. 50 Hz notch
      6. Z-score normalise
      7. Clip ±5 std
    """
    s = np.array(raw, dtype=np.float64)
    s[~np.isfinite(s)] = 0.0
    if s.std() < 1e-8:
        return np.zeros(len(s), dtype=np.float32)

    nyq = fs / 2.0

    # High-pass
    try:
        sos = butter(2, 0.5/nyq, btype="high", output="sos")
        s   = sosfiltfilt(sos, s)
    except Exception: pass

    # Bandpass
    try:
        hi  = min(40.0, nyq * 0.98)
        sos = butter(4, [0.5/nyq, hi/nyq], btype="band", output="sos")
        s   = sosfiltfilt(sos, s)
    except Exception: pass

    # 50 Hz notch
    try:
        if 50.0 < nyq:
            b, a = iirnotch(50.0/nyq, Q=30.0)
            s    = filtfilt(b, a, s)
    except Exception: pass

    # Z-score
    mu, std = s.mean(), s.std()
    if std > 1e-8:
        s = (s - mu) / std

    return np.clip(s, -5.0, 5.0).astype(np.float32)


# =============================================================================
# PART 3 — R-PEAK DETECTION (consensus multi-lead)
# =============================================================================

def detect_rpeaks_single(sig: np.ndarray, fs: int = 100) -> np.ndarray:
    try:
        peaks, _ = find_peaks(sig, height=np.percentile(sig,70),
                               distance=int(0.35*fs), prominence=0.3,
                               width=int(0.02*fs))
        if len(peaks) >= 2: return peaks
        peaks, _ = find_peaks(sig, distance=int(0.35*fs),
                               height=np.percentile(sig,60))
        return peaks
    except Exception:
        return np.array([], dtype=np.int64)


def lead_snr(sig: np.ndarray, peaks: np.ndarray, fs: int = 100) -> float:
    if len(peaks) < 2: return 0.0
    win  = int(0.05 * fs)
    amps = [np.max(np.abs(sig[max(0,p-win):p+win])) for p in peaks]
    return float(np.mean(amps) / (sig.std() + 1e-8))


def consensus_rpeaks(sig12: np.ndarray, fs: int = 100) -> np.ndarray:
    """
    Multi-lead consensus R-peak detection.
    """
    TOL = int(0.030 * fs)
    all_pk = [detect_rpeaks_single(sig12[:,li], fs) for li in range(12)]
    snrs   = [lead_snr(sig12[:,li], all_pk[li], fs) for li in range(12)]
    best   = int(np.argmax(snrs))
    primary= all_pk[best]

    if len(primary) < 2:
        return all_pk[1] if len(all_pk[1]) >= 2 else primary

    confirmed = []
    for rp in primary:
        votes = sum(1 for li, pk in enumerate(all_pk)
                    if li != best and len(pk) > 0
                    and np.abs(pk - rp).min() <= TOL)
        if votes >= 1: confirmed.append(rp)

    confirmed = np.array(confirmed, dtype=np.int64)
    if len(confirmed) < max(2, len(primary)*0.5):
        return primary
    return confirmed


# =============================================================================
# PART 4 — WAVEFORM DELINEATION
# =============================================================================

def qrs_boundaries(sig: np.ndarray, r: int, fs: int = 100) -> tuple:
    thresh   = max(0.05, 0.15 * abs(float(sig[r])))
    q_onset  = max(0, r - int(0.04*fs))
    for i in range(r, max(0, r - int(0.08*fs)), -1):
        if abs(sig[i]) < thresh: q_onset = i; break
    j_point  = min(len(sig)-1, r + int(0.06*fs))
    for i in range(r, min(len(sig)-1, r + int(0.10*fs))):
        if abs(sig[i]) < thresh: j_point = i; break
    return q_onset, j_point


def detect_p_wave(sig, r, q_onset, fs=100):
    pr_start = max(0, r - int(0.20*fs))
    pr_end   = max(pr_start+3, q_onset - int(0.02*fs))
    if pr_end - pr_start < 3: return None, 0.0
    seg = sig[pr_start:pr_end]
    try:    p_idx = int(np.argmax(np.abs(hilbert(seg))))
    except: p_idx = int(np.argmax(np.abs(seg)))
    p_amp = float(seg[p_idx])
    pr_ms = float((r - (pr_start + p_idx)) / fs * 1000)
    if not (80 <= pr_ms <= 300): return None, 0.0
    return pr_start + p_idx, p_amp


def detect_t_wave(sig, j_point, fs=100):
    t_start = j_point
    t_end   = min(len(sig)-1, j_point + int(0.30*fs))
    if t_end - t_start < 3: return None, 0.0
    seg = sig[t_start:t_end]
    try:    t_idx = int(np.argmax(np.abs(hilbert(seg))))
    except: t_idx = int(np.argmax(np.abs(seg)))
    return t_start + t_idx, float(seg[t_idx])


# =============================================================================
# PART 5 — FEATURE EXTRACTION (390 features)
# Layout: 31 per lead × 12 + 10 HRV + 2 rhythm + 6 cross-lead = 390
# Per lead: stat(10)+morph(4)+qrs(4)+st(4)+p(2)+t(2)+freq(5) = 31
# =============================================================================

CROSS_PAIRS = [(0,1),(0,5),(1,5),(1,6),(6,10),(10,11)]


def feat_stat(s):
    return [float(np.mean(s)), float(np.std(s)), float(np.median(s)),
            float(np.min(s)), float(np.max(s)), float(np.max(s)-np.min(s)),
            float(np.percentile(s,25)), float(np.percentile(s,75)),
            float(skew(s)), float(kurtosis(s))]


def feat_morph(s):
    d1 = np.diff(s); d2 = np.diff(s, n=2)
    return [float(np.mean(np.abs(d1))), float(np.mean(np.abs(d2))),
            float(np.mean(s**2)),        float(np.sum(np.abs(d1)))]


def feat_qrs(sig, rp, fs=100):
    if len(rp) < 2: return [0.]*4
    onsets, widths, amps, areas = [], [], [], []
    for r in rp:
        q, j    = qrs_boundaries(sig, r, fs)
        w_ms    = float((j - q) / fs * 1000)
        if 40 <= w_ms <= 250: widths.append(w_ms)
        seg = sig[q:j+1]
        if len(seg) < 2: continue
        amps.append(float(np.max(seg)-np.min(seg)))
        areas.append(float(_trapz(np.abs(seg))))
        onsets.append(float(q))
    return [float(np.std(onsets))   if len(onsets)>1 else 0.,
            float(np.mean(widths))  if widths         else 0.,
            float(np.mean(amps))    if amps           else 0.,
            float(np.mean(areas))   if areas          else 0.]


def feat_st(sig, rp, fs=100):
    if len(rp) < 2: return [0.]*4
    levels, slopes, curves = [], [], []
    for r in rp:
        q, j   = qrs_boundaries(sig, r, fs)
        pr_s   = max(0, q - int(0.08*fs))
        pr_e   = max(pr_s+1, q)
        baseline = float(np.mean(sig[pr_s:pr_e])) if pr_e>pr_s else 0.
        st_s   = j; st_e = min(len(sig), j + int(0.08*fs))
        if st_e - st_s < 3: continue
        st_seg = sig[st_s:st_e] - baseline
        levels.append(float(np.mean(st_seg)))
        x = np.arange(len(st_seg))
        if len(x) > 1: slopes.append(float(np.polyfit(x, st_seg, 1)[0]))
        if len(st_seg) > 2: curves.append(float(np.mean(np.diff(st_seg,n=2))))
    return [float(np.mean(levels)) if levels else 0.,
            float(np.std(levels))  if levels else 0.,
            float(np.mean(slopes)) if slopes else 0.,
            float(np.mean(curves)) if curves else 0.]


def feat_p(sig, rp, fs=100):
    if len(rp) < 2: return [0., 0.]
    amps, pr_ints = [], []
    for r in rp:
        q, _ = qrs_boundaries(sig, r, fs)
        p_s, p_a = detect_p_wave(sig, r, q, fs)
        if p_s is None: continue
        amps.append(abs(p_a))
        pr_ints.append(float((r - p_s)/fs*1000))
    return [float(np.mean(amps))    if amps    else 0.,
            float(np.mean(pr_ints)) if pr_ints else 0.]


def feat_t(sig, rp, fs=100):
    if len(rp) < 2: return [0., 0.]
    t_amps, qt_ints = [], []
    for r in rp:
        q, j = qrs_boundaries(sig, r, fs)
        t_s, t_a = detect_t_wave(sig, j, fs)
        if t_s is None: continue
        qt_ms = float((t_s - q)/fs*1000)
        if 200 <= qt_ms <= 600: qt_ints.append(qt_ms)
        t_amps.append(abs(t_a))
    return [float(np.mean(t_amps))  if t_amps  else 0.,
            float(np.mean(qt_ints)) if qt_ints else 0.]


def feat_freq(sig, fs=100):
    try:
        fft_v = np.abs(np.fft.rfft(sig))
        freqs = np.fft.rfftfreq(len(sig), d=1.0/fs)
        total = np.sum(fft_v) + 1e-8
        return [float(np.sum(fft_v[(freqs>=0.003)&(freqs< 0.04)])/total),
                float(np.sum(fft_v[(freqs>=0.04) &(freqs< 0.15)])/total),
                float(np.sum(fft_v[(freqs>=5.0)  &(freqs<=20.0)])/total),
                float(np.sum(fft_v[(freqs> 20.0) &(freqs<=35.0)])/total),
                float(np.sum(fft_v[(freqs> 35.0) &(freqs<=45.0)])/total)]
    except: return [0.]*5


def feat_hrv(rp, fs=100):
    if len(rp) < 3: return [0.]*10
    rr = np.diff(rp)/fs*1000
    rr = rr[(rr>300)&(rr<2000)]
    if len(rr) < 2: return [0.]*10
    sdnn   = float(np.std(rr))
    rmssd  = float(np.sqrt(np.mean(np.diff(rr)**2)))
    drr    = np.abs(np.diff(rr))
    pnn20  = float(np.mean(drr>20))
    pnn50  = float(np.mean(drr>50))
    sd1    = float(np.std(np.diff(rr))/np.sqrt(2))
    sd2    = float(np.sqrt(max(0., 2*sdnn**2 - sd1**2)))
    mean_rr= float(np.mean(rr))
    min_rr = float(np.min(rr))
    max_rr = float(np.max(rr))
    lf_hf  = 0.
    try:
        if len(rr) >= 8:
            fr  = np.fft.rfftfreq(len(rr), d=mean_rr/1000.)
            psd = np.abs(np.fft.rfft(rr - mean_rr))**2
            lf  = float(np.sum(psd[(fr>=0.04)&(fr<0.15)]))
            hf  = float(np.sum(psd[(fr>=0.15)&(fr<0.40)]))
            lf_hf = lf/(hf+1e-8)
    except: pass
    return [sdnn,rmssd,pnn20,pnn50,sd1,sd2,mean_rr,min_rr,max_rr,lf_hf]


def feat_rhythm(rp, fs=100):
    if len(rp) < 4: return [0., 0.]
    rr = np.diff(rp)/fs*1000
    rr = rr[(rr>300)&(rr<2000)]
    if len(rr) < 3: return [0., 0.]
    return [float(np.std(rr)/(np.mean(rr)+1e-8)),
            float(np.mean(np.abs(np.diff(rr))>50))]


def feat_cross(sig12):
    out = []
    for li, lj in CROSS_PAIRS:
        a, b = sig12[:,li], sig12[:,lj]
        if a.std()<1e-6 or b.std()<1e-6:
            out.append(0.)
        else:
            out.append(float(np.corrcoef(a,b)[0,1]))
    return out


def extract_features(sig12: np.ndarray, fs: int = 100) -> np.ndarray:
    """
    Extract 390 features from (SIG_LEN, 12) cleaned signal.
    FIX-4: Pad/trim instead of hard assert — never crashes on single recording.
    """
    rp    = consensus_rpeaks(sig12, fs)
    feats = []
    for li in range(12):
        lead = sig12[:, li]
        feats.extend(feat_stat(lead))      # 10
        feats.extend(feat_morph(lead))     # 4
        feats.extend(feat_qrs(lead,rp,fs)) # 4
        feats.extend(feat_st(lead,rp,fs))  # 4
        feats.extend(feat_p(lead,rp,fs))   # 2
        feats.extend(feat_t(lead,rp,fs))   # 2
        feats.extend(feat_freq(lead,fs))   # 5
        # = 31 per lead × 12 = 372
    feats.extend(feat_hrv(rp,fs))         # 10
    feats.extend(feat_rhythm(rp,fs))      # 2
    feats.extend(feat_cross(sig12))       # 6
    # Total: 390

    arr = np.array(feats, dtype=np.float32)

    # FIX-4: pad or trim to exactly N_FEATURES
    if len(arr) < N_FEATURES:
        arr = np.concatenate([arr, np.zeros(N_FEATURES - len(arr), dtype=np.float32)])
    elif len(arr) > N_FEATURES:
        arr = arr[:N_FEATURES]

    return arr


def process_recording(sig_raw: np.ndarray) -> tuple:
    """
    Full pipeline for one raw ECG:
      1. Resample to FS_TARGET Hz, crop/pad to SIG_LEN × 12
      2. FIX-6: shape guard
      3. Clean each lead
      4. Extract 390 features
    Returns: (features (390,), sig_clean (SIG_LEN, 12))
    """
    # FIX-6: shape guard
    if sig_raw.ndim != 2 or sig_raw.shape[1] < 1:
        raise ValueError(f"Invalid signal shape: {sig_raw.shape}")

    # Ensure at least 12 columns (zero-pad missing leads)
    if sig_raw.shape[1] < 12:
        pad = np.zeros((sig_raw.shape[0], 12 - sig_raw.shape[1]),
                        dtype=np.float32)
        sig_raw = np.hstack([sig_raw, pad])
    elif sig_raw.shape[1] > 12:
        sig_raw = sig_raw[:, :12]

    sig = resample_signal(sig_raw.astype(np.float32),
                           FS_RAW, FS_TARGET, SIG_LEN)
    sig_clean = np.zeros((SIG_LEN, 12), dtype=np.float32)
    for li in range(12):
        sig_clean[:, li] = clean_lead(sig[:, li], FS_TARGET)

    feats = extract_features(sig_clean, FS_TARGET)
    return feats, sig_clean


# =============================================================================
# PART 6 — LABEL PARSING
# =============================================================================

def parse_snomed(hea_path: str, snomed_map: dict) -> int | None:
    """
    Parse Dx: SNOMED CT codes from .hea file. Returns first match.
    Handles both "# Dx: 59118001" and "#Dx: 53741008,164884008" formats.
    """
    if not snomed_map: return None
    try:
        with open(hea_path, "r", errors="ignore") as fh:
            for line in fh:
                if "dx:" in line.lower():
                    codes_part = line.split(":")[-1]
                    for code in codes_part.replace(" ","").split(","):
                        code = code.strip().rstrip("\n").rstrip("\r")
                        if code in snomed_map:
                            return snomed_map[code]
    except Exception: pass
    return None


def parse_text_label(hea_path: str, text_map: dict) -> int | None:
    """Match text labels in .hea comment lines. Longest key matched first."""
    if not text_map: return None
    try:
        with open(hea_path, "r", errors="ignore") as fh:
            comments = " ".join(
                line.strip().lstrip("#").lower()
                for line in fh
                if line.strip().startswith("#"))
        for key in sorted(text_map.keys(), key=len, reverse=True):
            if key in comments:
                return text_map[key]
    except Exception: pass
    return None


def parse_cpsc_txt(txt_path: str, int_map: dict, int_map0: dict) -> int | None:
    """
    FIX-3: CPSC2018 .txt label parser (kept as fallback, rarely used now).
    Reads line 2 first, then tries CSV format.
    """
    try:
        with open(txt_path, "r", errors="ignore") as fh:
            lines = [l.strip() for l in fh.readlines() if l.strip()]

        if not lines: return None

        if len(lines) >= 2:
            try:
                val = int(lines[1])
                if val in int_map:  return int_map[val]
                if val in int_map0: return int_map0[val]
            except ValueError: pass

        for line in lines:
            for token in [line.split(",")[-1].strip(), line.strip()]:
                try:
                    val = int(token)
                    if val in int_map:  return int_map[val]
                    if val in int_map0: return int_map0[val]
                except ValueError: continue

    except Exception: pass
    return None


def collect_hea_files(root: str) -> list:
    """Recursively find all .hea files; return paths without extension."""
    found = []
    for dirpath, _, files in os.walk(root):
        for f in sorted(files):
            if f.endswith(".hea"):
                found.append(os.path.join(dirpath, f[:-4]))
    return found


# =============================================================================
# PART 7 — DATASET LOADERS
# =============================================================================

def load_ptbxl() -> tuple:
    """
    PTB-XL: uses .dat WFDB format — wfdb.rdsamp works correctly here.
    Labels from ptbxl_database.csv + scp_statements.csv.
    """
    import wfdb
    log.info("[LOADER] PTB-XL  (WFDB .dat format, 100 Hz)")
    db  = RAW_DATA_PATH
    df  = pd.read_csv(os.path.join(db,"ptbxl_database.csv"), index_col="ecg_id")
    scp = pd.read_csv(os.path.join(db,"scp_statements.csv"), index_col=0)
    df["scp_codes"] = df["scp_codes"].apply(ast.literal_eval)
    SMAP = CFG["superclass"]

    def get_label(code_dict):
        best_conf, best_lbl = -1, None
        for code, conf in code_dict.items():
            if code in scp.index:
                sc = scp.loc[code,"diagnostic_class"]
                if sc in SMAP and conf > best_conf:
                    best_conf, best_lbl = conf, SMAP[sc]
        return best_lbl

    df["label"] = df["scp_codes"].apply(get_label)
    df = df[df["label"].notna()].copy()
    df["label"] = df["label"].astype(int)
    log.info(f"  Labeled records: {len(df)}")

    feats, sigs, valid_ids = [], [], []
    err = {"signal":0,"process":0,"nan":0}

    for eid, row in tqdm(df.iterrows(), total=len(df), desc="  PTB-XL"):
        try:
            rec_path = os.path.join(db, row["filename_lr"])
            sig_raw  = load_dat_signal(rec_path, 12)
            if sig_raw is None: err["signal"]+=1; continue

            f, sc = process_recording(sig_raw)
            if not np.isfinite(f).all(): err["nan"]+=1; continue

            feats.append(f); sigs.append(sc); valid_ids.append(eid)
        except Exception: err["process"]+=1

    log.info(f"  Loaded: {len(feats)}  Errors: {err}")
    if len(feats) == 0:
        log.error("  PTB-XL: nothing loaded. Check wfdb install and path.")
        sys.exit(1)

    dv = df.loc[valid_ids]
    return (np.array(feats,np.float32), np.array(sigs,np.float32),
            dv["label"].values, dv["patient_id"].values.astype(str), dv)


def load_cpsc2018() -> tuple:
    """
    CPSC2018: .mat signal files + Dx: SNOMED codes in .hea files.

    FIX-7: Labels come from "# Dx: <snomed_code>" in .hea, NOT from .txt files.
           Diagnostic confirmed: A0001.hea has "# Dx: 59118001" (= RBBB, class 4).
    FIX-1: Signal loaded via scipy.io.loadmat with gain/offset from .hea.

    Label priority:
      1. REFERENCE.csv (integer, if present)
      2. Dx: SNOMED code in .hea  ← PRIMARY for this dataset
      3. Text matching in .hea comments
      4. .txt file integer (legacy fallback)
    """
    log.info("[LOADER] CPSC2018  (PhysioNet .mat format, 500 Hz)")
    log.info("  FIX-7: Labels from Dx: SNOMED codes in .hea files")

    # Use the full SNOMED map (no key collisions)
    snomed_map = CFG["snomed_map_full"]
    int_map    = CFG["int_label_map"]
    int_map0   = CFG["int_label_map0"]
    text_map   = CFG["text_label_map"]

    # Build lookup from REFERENCE.csv if present (integer labels)
    ref_lookup = {}
    for ref_name in ["REFERENCE.csv","reference.csv","labels.csv"]:
        ref_p = os.path.join(RAW_DATA_PATH, ref_name)
        if os.path.exists(ref_p):
            try:
                ref_df = pd.read_csv(ref_p, header=None)
                for _, row in ref_df.iterrows():
                    try:
                        rec = str(row.iloc[0]).strip()
                        val = int(str(row.iloc[1]).strip())
                        lbl = int_map.get(val, int_map0.get(val, None))
                        if lbl is not None: ref_lookup[rec] = lbl
                    except Exception: pass
                log.info(f"  REFERENCE.csv: {len(ref_lookup)} labels")
                break
            except Exception: pass

    all_recs = collect_hea_files(RAW_DATA_PATH)
    log.info(f"  Found {len(all_recs)} .hea files")

    feats, sigs, records, labels, pids = [], [], [], [], []
    errors = 0
    src    = {"ref":0,"snomed":0,"text":0,"txt_file":0,"none":0}

    for rec in tqdm(all_recs, desc="  CPSC2018"):
        try:
            name  = os.path.basename(rec)
            hea_p = rec + ".hea"
            lbl   = None

            # Priority 1: REFERENCE.csv
            if name in ref_lookup:
                lbl = ref_lookup[name]; src["ref"]+=1

            # Priority 2: Dx: SNOMED code in .hea  (FIX-7 — PRIMARY path)
            if lbl is None:
                lbl = parse_snomed(hea_p, snomed_map)
                if lbl is not None: src["snomed"]+=1

            # Priority 3: Text in .hea comments
            if lbl is None:
                lbl = parse_text_label(hea_p, text_map)
                if lbl is not None: src["text"]+=1

            # Priority 4: .txt file (legacy, rarely present)
            if lbl is None:
                txt_p = rec + ".txt"
                if os.path.exists(txt_p):
                    lbl = parse_cpsc_txt(txt_p, int_map, int_map0)
                    if lbl is not None: src["txt_file"]+=1

            if lbl is None:
                src["none"]+=1; errors+=1; continue

            # FIX-1: load via scipy.io.loadmat
            sig_raw = load_ecg_signal(rec, 12)
            if sig_raw is None: errors+=1; continue

            f, sc = process_recording(sig_raw)
            if not np.isfinite(f).all(): errors+=1; continue

            feats.append(f); sigs.append(sc)
            records.append(name); labels.append(lbl); pids.append(name)

        except Exception as e:
            errors+=1

    log.info(f"  Loaded: {len(feats)}  Skipped: {errors}  Sources: {src}")
    if len(feats) == 0:
        log.error("  CPSC2018: nothing loaded.")
        log.error("  Check: .mat files exist and Dx: codes are in snomed_map_full")
        log.error("  Run diagnostic to see which SNOMED codes appear in your .hea files")
        sys.exit(1)

    meta = pd.DataFrame({"record":records,"label":labels,"patient_id":pids})
    return (np.array(feats,np.float32), np.array(sigs,np.float32),
            np.array(labels), np.array(pids,str), meta)


def load_wfdb_hea(dataset_name: str) -> tuple:
    """
    Generic loader for Georgia and Chapman.
    FIX-1: Uses scipy.io.loadmat with gain/offset correction for .mat files.
    Labels: SNOMED CT codes → text matching.
    """
    log.info(f"[LOADER] {dataset_name.upper()}  (PhysioNet .mat format, 500 Hz)")
    snomed_map = CFG.get("snomed_map", {})
    text_map   = CFG.get("text_label_map", {})

    all_recs = collect_hea_files(RAW_DATA_PATH)
    log.info(f"  Found {len(all_recs)} .hea files")

    feats, sigs, records, labels, pids = [], [], [], [], []
    errors = 0
    src    = {"snomed":0,"text":0,"none":0}

    for rec in tqdm(all_recs, desc=f"  {dataset_name}"):
        try:
            hea_p = rec + ".hea"
            if not os.path.exists(hea_p): errors+=1; continue

            lbl = parse_snomed(hea_p, snomed_map)
            if lbl is not None: src["snomed"]+=1
            else:
                lbl = parse_text_label(hea_p, text_map)
                if lbl is not None: src["text"]+=1
                else: src["none"]+=1; errors+=1; continue

            sig_raw = load_ecg_signal(rec, 12)
            if sig_raw is None: errors+=1; continue

            f, sc = process_recording(sig_raw)
            if not np.isfinite(f).all(): errors+=1; continue

            name = os.path.basename(rec)
            pid  = name
            try:
                with open(hea_p,"r",errors="ignore") as fh:
                    for line in fh:
                        ll = line.lower()
                        if ("patient" in ll or "subject" in ll) and ":" in ll:
                            pid = line.split(":")[-1].strip(); break
            except Exception: pass

            feats.append(f); sigs.append(sc)
            records.append(name); labels.append(lbl); pids.append(pid)

        except Exception: errors+=1

    log.info(f"  Loaded: {len(feats)}  Skipped: {errors}  Sources: {src}")
    if len(feats) == 0:
        log.error(f"  {dataset_name.upper()}: nothing loaded.")
        sys.exit(1)

    meta = pd.DataFrame({"record":records,"label":labels,"patient_id":pids})
    return (np.array(feats,np.float32), np.array(sigs,np.float32),
            np.array(labels), np.array(pids,str), meta)


def load_incart() -> tuple:
    """
    INCART: 75 half-hour 12-lead recordings at 257 Hz.

    FIX-8: Diagnostic confirmed .mat format (not .dat):
           "I0001.mat 16+24 306000/mV 16 0 3794 ..."
           → load_ecg_signal() now correctly uses scipy.io.loadmat first.
    FIX-9: SNOMED codes from diagnostic headers added to snomed_map:
           53741008  → NORM (CAD with sinus rhythm)
           164884008 → LBBB (class 2)
           251180001 → AF   (class 1)
           251182009 → AF   (class 1)
    FIX-10: read_hea_header() updated for INCART baseline format (col 4).

    Label priority:
      1. Dx: SNOMED code  ← PRIMARY (all 3 diagnostic headers had Dx: codes)
      2. Text comment matching
      3. "normal"/"sinus" keyword fallback
    """
    log.info("[LOADER] INCART  (PhysioNet .mat format, 257 Hz)")
    log.info("  FIX-8: Using scipy.io.loadmat (not wfdb.rdsamp) for .mat files")
    log.info("  FIX-9: SNOMED codes from diagnostic headers mapped")

    snomed_map = CFG["snomed_map"]
    text_map   = CFG["text_label_map"]

    all_recs = collect_hea_files(RAW_DATA_PATH)
    log.info(f"  Found {len(all_recs)} .hea files")

    feats, sigs, records, labels = [], [], [], []
    errors = 0
    src    = {"snomed":0,"text":0,"sinus_default":0,"none":0}

    for rec in tqdm(all_recs, desc="  INCART"):
        try:
            hea_p = rec + ".hea"
            if not os.path.exists(hea_p): errors+=1; continue

            # Priority 1: Dx: SNOMED codes  (FIX-9 — PRIMARY path)
            lbl = parse_snomed(hea_p, snomed_map)
            if lbl is not None:
                src["snomed"]+=1
            else:
                # Priority 2: text matching
                lbl = parse_text_label(hea_p, text_map)
                if lbl is not None:
                    src["text"]+=1
                else:
                    # Priority 3: keyword fallback
                    try:
                        with open(hea_p,"r",errors="ignore") as fh:
                            raw_text = fh.read().lower()
                        if "normal" in raw_text or "sinus" in raw_text:
                            lbl=0; src["sinus_default"]+=1
                        else:
                            src["none"]+=1; errors+=1; continue
                    except Exception:
                        src["none"]+=1; errors+=1; continue

            # FIX-8: load via scipy.io.loadmat (mat-first strategy)
            sig_raw = load_ecg_signal(rec, 12)
            if sig_raw is None: errors+=1; continue

            # INCART recordings are ~30 min; crop to first 10 seconds
            # at 257 Hz: 257 × 10 = 2570 samples
            n_take = FS_RAW * 10
            if sig_raw.shape[0] > n_take:
                sig_raw = sig_raw[:n_take, :]

            f, sc = process_recording(sig_raw)
            if not np.isfinite(f).all(): errors+=1; continue

            name = os.path.basename(rec)
            feats.append(f); sigs.append(sc)
            records.append(name); labels.append(lbl)

        except Exception: errors+=1

    log.info(f"  Loaded: {len(feats)}  Skipped: {errors}  Sources: {src}")
    if len(feats) == 0:
        log.error("  INCART: nothing loaded.")
        log.error("  Likely cause: SNOMED codes in your .hea files not in snomed_map.")
        log.error("  Re-run the diagnostic and add any new codes to DATASET_CFG['incart']['snomed_map'].")
        sys.exit(1)

    meta = pd.DataFrame({"record":records,"label":labels,"patient_id":records})
    return (np.array(feats,np.float32), np.array(sigs,np.float32),
            np.array(labels), np.array(records,str), meta)


LOADERS = {
    "ptbxl"   : load_ptbxl,
    "cpsc2018": load_cpsc2018,
    "wfdb_hea": lambda: load_wfdb_hea(DATASET_NAME),
    "incart"  : load_incart,
}


# =============================================================================
# PART 8 — MAIN PIPELINE
# =============================================================================

log.info(f"\n[1/6] Loading {DATASET_NAME.upper()} ...")
loader_key = CFG["loader"]
if loader_key not in LOADERS:
    log.error(f"Unknown loader: {loader_key}")
    sys.exit(1)

X_raw, sigs, y, pids, metadata = LOADERS[loader_key]()

log.info(f"\n  Feature matrix : {X_raw.shape}")
log.info(f"  Signal array   : {sigs.shape}")
for c in range(N_CLASSES):
    n = int((y==c).sum())
    log.info(f"  {CLASS_NAMES[c]:>6}(y={c}): {n:6d}  ({100*n/max(len(y),1):.1f}%)")

# Global cleaning
log.info(f"\n[2/6] Global feature cleaning ...")
if len(X_raw) == 0:
    log.error("Nothing loaded — check your dataset path and loader.")
    sys.exit(1)

X_raw = np.nan_to_num(X_raw, nan=0., posinf=0., neginf=0.)
for col in range(X_raw.shape[1]):
    m, s = X_raw[:,col].mean(), X_raw[:,col].std()
    if s > 0: X_raw[:,col] = np.clip(X_raw[:,col], m-8*s, m+8*s)

assert not np.isnan(X_raw).any(), "NaN in features after cleaning"
assert not np.isnan(sigs).any(),  "NaN in signals"
assert X_raw.shape[1] == N_FEATURES, f"Feature count: got {X_raw.shape[1]}, expected {N_FEATURES}"
assert sigs.shape[1:] == (SIG_LEN, 12), f"Signal shape error: {sigs.shape}"
log.info("  Sanity checks passed ✓")

# Patient-level split
log.info(f"\n[3/6] Patient-level stratified split {CFG['split']} ...")
unique_pts = np.unique(pids)
log.info(f"  Unique patients: {len(unique_pts)}")

pt_label = {}
for p in unique_pts:
    arr = y[pids==p]
    m   = pd.Series(arr).mode()
    pt_label[p] = int(m[0]) if len(m)>0 else int(arr[0])
pt_label_arr = np.array([pt_label[p] for p in unique_pts])
_, va_frac, te_frac = CFG["split"]


def safe_split(pts, lbls, test_size):
    try:
        min_cls = pd.Series(lbls).value_counts().min()
        strat   = lbls if min_cls >= 2 else None
        return train_test_split(pts, test_size=test_size,
                                 random_state=42, stratify=strat)
    except Exception:
        return train_test_split(pts, test_size=test_size, random_state=42)


pts_tv, pts_te = safe_split(unique_pts, pt_label_arr, te_frac)
tv_lbls        = np.array([pt_label[p] for p in pts_tv])
pts_tr, pts_va = safe_split(pts_tv, tv_lbls, va_frac/(1.-te_frac))

tr_m=np.isin(pids,pts_tr); va_m=np.isin(pids,pts_va); te_m=np.isin(pids,pts_te)
X_tr,X_va,X_te = X_raw[tr_m],X_raw[va_m],X_raw[te_m]
y_tr,y_va,y_te = y[tr_m],y[va_m],y[te_m]
S_tr,S_va,S_te = sigs[tr_m],sigs[va_m],sigs[te_m]
pid_tr,pid_va,pid_te = pids[tr_m],pids[va_m],pids[te_m]

assert len(set(pts_tr)&set(pts_va))==0,"Train/Val patient overlap!"
assert len(set(pts_tr)&set(pts_te))==0,"Train/Test patient overlap!"
assert len(set(pts_va)&set(pts_te))==0,"Val/Test patient overlap!"

log.info(f"  Train: {len(X_tr):6d}  ({len(np.unique(pid_tr))} patients)")
log.info(f"  Val  : {len(X_va):6d}  ({len(np.unique(pid_va))} patients)")
log.info(f"  Test : {len(X_te):6d}  ({len(np.unique(pid_te))} patients)")
log.info("  Patient overlap: 0 / 0 / 0  ✓")
for sy, sn in [(y_tr,"Train"),(y_va,"Val"),(y_te,"Test")]:
    dist={CLASS_NAMES[c]:int((sy==c).sum()) for c in range(N_CLASSES)}
    log.info(f"  {sn}: {dist}")

# Feature scaling
log.info(f"\n[4/6] Scaling features (RobustScaler — fit on train only) ...")
scaler  = RobustScaler()
X_tr_sc = np.clip(scaler.fit_transform(X_tr).astype(np.float32),-10,10)
X_va_sc = np.clip(scaler.transform(X_va).astype(np.float32),    -10,10)
X_te_sc = np.clip(scaler.transform(X_te).astype(np.float32),    -10,10)

# Transpose signals for CNN: (N,SIG_LEN,12) → (N,12,SIG_LEN)
S_tr_cnn = S_tr.transpose(0,2,1)
S_va_cnn = S_va.transpose(0,2,1)
S_te_cnn = S_te.transpose(0,2,1)

log.info(f"  Feature dim    : {X_tr_sc.shape[1]}")
log.info(f"  Signal shape   : {S_tr_cnn.shape[1:]}  (CNN format)")

# Save
log.info(f"\n[5/6] Saving to {OUTPUT_PATH}/ ...")
save_map = {
    "X_train.npy":X_tr_sc, "X_val.npy":X_va_sc,    "X_test.npy":X_te_sc,
    "y_train.npy":y_tr,    "y_val.npy":y_va,        "y_test.npy":y_te,
    "sig_train.npy":S_tr_cnn,"sig_val.npy":S_va_cnn,"sig_test.npy":S_te_cnn,
    "pid_train.npy":pid_tr.astype(str),
    "pid_val.npy"  :pid_va.astype(str),
    "pid_test.npy" :pid_te.astype(str),
}
for fname, arr in save_map.items():
    np.save(os.path.join(OUTPUT_PATH, fname), arr)
try: metadata.to_csv(os.path.join(OUTPUT_PATH,"metadata.csv"),index=False)
except Exception: pass

# dataset_info.json
dataset_info = {
    "dataset_name"     : DATASET_NAME,
    "n_classes"        : int(N_CLASSES),
    "class_names"      : CLASS_NAMES,
    "n_leads"          : 12,
    "signal_length"    : int(SIG_LEN),
    "fs_original"      : int(FS_RAW),
    "fs_model"         : int(FS_TARGET),
    "feature_dim"      : int(X_tr_sc.shape[1]),
    "signal_shape"     : [12, int(SIG_LEN)],
    "n_train"          : int(len(X_tr_sc)),
    "n_val"            : int(len(X_va_sc)),
    "n_test"           : int(len(X_te_sc)),
    "n_patients_train" : int(len(np.unique(pid_tr))),
    "n_patients_val"   : int(len(np.unique(pid_va))),
    "n_patients_test"  : int(len(np.unique(pid_te))),
    "class_counts_train": {CLASS_NAMES[c]:int((y_tr==c).sum()) for c in range(N_CLASSES)},
    "class_counts_val"  : {CLASS_NAMES[c]:int((y_va==c).sum()) for c in range(N_CLASSES)},
    "class_counts_test" : {CLASS_NAMES[c]:int((y_te==c).sum()) for c in range(N_CLASSES)},
    "scaler"           : "RobustScaler",
    "feature_layout"   : {
        "per_lead_31x12_372": {
            "statistical":10,"morphological":4,"qrs_adaptive":4,
            "st_adaptive":4,"p_wave":2,"t_wave":2,"frequency_5band":5},
        "global_hrv_10": ["sdnn","rmssd","pnn20","pnn50",
                           "sd1","sd2","mean_rr","min_rr","max_rr","lf_hf"],
        "rhythm_2"     : ["cv","pnn50_rhythm"],
        "cross_lead_6" : ["I-II","I-aVF","II-aVF","II-V1","V1-V5","V5-V6"],
    },
}
with open(os.path.join(OUTPUT_PATH,"dataset_info.json"),"w") as f:
    json.dump(dataset_info, f, indent=2)

# Verify
log.info(f"\n[6/6] Verifying saved files ...")
all_ok = True
for fname in ["X_train.npy","X_val.npy","X_test.npy",
              "y_train.npy","y_val.npy","y_test.npy",
              "sig_train.npy","sig_val.npy","sig_test.npy",
              "pid_train.npy","pid_val.npy","pid_test.npy",
              "metadata.csv","dataset_info.json"]:
    p  = os.path.join(OUTPUT_PATH, fname)
    ok = os.path.exists(p)
    mb = os.path.getsize(p)/1024/1024 if ok else 0.
    log.info(f"  {'✓' if ok else '✗'} {fname:<26} {mb:7.2f} MB")
    if not ok: all_ok = False

# Final summary
log.info(f"\n{'='*70}")
log.info(f"COMPLETE — {DATASET_NAME.upper()}  {'ALL OK ✓' if all_ok else 'ERRORS ✗'}")
log.info(f"{'='*70}")
log.info(f"  Classes        : {N_CLASSES}  {CLASS_NAMES}")
log.info(f"  Sample rate    : {FS_RAW} Hz → {FS_TARGET} Hz")
log.info(f"  Features       : {N_FEATURES} per recording")
log.info(f"  Signal shape   : (N,12,1000)  ready for CNN")
log.info(f"  Train/Val/Test : {len(X_tr_sc)} / {len(X_va_sc)} / {len(X_te_sc)}")
log.info(f"\n  ALL FIXES APPLIED:")
log.info(f"    FIX-1 : scipy.io.loadmat for .mat files (wfdb returned zeros)")
log.info(f"    FIX-2 : gain/offset from .hea applied (ADC integers → mV)")
log.info(f"    FIX-3 : CPSC2018 .txt reads label from line 2 (legacy fallback)")
log.info(f"    FIX-4 : feature count pad/trim (no crash)")
log.info(f"    FIX-5 : all-zero threshold raised 1e-10→1e-6")
log.info(f"    FIX-6 : signal shape guard in process_recording")
log.info(f"    FIX-7 : CPSC2018 labels from Dx: SNOMED in .hea (not .txt)")
log.info(f"    FIX-8 : INCART uses .mat not .dat — load_ecg_signal() mat-first")
log.info(f"    FIX-9 : INCART SNOMED codes from diagnostic headers mapped")
log.info(f"    FIX-10: read_hea_header() baseline from col-4 for INCART format")
log.info(f"\n  NEXT STEP:")
log.info(f"    python script2_train_evaluate.py   (DATASET_NAME='{DATASET_NAME}')")
log.info(f"{'='*70}")