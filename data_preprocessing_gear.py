
from __future__ import annotations
import os, re, glob, warnings
from dataclasses import dataclass
from typing import Tuple, List, Optional, Dict, OrderedDict as _OD
from collections import OrderedDict

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy import signal
from scipy.signal import butter, filtfilt, hilbert
from scipy.stats import kurtosis, skew

warnings.filterwarnings("ignore")

# ===================== CONFIG =====================
@dataclass
class Config:
    data_dir: str = r"./dataset_shiyantai/gear/L8way_0.2_20_2048"   # 你的 .mat 目录
    file_glob: str = "**/*.mat"
    channels: int = 3
    fs: int = 20480

    # 预处理
    bp_lo: float = 150.0
    bp_hi: float = 6500.0
    bp_order: int = 4

    # 包络扫描
    env_min: float = 200.0
    env_max: float = 6500.0
    env_bands_per_octave: int = 3
    env_bandwidth_ratio: float = 0.6
    env_filter_order: int = 4

    # 滑窗
    sliding_enabled: bool = True
    win_len: int = 1024
    slide_overlap: float = 0.65
    max_slides_per_file: Optional[int] = None

    # 导出
    SAVE_FEATURE_CSV: bool = True
    out_csv: str = "./dataset_shiyantai/gear/L8way_0.2_20_2048/features_L8way_0.2_20_2048.csv"

CFG = Config()

# ====== 文件名解析 ======
VALID_SPEEDS = {750, 850, 960, 1224}
_speed_pat = re.compile(r'(?<!\d)(750|850|960|1224)(?!\d)')

def extract_speed_from_fname(path: str) -> int:
    base = os.path.basename(path)
    m = _speed_pat.search(base)
    return int(m.group(1)) if m else -1

LABEL_MAP = {
    "n": 0,       # 正常
    "gp0": 1,         #
    "gb1": 2, "gb2": 3, "gb3": 4,   #
    "gc1": 5, "gc2": 6, "gc3": 7,   #
}

def extract_label_from_fname(path: str) -> int:
    name = os.path.splitext(os.path.basename(path))[0]
    toks = name.split('-')
    cand = toks[0].lower() if toks else None
    if cand in LABEL_MAP:
        return LABEL_MAP[cand]
    raise ValueError(f"无法从文件名解析标签，请在 LABEL_MAP 中配置：{path}（得到候选='{cand}'）")

# ===================== 信号与特征 =====================
def butter_bandpass(lo: float, hi: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    lo_n = max(1e-6, lo / nyq)
    hi_n = min(0.999, hi / nyq)
    return signal.butter(order, [lo_n, hi_n], btype="band")

def base_cleaning(x: np.ndarray, fs: int, lo: float, hi: float, order: int) -> np.ndarray:
    x = signal.detrend(x, type="linear")
    b, a = butter_bandpass(lo, hi, fs, order)
    return filtfilt(b, a, x)

def env_kurtosis_scan(x: np.ndarray, fs: int, fmin: float, fmax: float,
                      bands_per_octave: int = 3, bw_ratio: float = 1/3.0):
    centers = []
    f = fmin
    while f <= fmax:
        centers.append(f)
        f *= 2 ** (1 / bands_per_octave)
    best_k, best_env, best_band = -np.inf, None, (fmin, fmax)
    for fc in centers:
        bw = fc * bw_ratio
        f1, f2 = max(10.0, fc - bw / 2), min(fs/2 - 10.0, fc + bw / 2)
        if f2 <= f1 + 10:
            continue
        b, a = butter_bandpass(f1, f2, fs, order=4)
        y = filtfilt(b, a, x)
        env = np.abs(hilbert(y))
        k = kurtosis(env, fisher=True, bias=False)
        if k > best_k:
            best_k, best_env, best_band = k, env, (f1, f2)
    return best_env if best_env is not None else np.abs(hilbert(x)), best_band

def envelope_spectrum(env: np.ndarray, fs: int):
    N = len(env)
    win = signal.windows.hann(N, sym=False)
    spec = np.fft.rfft((env - np.mean(env)) * win)
    freqs = np.fft.rfftfreq(N, d=1.0/fs)
    mag = np.abs(spec)
    return freqs, mag

def basic_time_features(x: np.ndarray) -> Dict[str, float]:
    eps = 1e-12
    rms = np.sqrt(np.mean(x**2))
    peak = np.max(np.abs(x))
    return {
        "rms": float(rms),
        "kurt": float(kurtosis(x, fisher=True, bias=False)),
        "skew": float(skew(x, bias=False)),
        "crest_factor": float(peak / (rms + eps)),
        "shape_factor": float((np.mean(np.abs(x)) + eps) / (np.sqrt(np.mean(x**2)) + eps)),
        "impulse_factor": float(peak / (np.mean(np.abs(x)) + eps)),
    }

def band_energy_features(freqs: np.ndarray, mag: np.ndarray, bands: List[Tuple[float, float]], prefix: str):
    out = {}
    for i, (f1, f2) in enumerate(bands):
        idx = (freqs >= f1) & (freqs < f2)
        out[f"{prefix}_band{i}_energy"] = float(np.sum(mag[idx]**2))
    return out

def top_peaks(freqs: np.ndarray, mag: np.ndarray, k: int = 3, min_dist_hz: float = 5.0):
    idx, _ = signal.find_peaks(mag)
    peaks = sorted([(freqs[i], mag[i]) for i in idx], key=lambda x: x[1], reverse=True)
    out = []
    for f, m in peaks:
        if all(abs(f - f0) >= min_dist_hz for f0, _ in out):
            out.append((f, m))
        if len(out) >= k:
            break
    return out

# ===================== 滑窗 =====================
def yield_segments_from_mat(mat: dict, cfg: Config):
    """
    读取变量 'data'：形状应为 (channels, N) = (3, 6144)
    按 win_len/overlap 产生 (channels, win_len) 片段
    """
    if "data" not in mat:
        raise ValueError("MAT 中未找到变量 'data'")
    arr = np.array(mat["data"])
    if arr.ndim != 2 or arr.shape[0] != cfg.channels:
        raise ValueError(f"'data' 形状应为 ({cfg.channels}, N)，实际 {arr.shape}")
    L = arr.shape[1]
    win_len = cfg.win_len
    if L < win_len:
        return
    hop = max(1, int(win_len * (1.0 - cfg.slide_overlap))) if cfg.sliding_enabled else win_len

    emitted = 0
    for start in range(0, L - win_len + 1, hop):
        if cfg.max_slides_per_file is not None and emitted >= cfg.max_slides_per_file:
            break
        yield emitted, arr[:, start:start+win_len].astype(float)
        emitted += 1

# ===================== 单窗口→特征 =====================
def process_one_window(xcxN: np.ndarray, cfg: Config) -> Dict[str, float]:
    assert xcxN.shape[0] == cfg.channels == 3
    fs = cfg.fs
    feats: Dict[str, float] = OrderedDict()
    for ch in range(cfg.channels):
        x = xcxN[ch, :].astype(float)
        x = base_cleaning(x, fs, cfg.bp_lo, cfg.bp_hi, cfg.bp_order)
        env, band = env_kurtosis_scan(x, fs, cfg.env_min, cfg.env_max,
                                      cfg.env_bands_per_octave, cfg.env_bandwidth_ratio)
        f_env, m_env = envelope_spectrum(env, fs)
        # 时域 6
        for k, v in basic_time_features(x).items():
            feats[f"ch{ch}_{k}"] = v
        # 选带上下界 2
        feats[f"ch{ch}_env_band_lo"], feats[f"ch{ch}_env_band_hi"] = band
        # 主峰 3×2
        peaks = top_peaks(f_env, m_env, k=3)
        for i in range(1, 4):
            pf, pm = (peaks[i-1] if i <= len(peaks) else (0.0, 0.0))
            feats[f"ch{ch}_env_peak{i}_freq"] = float(pf)
            feats[f"ch{ch}_env_peak{i}_amp"] = float(pm)
        # 能量分箱 5（可按需调整频段）
        bands = [(0,200), (200,1000), (1000,3000), (3000,6000), (6000,10000)]
        # bands=[(0,50), (50,150), (150,400), (400,800), (800,2000)]
        for k, v in band_energy_features(f_env, m_env, bands, "env").items():
            feats[f"ch{ch}_{k}"] = v
    return feats

# ===================== 提取：.mat → DataFrame =====================
def extract_features_to_df(cfg: Config) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(cfg.data_dir, cfg.file_glob), recursive=True))
    if not paths:
        raise FileNotFoundError(f"No .mat files found under {cfg.data_dir}")

    rows = []
    for p in paths:
        try:
            speed = extract_speed_from_fname(p)
            if speed not in VALID_SPEEDS:
                raise ValueError(f"无法解析转速或不在集合 {sorted(VALID_SPEEDS)}: {p}")
            label_val = extract_label_from_fname(p)

            mat = loadmat(p)
            for win_idx, seg in yield_segments_from_mat(mat, cfg):
                feats = process_one_window(seg, cfg)
                rows.append({
                    "path": p,
                    "speed": int(speed),
                    "win_idx": int(win_idx),
                    "label": int(label_val),
                    **feats
                })
        except Exception as e:
            print(f"[WARN] Skipped {p}: {e}")
            continue

    # if not rows:
    #     raise RuntimeError("No rows produced from .mat files.")
    df = pd.DataFrame(rows)

    # 列顺序：元数据在前
    meta_cols = ["path", "speed", "win_idx", "label"]
    feat_cols = [c for c in df.columns if c not in meta_cols]
    df = df[meta_cols + feat_cols]
    print(f"[Features] rows={len(df)}, cols={len(df.columns)}")

    if cfg.SAVE_FEATURE_CSV:
        os.makedirs(os.path.dirname(cfg.out_csv) or ".", exist_ok=True)
        df.to_csv(cfg.out_csv, index=False)
        print(f"[CSV] Saved raw features to {cfg.out_csv} with shape {df.shape}")
    return df

# ===================== 主流程 =====================
def main(cfg: Config = CFG):
    df = extract_features_to_df(cfg)
    print("特征已保存到 CSV，完成。")
    return df

if __name__ == "__main__":
    main(CFG)
