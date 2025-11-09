import os
from typing import List, Tuple, Dict
import re
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
import glob
from collections import Counter

# ===== 固定留出的转速 =====
TEST_SPEED = 750  # 可选：750 / 850 / 960 / 1224

# ===== 超参 =====
SEQ_LEN = 10    # 窗口长度
NUM_NODES = 3
IN_DIM = 19     # 每个节点的特征维度

OUT_TRAIN = "./all_graph_data/shiyantai_graph_data/train"
OUT_TEST  = "./all_graph_data/shiyantai_graph_data/test"


# 支持的转速集合（用于校验）
VALID_SPEEDS = {750, 850, 960, 1224}

# ===== 特征列（每通道19维） =====
def feature_columns(ch:int):
    cols = []
    # ---- 时域 6 ----
    cols += [f"ch{ch}_rms", f"ch{ch}_kurt", f"ch{ch}_skew",
             f"ch{ch}_crest_factor", f"ch{ch}_shape_factor", f"ch{ch}_impulse_factor"]
    # ---- 包络主峰 (3×2=6) ----
    for i in range(1,4):
        cols += [f"ch{ch}_env_peak{i}_freq", f"ch{ch}_env_peak{i}_amp"]
    # ---- 能量分箱 (5) ----
    for i in range(5):
        cols += [f"ch{ch}_env_band{i}_energy"]
    # ---- 包络选带上下界 (2) ----
    cols += [f"ch{ch}_env_band_lo", f"ch{ch}_env_band_hi"]
    assert len(cols) == IN_DIM, f"列数不符: {len(cols)} != {IN_DIM}"
    return cols

# ===== Helpers =====
def _to_tensor_row(row: pd.Series, ch: int) -> torch.Tensor:
    cols = feature_columns(ch)
    vals = []
    for c in cols:
        v = row[c] if c in row and pd.notna(row[c]) and pd.api.types.is_number(row[c]) else 0.0
        try:
            vals.append(float(v))
        except Exception:
            vals.append(0.0)
    return torch.tensor(vals, dtype=torch.float32)

def rows_to_sample(rows: List[pd.Series]) -> Data:
    # 拼接节点特征：时间 × 通道
    x_list = []
    for t in range(SEQ_LEN):
        r = rows[t]
        for ch in range(NUM_NODES):
            x_list.append(_to_tensor_row(r, ch))
    X = torch.stack(x_list, dim=0)  # (SEQ_LEN*NUM_NODES, IN_DIM)

    # 标签
    y_val = rows[0]["label"] if "label" in rows[0] else np.nan
    y = torch.tensor([int(y_val)], dtype=torch.long)

    # 速度（来自 CSV 的 speed 列）
    spd = rows[0]["speed"] if "speed" in rows[0] else np.nan
    speed = int(spd) if pd.notna(spd) else -1

    data = Data(x=X, y=y)
    data.load = torch.tensor([-1], dtype=torch.long)
    data.speed = torch.tensor([speed], dtype=torch.long)
    return data

# ===== 正常滑窗取样 =====
def build_sequences(df: pd.DataFrame) -> Tuple[List[Data], Dict[str,int]]:
    """
    对每个 (path) 分组，按 win_idx 排序，步长=1 的滑窗取样。
    注意：这里不再依赖 domain_id；speed 用于后续划分。
    """
    if "win_idx" not in df.columns:
        df["win_idx"] = 0

    seqs: List[Data] = []
    diag = {"path_groups": 0, "usable_paths": 0, "samples": 0}
    df1 = df.sort_values(["path", "win_idx"]).reset_index(drop=True)

    for path, g in df1.groupby(["path"], sort=False):
        diag["path_groups"] += 1
        g = g.sort_values("win_idx")

        if len(g) < SEQ_LEN:
            continue

        num_samples = len(g) - SEQ_LEN + 1
        for start in range(0, num_samples):
            seg = g.iloc[start:start + SEQ_LEN]
            rows = [seg.iloc[t] for t in range(SEQ_LEN)]
            seqs.append(rows_to_sample(rows))
            diag["samples"] += 1

        diag["usable_paths"] += 1

    print(f"[Sequences] built with sliding windows: {diag['samples']} (usable_paths={diag['usable_paths']})")
    return seqs, diag

# ===== 留一转速（LOSO）划分 =====
def split_leave_one_speed(samples: List[Data], test_speed: int):
    train, test = [], []
    for d in samples:
        spd = int(d.speed.item())
        (test if spd == test_speed else train).append(d)
    if not test:
        raise RuntimeError(f"No samples found for TEST_SPEED={test_speed}.")
    return train, test

# ===== RobustScale（按训练集统计） =====
def robust_scale_inplace(train: List[Data], test: List[Data]):
    def stack_X(ds):
        X = [d.x.numpy() for d in ds]
        return np.concatenate([x.reshape(-1, IN_DIM) for x in X], axis=0)
    Xtr = stack_X(train)
    med = np.median(Xtr, axis=0)
    q75 = np.percentile(Xtr, 75, axis=0); q25 = np.percentile(Xtr, 25, axis=0)
    iqr = np.where((q75 - q25) == 0, 1.0, (q75 - q25))
    def apply(ds):
        for d in ds:
            x = d.x.numpy()
            x = (x - med) / iqr
            d.x = torch.tensor(x, dtype=torch.float32)
    apply(train); apply(test)

# ===== 保存数据集 =====
def save_dataset(train: List[Data], test: List[Data]):
    for out_dir in [OUT_TRAIN, OUT_TEST]:
        if os.path.exists(out_dir):
            files = glob.glob(os.path.join(out_dir, "*.pt"))
            for f in files:
                try:
                    os.remove(f)
                except Exception as e:
                    print(f"Warning: 删除文件失败 {f}: {e}")
        else:
            os.makedirs(out_dir, exist_ok=True)
    for i, d in enumerate(train):
        torch.save(d, os.path.join(OUT_TRAIN, f"sample_train_{i:06d}.pt"))
    for i, d in enumerate(test):
        torch.save(d, os.path.join(OUT_TEST,  f"sample_test_{i:06d}.pt"))

# ===== 主流程 =====
def main(csv_path: str, test_speed: int = TEST_SPEED):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"{csv_path} not found.")
    df = pd.read_csv(csv_path)

    # 用 speed 替代 domain_id
    for c in ["path", "speed", "label"]:
        if c not in df.columns:
            raise ValueError(f"Column '{c}' missing in {csv_path}")

    # 基本校验：CSV 中的 speed 是否包含指定测试转速
    speeds = sorted(s for s in df["speed"].dropna().unique().tolist() if int(s) in VALID_SPEEDS)
    print(f"Available speeds in CSV: {speeds}")
    if test_speed not in speeds:
        raise RuntimeError(f"TEST_SPEED={test_speed} not found in CSV. Available: {speeds}")

    samples, _ = build_sequences(df)
    if not samples:
        raise RuntimeError("No sequences could be built.")

    train, test = split_leave_one_speed(samples, test_speed=test_speed)
    robust_scale_inplace(train, test)
    save_dataset(train, test)

    print(f"LOSO (Leave-One-Speed-Out) with TEST_SPEED={test_speed}. "
          f"Train={len(train)}, Test={len(test)}")
    print("Train label distribution:", Counter(int(d.y.item()) for d in train))
    print(" Test label distribution:", Counter(int(d.y.item()) for d in test))

if __name__ == "__main__":
    # CSV 包含 path/speed/win_idx/label/特征列
    main(r"./dataset_shiyantai/gear/L8way_0.2_20_2048/features_L8way_0.2_20_2048.csv", TEST_SPEED)
