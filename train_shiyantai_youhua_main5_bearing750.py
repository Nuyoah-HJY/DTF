import os, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import ChebConv
from torch_geometric.loader import DataLoader
from sklearn.metrics import confusion_matrix, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from torch.optim.lr_scheduler import ReduceLROnPlateau
from collections import Counter
import glob


# 固定随机种子
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
set_seed(42)

class Config:
    seq_len = 10 # 每个样本的时间步数
    num_nodes = 3  #通道数
    in_dim = 19 # 输入维度
    top_k = 2 #一般1~3
    cheb_order = 2

    hidden_dim = 100
    num_classes = 8  # 10 分类任务
    dropout_rate = 0.2
    gate_adapt_rate = 0.01
    gate_init_value = 0.5 #gate初始值
    gate_min_value = 0.0
    gate_max_value = 1.0
    gate_freeze_epochs = 5  # 冻结次数
    epochs = 100
    batch_size = 32
    lr = 0.001
    weight_decay = 1e-4
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    use_dynamic = True       # False → 仅物理图; True → 物理+功能图


# ===== 图卷积层（带残差） =====
class ResidualChebConv(nn.Module):
    def __init__(self, in_channels, out_channels, K):
        super().__init__()
        self.conv = ChebConv(in_channels, out_channels, K)
        self.res_fc = nn.Linear(in_channels, out_channels) if in_channels != out_channels else nn.Identity()
        self.bn = nn.BatchNorm1d(out_channels)
    def forward(self, x, edge_index, edge_weight=None):
        res = self.res_fc(x)
        x = self.conv(x, edge_index, edge_weight)
        x = self.bn(x)
        return F.relu(x + res)


# ===== 动态图学习器 =====
class DynamicGraphLearner(nn.Module):
    def __init__(self, feat_dim):
        super().__init__()
        self.edge_predictor = nn.Sequential(
            nn.Linear(feat_dim * 2, 600), nn.ReLU(), nn.Linear(600, 1)
        )
    def forward(self, x):
        if x.dim() == 4:
            B, T, N, Fd = x.size()
            x = x.view(B, T*N, Fd)
        x_norm = F.normalize(x, p=2, dim=-1)
        adj = torch.matmul(x_norm, x_norm.transpose(1, 2))  # (B,M,M)
        eye = torch.eye(adj.size(1), device=adj.device).unsqueeze(0)
        adj = adj * (1 - eye)
        sparse = torch.zeros_like(adj)
        k = min(Config.top_k, adj.size(1))
        for b in range(adj.size(0)):
            vals, idx = torch.topk(adj[b], k=k, dim=1)
            sparse[b].scatter_(1, idx, vals)
        sparse = (sparse + sparse.transpose(1, 2)) / 2
        return sparse, None


# ===== 物理/动态图构建器 =====
class EnhancedGraphBuilder:
    def __init__(self, num_nodes, seq_len):
        self.num_nodes = num_nodes; self.seq_len = seq_len
        self.dynamic_learner = DynamicGraphLearner(Config.in_dim)
    def build_physics_graph(self, num_nodes=None):
        if num_nodes is None: num_nodes = self.num_nodes
        edges = []
        for t in range(self.seq_len):
            s = t * num_nodes
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if i != j: edges.append([s + i, s + j])
                edges.append([s + i, s + i])
        for t in range(1, self.seq_len):
            p = (t - 1) * num_nodes; c = t * num_nodes
            for i in range(num_nodes):
                for j in range(num_nodes):
                    edges.append([p + i, c + j]); edges.append([c + j, p + i])
        return torch.tensor(edges, dtype=torch.long).t().contiguous()
    def build_dynamic_edges(self, x):
        if x.dim() == 2:
            B = x.size(0) // (Config.seq_len * Config.num_nodes)
            x = x.view(B, Config.seq_len, Config.num_nodes, Config.in_dim)
        adj, _ = self.dynamic_learner(x)
        edges = []; num_per_graph = Config.seq_len * Config.num_nodes
        for b in range(x.size(0)):
            rows, cols = torch.where(adj[b] > 0)
            if rows.numel() == 0:
                idx = torch.arange(num_per_graph, device=x.device); rows, cols = idx, idx
            off = b * num_per_graph
            edges.append(torch.stack([rows + off, cols + off], dim=0))
        return torch.cat(edges, dim=1), None


# ===== 多头时空融合模块 =====
class MultiHeadSpatialTemporalFusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.physical_gate = nn.Parameter(torch.tensor(config.gate_init_value, dtype=torch.float32), requires_grad=True)
        self.spatial_attention = None; self.temporal_heads = None; self.transform = None
    def forward(self, x_phy, x_fun):
        B, T, N, Fd = x_phy.size()
        if self.spatial_attention is None:
            self.spatial_attention = nn.Sequential(
                nn.Conv1d(2*Fd, Fd, kernel_size=1), nn.ReLU(),
                nn.Conv1d(Fd, 2*Fd, kernel_size=1), nn.Sigmoid()
            )
            self.temporal_heads = nn.ModuleList([
                nn.Sequential(nn.Linear(2*Fd, Fd), nn.ReLU(), nn.Linear(Fd, 1), nn.Sigmoid())
                for _ in range(T)
            ])
            self.transform = nn.Sequential(nn.Linear(Fd, self.config.hidden_dim // 2), nn.ReLU())
        spatial = torch.cat([x_phy, x_fun], dim=-1).permute(0,1,3,2)   # (B,T,2F,N)
        attn = self.spatial_attention(spatial.reshape(-1, 2*Fd, N)).reshape(B,T,2*Fd,N)
        fused = (spatial * attn).permute(0,1,3,2)                      # (B,T,N,2F)
        phy_list, fun_list = [], []
        for t in range(T):
            tf = fused[:, t]
            alpha_t = self.temporal_heads[t](tf.reshape(B*N, 2*Fd)).view(B, N, 1)
            phy_list.append(alpha_t * x_phy[:, t])
            fun_list.append((1 - alpha_t) * x_fun[:, t])
        g_phy = torch.stack(phy_list, dim=1).mean(dim=2).mean(dim=1)
        g_fun = torch.stack(fun_list, dim=1).mean(dim=2).mean(dim=1)
        gate = torch.clamp(self.physical_gate, self.config.gate_min_value, self.config.gate_max_value)
        out = self.transform(gate * g_phy + (1 - gate) * g_fun)
        return out, gate


# ===== 双分支时空网络 =====
class DualBranchSTN(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.graph_builder = EnhancedGraphBuilder(config.num_nodes, config.seq_len)
        self.temp_embed = nn.Parameter(torch.randn(config.seq_len, 1, config.in_dim))
        self.spat_embed = nn.Parameter(torch.randn(1, config.num_nodes, config.in_dim))
        self.phy_convs = nn.ModuleList([
            ResidualChebConv(config.in_dim, config.hidden_dim, config.cheb_order),
            ResidualChebConv(config.hidden_dim, config.hidden_dim // 2, config.cheb_order),
            ResidualChebConv(config.hidden_dim // 2, config.hidden_dim // 4, config.cheb_order)
        ])
        self.fun_convs = nn.ModuleList([
            ResidualChebConv(config.in_dim, config.hidden_dim, config.cheb_order),
            ResidualChebConv(config.hidden_dim, config.hidden_dim // 2, config.cheb_order),
            ResidualChebConv(config.hidden_dim // 2, config.hidden_dim // 4, config.cheb_order)
        ])
        self.fusion_module = MultiHeadSpatialTemporalFusion(config)
        self.fc = nn.Sequential(
            nn.Linear(config.hidden_dim // 2, config.hidden_dim // 2),
            nn.ReLU(), nn.Dropout(config.dropout_rate),
            nn.Linear(config.hidden_dim // 2, config.num_classes)
        )
    def forward(self, phy_data, fun_data):
        x_phy_nodes = self._process_branch(phy_data, self.phy_convs, phy_data.edge_index)
        fun_input = fun_data.x
        if fun_input.dim() == 2:
            B = fun_input.size(0) // (self.config.seq_len * self.config.num_nodes)
            fun_input = fun_input.view(B, self.config.seq_len, self.config.num_nodes, self.config.in_dim)
        if self.config.use_dynamic:
            fun_edges, _ = self.graph_builder.build_dynamic_edges(fun_input)
            fun_data.edge_index = fun_edges.to(fun_data.x.device)
        else:
            fun_data.edge_index = phy_data.edge_index
        x_fun_nodes = self._process_branch(fun_data, self.fun_convs, fun_data.edge_index)
        total_nodes = x_phy_nodes.size(0)
        B = total_nodes // (self.config.seq_len * self.config.num_nodes)
        x_phy = x_phy_nodes.view(B, self.config.seq_len, self.config.num_nodes, -1)
        x_fun = x_fun_nodes.view(B, self.config.seq_len, self.config.num_nodes, -1)
        fused, gate = self.fusion_module(x_phy, x_fun)
        logits = F.log_softmax(self.fc(fused), dim=1)
        return logits, gate, fused
    def _process_branch(self, data, conv_layers, edge_index):
        x = data.x.view(-1, self.config.seq_len, self.config.num_nodes, self.config.in_dim)
        x = x + self.temp_embed.unsqueeze(0); x = x + self.spat_embed.unsqueeze(0)
        x = x.view(-1, self.config.in_dim)
        for conv in conv_layers: x = conv(x, edge_index)
        return x
    @staticmethod
    def build_with_dummy(config, device):
        model = DualBranchSTN(config).to(device)
        dummy = Data(x=torch.zeros(config.seq_len * config.num_nodes, config.in_dim).to(device),
                     edge_index=torch.zeros(2, 0, dtype=torch.long).to(device),
                     y=torch.tensor([0]).to(device))
        model(dummy, dummy); return model


# ===== 数据集 =====
class FaultDiagnosisDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir):
        self.samples = []; self.graph_builder = EnhancedGraphBuilder(Config.num_nodes, Config.seq_len)
        self._domains = set()
        for fname in os.listdir(root_dir):
            if fname.endswith('.pt'):
                path = os.path.join(root_dir, fname)
                try: data = torch.load(path, map_location="cpu", weights_only=False)
                except TypeError: data = torch.load(path, map_location="cpu")
                phy, fun, y = self._process_data(data)
                self.samples.append((phy, fun, y))
                self._domains.add(int(phy.domain_id.item()))
    def _process_data(self, raw_data):
        num_nodes = Config.seq_len * Config.num_nodes
        x = raw_data.x.view(num_nodes, Config.in_dim)
        y = raw_data.y.long().view(-1)   # 强制 LongTensor
        speed = raw_data.speed if hasattr(raw_data, "speed") else torch.tensor([-1], dtype=torch.long)
        phy = Data(x=x,edge_index=self.graph_builder.build_physics_graph(Config.num_nodes),y=y)
        phy.domain_id = speed.clone()
        fun = Data(x=x.clone(),edge_index=torch.zeros(2, 0, dtype=torch.long),y=y.clone())
        fun.domain_id = speed.clone()
        return phy, fun, y
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx): return self.samples[idx]
    def unique_domains(self): return sorted(list(self._domains))


# ===== Gate 自适应 =====
class GateAdaptationManager:
    def __init__(self, config):
        self.config = config; self.target_gate = config.gate_init_value
        self.last_loss, self.best_loss = float('inf'), float('inf')
    def compute_penalty(self, gate_value, current_loss, batch_size=1):
        if current_loss < self.best_loss: self.best_loss = current_loss
        if current_loss < self.last_loss:
            self.target_gate = 0.9 * self.target_gate + 0.1 * gate_value.item()
        else:
            self.target_gate = max(self.config.gate_min_value,
                                   min(self.config.gate_max_value,
                                       self.target_gate * (1 - self.config.gate_adapt_rate)))
        self.last_loss = current_loss
        return 0.02 * (gate_value - torch.tensor(self.target_gate, device=gate_value.device)) ** 2


# ===== 域自适应辅助 =====
class GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha): ctx.alpha = alpha; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.alpha * g, None
class GradientReversalLayer(nn.Module):
    def __init__(self, alpha=1.0): super().__init__(); self.alpha = alpha
    def forward(self, x): return GradientReversalFunction.apply(x, self.alpha)
class DomainClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_domains=2):
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, num_domains))
    def forward(self, x): return self.fc(x)


# ===== MMD Loss =====
def compute_mmd_loss(src, tgt, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    b = min(src.size(0), tgt.size(0))
    if b == 0: return torch.tensor(0.0, device=src.device)
    src, tgt = src[:b], tgt[:b]
    total = torch.cat([src, tgt], dim=0)
    total0 = total.unsqueeze(0).expand(total.size(0), total.size(0), total.size(1))
    total1 = total.unsqueeze(1).expand(total.size(0), total.size(0), total.size(1))
    L2 = ((total0 - total1) ** 2).sum(2)
    if fix_sigma: bw = fix_sigma
    else: bw = torch.sum(L2.data) / (b ** 2 - b + 1e-6)
    bw = bw / (kernel_mul ** (kernel_num // 2) + 1e-6)
    kernels = 0
    for i in range(kernel_num):
        kernels = kernels + torch.exp(-L2 / (bw * (kernel_mul ** i) + 1e-6))
    XX = kernels[:b,:b]; YY = kernels[b:,b:]; XY = kernels[:b,b:]; YX = kernels[b:,:b]
    return torch.mean(XX + YY - XY - YX)


# ===== 训练入口 =====
def train_model():
    cfg = Config()
    os.makedirs("best_model", exist_ok=True)

    train_set = FaultDiagnosisDataset("./all_graph_data/shiyantai_graph_data/train")
    val_set   = FaultDiagnosisDataset("./all_graph_data/shiyantai_graph_data/test")
    train_loader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_set, batch_size=cfg.batch_size)

    # 类加权
    lbls = [int(y.item()) for _,_,y in train_set]
    cnt = Counter(lbls); total = sum(cnt.values())
    class_weights = torch.tensor([total / max(1, cnt.get(c, 1)) for c in range(cfg.num_classes)],
                                 dtype=torch.float32, device=cfg.device)
    print("Class counts:", cnt, " | weights:", [round(w,3) for w in class_weights.tolist()])

    # 域索引
    train_domains = train_set.unique_domains()
    domain2idx = {d:i for i,d in enumerate(train_domains)}
    num_domains = len(train_domains)
    print("Train RPM domains:", train_domains)
    print("Domain mapping:", domain2idx)

    model = DualBranchSTN.build_with_dummy(cfg, cfg.device)
    domain_clf = DomainClassifier(input_dim=cfg.hidden_dim//2, num_domains=num_domains).to(cfg.device)
    grl = GradientReversalLayer(alpha=0.0)

    optim = torch.optim.AdamW(list(model.parameters()) + list(domain_clf.parameters()),
                              lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = ReduceLROnPlateau(optim, mode='max', patience=8, factor=0.5)
    gate_mgr = GateAdaptationManager(cfg)

    best_acc = 0.0
    for epoch in range(cfg.epochs):
        p = epoch / max(1, (cfg.epochs - 1))
        alpha = float(2.0 / (1.0 + np.exp(-10 * p)) - 1.0)
        grl.alpha = alpha
        lambda_da  = 0.1 * alpha   #0.1/0.05
        lambda_mmd = 0.1 * alpha   #0.1/0.05

        model.train(); domain_clf.train()
        train_loss, epoch_gates = 0.0, []
        model.fusion_module.physical_gate.requires_grad = epoch >= cfg.gate_freeze_epochs

        for phy, fun, labels in train_loader:
            phy, fun, labels = phy.to(cfg.device), fun.to(cfg.device), labels.to(cfg.device)
            labels = labels.view(-1)  # ✅ 保证 labels 1D
            optim.zero_grad()
            outputs, gate, feats = model(phy, fun)
            epoch_gates.append(gate.item())

            base_loss = F.nll_loss(outputs, labels, weight=class_weights)

            dom_raw = phy.domain_id.view(-1).detach().cpu().numpy()
            dom_idx = torch.tensor([domain2idx[int(v)] for v in dom_raw], dtype=torch.long, device=cfg.device)

            da_loss = torch.tensor(0.0, device=cfg.device)
            if feats.size(0) == dom_idx.size(0):
                da_logits = domain_clf(grl(feats))
                da_loss = F.cross_entropy(da_logits, dom_idx)

            mmd_loss = torch.tensor(0.0, device=cfg.device)
            uniq = dom_idx.unique()
            if uniq.numel() >= 2:
                pairs = []
                for i in range(uniq.numel()):
                    for j in range(i+1, uniq.numel()):
                        fi = feats[dom_idx == uniq[i]]; fj = feats[dom_idx == uniq[j]]
                        if fi.size(0) > 0 and fj.size(0) > 0: pairs.append(compute_mmd_loss(fi, fj))
                if pairs: mmd_loss = torch.stack(pairs).mean()

            penalty = torch.tensor(0.0, device=cfg.device)
            if epoch >= cfg.gate_freeze_epochs:
                penalty = gate_mgr.compute_penalty(gate, base_loss.item(), batch_size=labels.size(0))

            loss = base_loss + lambda_da * da_loss + lambda_mmd * mmd_loss + penalty
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()

            train_loss += loss.item() * labels.size(0)

        avg_train = train_loss / len(train_set)
        avg_gate = sum(epoch_gates)/len(epoch_gates) if epoch_gates else cfg.gate_init_value
        print(f"Epoch {epoch}: Train Loss={avg_train:.4f}, Gate={avg_gate:.4f}")

        # 验证
        model.eval(); domain_clf.eval()
        val_loss, correct = 0.0, 0
        with torch.no_grad():
            for phy, fun, labels in val_loader:
                phy, fun, labels = phy.to(cfg.device), fun.to(cfg.device), labels.to(cfg.device)
                labels = labels.view(-1)
                out, _, _ = model(phy, fun)
                val_loss += F.nll_loss(out, labels, reduction='sum').item()
                correct += out.argmax(dim=1).eq(labels).sum().item()
        val_loss /= len(val_set); val_acc = correct / len(val_set)
        sched.step(val_acc)
        print(f"LR: {optim.param_groups[0]['lr']:.6f} | Val Loss={val_loss:.4f} | Val Acc={val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc

            best_dir = "best_model"
            for old_ckpt in glob.glob(os.path.join(best_dir, "*.pth")):
                try:
                    os.remove(old_ckpt)
                except Exception as e:
                    print(f"Warning: 删除旧模型失败 {old_ckpt}: {e}")

            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                        'val_acc': val_acc, 'gate_value': float(avg_gate)},
                       f"best_model/best_model_{best_acc:.4f}.pth")
            print(f"New best model saved with accuracy: {best_acc:.4f}")

    print("Training completed!")

    # ===== 最终评估 =====
    print("\nLoading best model for final evaluation...")
    ckpt = f"best_model/best_model_{best_acc:.4f}.pth"
    model = DualBranchSTN.build_with_dummy(cfg, cfg.device)
    state = torch.load(ckpt, map_location=cfg.device)
    model.load_state_dict(state['model_state_dict']); model.eval()

    final_preds, final_labels, gates = [], [], []
    with torch.no_grad():
        for phy, fun, labels in val_loader:
            phy, fun, labels = phy.to(cfg.device), fun.to(cfg.device), labels.to(cfg.device)
            labels = labels.view(-1)
            out, g, _ = model(phy, fun)
            final_preds.extend(out.argmax(dim=1).cpu().numpy())
            final_labels.extend(labels.cpu().numpy())
            gates.append(g.item())

    acc = 100 * (np.array(final_preds) == np.array(final_labels)).mean()
    prec = precision_score(final_labels, final_preds, average='macro', zero_division=0)
    rec  = recall_score(final_labels, final_preds,  average='macro', zero_division=0)
    print("\nFinal Evaluation Results:")
    print("="*50)
    print(f"Accuracy: {acc:.2f}%"); print(f"Precision (macro): {prec:.4f}"); print(f"Recall (macro): {rec:.4f}")
    if gates:
        g = float(np.mean(gates))
        print(f"\nFinal Physical Branch Contribution: {g:.4f}")
        print(f"Final Functional Branch Contribution: {1-g:.4f}")

    # === 新增：分类报告 ===
    print("\nClassification Report:")
    print(classification_report(final_labels, final_preds, labels=list(range(cfg.num_classes)), digits=4))

    cm = confusion_matrix(final_labels, final_preds, labels=list(range(cfg.num_classes)))
    plt.figure(figsize=(10,8))
    # sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
    #             xticklabels=list(range(cfg.num_classes)), yticklabels=list(range(cfg.num_classes)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greys',
                xticklabels=list(range(cfg.num_classes)), yticklabels=list(range(cfg.num_classes)),annot_kws={'color': 'red'})
    plt.title(f"Confusion Matrix (Test RPM) | Acc: {acc:.2f}%", fontsize=14, pad=20)
    plt.xlabel("Predicted"); plt.ylabel("True")
    plt.tight_layout(); plt.savefig("final_confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show(); print("="*50); print("All visualizations saved!")


if __name__ == "__main__":
    train_model()
