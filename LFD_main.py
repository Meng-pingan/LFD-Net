"""
LFD-Net 训练脚本

"""
import torch
import numpy as np
import scipy.io as sio
import os
import random

from LFD_Model_fast import LFDNet
from utility import (load_HSI, reconstruction_SADloss, hyperVCA,
                     plot_endmembers, plot_abundances, plot_alpha, reconstruct,
                     volume_maximization_loss, abundance_sparsity_loss,
                     plot_spatial_attention,
                     generate_slic_segments, superpixel_consistency_loss,
                     plot_superpixel_segments)

seed = 42
random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

torch.backends.cudnn.deterministic = True  
torch.backends.cudnn.benchmark = False     


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}\n")

# ==================== 数据集配置 ====================
dataset = "sy30"  

dataset_paths = {
    "Samson": "Datasets/Samson.mat",
    "sy30": "Datasets/sy30.mat",
}

# 数据集特定超参数
hyperparams = {
    "Samson": {"epochs": 400, "lr": 0.03, "step_size": 35, "gamma": 0.7, "weight_decay": 2e-4},
    "sy30": {"epochs": 400, "lr": 0.01, "step_size": 35, "gamma": 0.7, "weight_decay": 1e-4},
}

# 数据集特定损失超参数配置
loss_hyperparams = {
    "Samson": {
        "phase1_epochs": 50,
        "lambda_minvol": 0.0008,
        "lambda_sparse": 0.03,
        "lambda_slic": 0.04,
        "slic_n_segments": 150,
        "slic_compactness": 0.1,
        "dropout": 0.2,
        "alpha_init": 0.5,  
        "kernel_size": 7    
    },
    "sy30": {
            "phase1_epochs": 50,
            "lambda_minvol": 0.0008,
            "lambda_sparse": 0.03,
            "lambda_slic": 0.04,
            "slic_n_segments": 200,
            "slic_compactness": 0.1,
            "dropout": 0.2,
            "alpha_init": 0.5,  
            "kernel_size": 7    
    }
}

# 编码器参数
ENCODER_HIDDEN_CHANNELS = 32  # 编码器中间层维度（第一层输出维度）
ENCODER_HIDDEN_DIM = 20  # 编码器输出层维度（第二层输出维度）

# ==================== 加载数据 ====================
print(f"Loading dataset: {dataset}")
hsi = load_HSI(dataset_paths[dataset])
data = hsi.array()  # [N, C]
num_endmembers = hsi.gt.shape[0]
rows, cols = hsi.rows, hsi.cols
num_bands = data.shape[1]

print(f"Data shape: {data.shape}, Bands: {num_bands}, Endmembers: {num_endmembers}")
print(f"Image size: {rows} x {cols}")

# 数据范围检查和鲁棒归一化
print(f"Data Range: Min={data.min():.4f}, Max={data.max():.4f}")
if data.max() > 1.0:
    print("Warning: Data > 1.0, performing robust normalization...")
    robust_max = np.percentile(data, 99.9)
    data = np.clip(data / robust_max, 0, 1)
    print(f"After normalization: Min={data.min():.4f}, Max={data.max():.4f}")

# 获取超参数
params = hyperparams[dataset]
EPOCHS = params["epochs"]
lr = params["lr"]
step_size = params["step_size"]
gamma = params["gamma"]
weight_decay = params["weight_decay"]

# 获取损失超参数
loss_params = loss_hyperparams[dataset]
PHASE1_EPOCHS = loss_params["phase1_epochs"]
LAMBDA_MINVOL = loss_params["lambda_minvol"]
LAMBDA_SPARSE = loss_params["lambda_sparse"]
LAMBDA_SLIC = loss_params["lambda_slic"]
SLIC_N_SEGMENTS = loss_params["slic_n_segments"]
SLIC_COMPACTNESS = loss_params["slic_compactness"]
dropout = loss_params["dropout"]
alpha_init = loss_params["alpha_init"]  # 分数阶微分α初始值

# ==================== 创建输出目录 ====================
output_dir = f"Results/LFD-Net/{dataset}"
os.makedirs(f"{output_dir}/mat", exist_ok=True)
os.makedirs(f"{output_dir}/endmember", exist_ok=True)
os.makedirs(f"{output_dir}/abundance", exist_ok=True)
os.makedirs(f"{output_dir}/alpha", exist_ok=True)
os.makedirs(f"{output_dir}/attention", exist_ok=True)
os.makedirs(f"{output_dir}/superpixel", exist_ok=True)
random.seed(seed)
# ==================== 生成SLIC超像素分割 ====================
print("\nGenerating SLIC superpixel segmentation...")
hsi_image = data.reshape(rows, cols, num_bands, order='F')
superpixel_labels, n_superpixels = generate_slic_segments(
    hsi_image, n_segments=SLIC_N_SEGMENTS, compactness=SLIC_COMPACTNESS
)
print(f"  Generated {n_superpixels} superpixels")
print(f"  Superpixel labels shape: {superpixel_labels.shape}")

superpixel_labels_tensor = torch.from_numpy(superpixel_labels).long().to(device)
plot_superpixel_segments(superpixel_labels, f"{output_dir}/superpixel/{dataset}_slic")

# ==================== 准备数据 ====================
original_HSI = torch.from_numpy(data.T.reshape(num_bands, rows, cols, order='F')).float()
original_HSI = original_HSI.unsqueeze(0).to(device)

# Ground Truth
abundance_GT = torch.from_numpy(hsi.abundance_gt)
endmember_GT = hsi.gt

# ==================== VCA初始化端元 ====================
print("\nRunning VCA for endmember initialization...")
print("Using fixed random seed for reproducibility (seed={})".format(seed))
from utility import order_endmembers

np.random.seed(seed)
vca_endmembers_trial, _ = hyperVCA(data.T, num_endmembers)

# 计算与GT的mSAD用于监控
vca_np = vca_endmembers_trial.T
_, vca_SAD_values = order_endmembers(endmember_GT.copy(), vca_np.copy())
vca_mSAD = vca_SAD_values.mean()
print(f"VCA initialization mSAD: {vca_mSAD:.4f} (reference only)")

vca_endmembers = torch.from_numpy(vca_endmembers_trial).float()

# ==================== 创建模型 ====================
print("\nCreating LFD-Net model...")
model = LFDNet(
    num_bands, num_endmembers,
    dropout=dropout,
    kernel_size=7,
    hidden_channels=ENCODER_HIDDEN_CHANNELS,
    hidden_dim=ENCODER_HIDDEN_DIM,
    alpha_init=alpha_init  
).to(device)

with torch.no_grad():
    model.decoder.weight.data = vca_endmembers.unsqueeze(2).unsqueeze(3).to(device)

def init_weights(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.zero_()

model.encoder.apply(init_weights)
model.spatial_attention.apply(init_weights)

optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

# ==================== 训练 ====================
print(f"Starting training...")

sad_results = []
rmse_results = []

for epoch in range(EPOCHS):
    model.train()
    random.seed(seed)
    abundance, reconstruction, spatial_weights, _ = model(original_HSI)
    loss_recon = reconstruction_SADloss(reconstruction, original_HSI)

    if epoch < PHASE1_EPOCHS:
        total_loss = loss_recon
        loss_volmax = torch.tensor(0.0)
        loss_sparse = torch.tensor(0.0)
        loss_slic = torch.tensor(0.0)
    else:
        endmembers = model.decoder.weight.squeeze()
        loss_volmax = volume_maximization_loss(endmembers)

        if LAMBDA_SPARSE > 0:
            loss_sparse = abundance_sparsity_loss(abundance)
        else:
            loss_sparse = torch.tensor(0.0)

        # 超像素一致性损失
        if LAMBDA_SLIC > 0:
            loss_slic = superpixel_consistency_loss(abundance, superpixel_labels_tensor)
        else:
            loss_slic = torch.tensor(0.0)

        total_loss = (loss_recon + LAMBDA_MINVOL * loss_volmax +
                      LAMBDA_SPARSE * loss_sparse + LAMBDA_SLIC * loss_slic)

    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    scheduler.step()

    if (epoch + 1) % 50 == 0 or epoch == 0:
        phase = "Phase1" if epoch < PHASE1_EPOCHS else "Phase2"
        print(f"Epoch [{epoch + 1}/{EPOCHS}] ({phase})")
        print(f"  Loss: {total_loss.item():.6f} ")


# ==================== 评估 ====================
model.eval()
with torch.no_grad():
    abundance, reconstruction, spatial_weights, _ = model(original_HSI)

    abundance_np = abundance.squeeze(0).cpu().numpy()
    abundance_np = abundance_np.transpose(1, 2, 0)

    endmembers_np = model.get_endmembers().cpu().numpy()
    endmembers_np = endmembers_np.T

    alpha_np = model.get_alpha().cpu().numpy()
    scale_np = model.get_scale().cpu().numpy()
    attention_np = spatial_weights.squeeze().cpu().numpy()

    E_final = model.decoder.weight.squeeze()
    final_volmax_loss = volume_maximization_loss(E_final).item()

abundance_GT_np = abundance_GT.numpy()
endmember_GT_np = endmember_GT

y_hat = reconstruct(abundance_np, endmembers_np.T)
RE = np.sqrt(np.mean((y_hat - data) ** 2))

# ==================== 保存结果 ====================

sio.savemat(f"{output_dir}/mat/LFD-Net_result.mat", {
    'A': abundance_np,
    'E': endmembers_np.T,
    'alpha': alpha_np,
    'scale': scale_np,
    'attention': attention_np,
    'superpixel_labels': superpixel_labels,
    'RE': RE,
    'volmax_loss': final_volmax_loss
})

SAD_index, SAD_values = order_endmembers(endmember_GT_np.copy(), endmembers_np.copy())
plot_endmembers(endmembers_np, endmember_GT_np, f"{output_dir}/endmember/{dataset}_endmembers", sad_results)
plot_abundances(abundance_np, abundance_GT_np, f"{output_dir}/abundance/{dataset}_abundance", rmse_results, SAD_index=SAD_index)
plot_spatial_attention(attention_np, f"{output_dir}/attention/{dataset}_attention")
plot_alpha(alpha_np, f"{output_dir}/alpha/{dataset}_alpha")








