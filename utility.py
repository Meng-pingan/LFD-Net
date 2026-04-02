import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.io as sio
import h5py

matplotlib.use('Agg')
matplotlib.rc("font", family='Microsoft YaHei')


class HSI:
    """高光谱图像数据类"""

    def __init__(self, data, rows, cols, gt, abundance_gt):
        if data.shape[0] < data.shape[1]:
            data = data.T
        self.bands = np.min(data.shape)
        self.cols = cols
        self.rows = rows
        # 使用Fortran顺序(列优先)匹配Matlab存储
        self.image = np.reshape(data, (self.rows, self.cols, self.bands), order='F')
        self.gt = gt
        self.abundance_gt = abundance_gt

    def array(self):
        """返回 像元*波段 的数据阵列"""
        # 使用Fortran顺序匹配Matlab
        return np.reshape(self.image, (self.rows * self.cols, self.bands), order='F')


def load_HSI(path):
    """加载高光谱数据"""
    try:
        data = sio.loadmat(path)
    except NotImplementedError:
        data = h5py.File(path, 'r')

    numpy_array = np.asarray(data['Y'], dtype=np.float32)
    numpy_array = numpy_array / np.max(numpy_array.flatten())
    n_rows = data['lines'].item()
    n_cols = data['cols'].item()

    gt = np.asarray(data['GT'], dtype=np.float32) if 'GT' in data.keys() else None
    abundance_gt = np.asarray(data['S_GT'], dtype=np.float32) if 'S_GT' in data.keys() else None

    return HSI(numpy_array, n_rows, n_cols, gt, abundance_gt)


class HSIDataset(torch.utils.data.Dataset):
    """高光谱数据集"""

    def __init__(self, img):
        self.img = img.float()

    def __getitem__(self, idx):
        return self.img

    def __len__(self):
        return 1


def numpy_SAD(y_true, y_pred):
    """计算光谱角距离"""
    cos = y_pred.dot(y_true) / (np.linalg.norm(y_true) * np.linalg.norm(y_pred) + 1e-8)
    cos = np.clip(cos, -1.0, 1.0)
    return np.arccos(cos)


def numpy_RMSE(y_true, y_pred):
    """计算RMSE"""
    diff = y_true - y_pred
    mse = np.mean(diff ** 2)
    return np.sqrt(mse)


def reconstruction_SADloss(output, target):
    """重构SAD损失"""
    _, band, h, w = output.shape
    output = torch.reshape(output, (band, h * w))
    target = torch.reshape(target, (band, h * w))
    loss = torch.acos(torch.clamp(torch.cosine_similarity(output, target, dim=0), -1 + 1e-7, 1 - 1e-7))
    return torch.mean(loss)


def volume_maximization_loss(endmember_matrix):
    """体积最大化约束损失
    """
    # 计算 Gram 矩阵
    G = endmember_matrix.T @ endmember_matrix  # [P, P]

    # 添加正则化保证正定（数值稳定性）
    eps = 1e-6
    P = endmember_matrix.shape[1]
    G_reg = G + eps * torch.eye(P, device=G.device)

    sign, logdet = torch.slogdet(G_reg)

    if sign <= 0:
        return torch.tensor(1e6, device=endmember_matrix.device)

    # 返回负的 logdet，这样最小化损失 = 最大化体积
    return -logdet


def abundance_sparsity_loss(abundance):
    """丰度稀疏性约束损失"""
    # 计算每个像素的丰度熵
    entropy = -torch.sum(abundance * torch.log(abundance + 1e-10), dim=1)  # [B, H, W]
    return torch.mean(entropy)


def total_variation_loss(abundance):
    """全变分(TV)损失 - 空间平滑约束"""
    diff_h = abundance[:, :, :, 1:] - abundance[:, :, :, :-1]  # [B, P, H, W-1]
    diff_v = abundance[:, :, 1:, :] - abundance[:, :, :-1, :]  # [B, P, H-1, W]
    # 计算L1范数 (各向异性TV)
    tv_h = torch.abs(diff_h).mean()
    tv_v = torch.abs(diff_v).mean()

    return tv_h + tv_v


def order_endmembers(endmembersGT, endmembers):
    """对齐预测端元和真实端元的顺序"""
    num_endmembers = endmembersGT.shape[0]
    SAD_matrix = np.zeros((num_endmembers, num_endmembers))
    SAD_index = np.zeros(num_endmembers, dtype=int)
    SAD_values = np.zeros(num_endmembers)

    egt = endmembersGT.copy()
    e = endmembers.copy()

    # 归一化
    for i in range(num_endmembers):
        egt[i] = egt[i] / (egt[i].max() + 1e-8)
        e[i] = e[i] / (e[i].max() + 1e-8)

    for i in range(num_endmembers):
        for j in range(num_endmembers):
            SAD_matrix[i, j] = numpy_SAD(e[j], egt[i])
        SAD_index[i] = np.nanargmin(SAD_matrix[i])
        SAD_values[i] = np.nanmin(SAD_matrix[i])
        e[SAD_index[i]] = np.inf

    return SAD_index, SAD_values


def order_abundance(abundanceGT, abundance):
    """对齐预测丰度和真实丰度的顺序"""
    num_endmembers = abundanceGT.shape[2]
    RMSE_matrix = np.zeros((num_endmembers, num_endmembers))
    RMSE_index = np.zeros(num_endmembers, dtype=int)
    RMSE_values = np.zeros(num_endmembers)

    agt = abundanceGT.copy()
    a = abundance.copy()

    for i in range(num_endmembers):
        for j in range(num_endmembers):
            RMSE_matrix[i, j] = numpy_RMSE(a[:, :, j], agt[:, :, i])
        RMSE_index[i] = np.nanargmin(RMSE_matrix[i])
        RMSE_values[i] = np.nanmin(RMSE_matrix[i])
        a[:, :, RMSE_index[i]] = np.inf

    return RMSE_index, RMSE_values


def hyperVCA(M, q):
    """VCA算法提取端元"""
    L, N = M.shape

    rMean = np.mean(M, axis=1, keepdims=True)
    RZeroMean = M - rMean
    U, S, V = np.linalg.svd(RZeroMean @ RZeroMean.T / N)
    Ud = U[:, :q]

    Rd = Ud.T @ RZeroMean
    P_R = np.sum(M ** 2) / N
    P_Rp = np.sum(Rd ** 2) / N + rMean.T @ rMean
    SNR = np.abs(10 * np.log10((P_Rp - (q / L) * P_R) / (P_R - P_Rp + 1e-8)))
    SNRth = 18 + 10 * np.log(q)

    if SNR > SNRth:
        d = q
        U, S, V = np.linalg.svd(M @ M.T / N)
        Ud = U[:, :d]
        Xd = Ud.T @ M
        u = np.mean(Xd, axis=1, keepdims=True)
        Y = Xd / (np.sum(Xd * u, axis=0, keepdims=True) + 1e-8)
    else:
        d = q - 1
        Ud = U[:, :d]
        R_zeroMean = M - rMean
        Xd = Ud.T @ R_zeroMean
        c = np.max([np.linalg.norm(Xd[:, j]) for j in range(N)])
        Y = np.vstack([Xd, c * np.ones((1, N))])

    e_u = np.zeros((q, 1))
    e_u[q - 1, 0] = 1
    A = np.zeros((q, q))
    A[:, 0] = e_u.flatten()
    I = np.eye(q)
    indices = np.zeros(q, dtype=int)

    for i in range(q):
        w = np.random.random((q, 1))
        tmp = (I - A @ np.linalg.pinv(A)) @ w
        f = tmp / (np.linalg.norm(tmp) + 1e-8)
        v = f.T @ Y
        k = np.argmax(np.abs(v))
        A[:, i] = Y[:, k]
        indices[i] = k

    if SNR > SNRth:
        endmembers = Ud @ Xd[:, indices]
    else:
        endmembers = Ud @ Xd[:, indices] + rMean

    return endmembers, indices


def plot_endmembers(endmembers, endmembersGT, save_path, sad_list):
    """绘制端元对比图"""
    num_endmembers = endmembers.shape[0]
    n = (num_endmembers + 1) // 2

    SAD_index, SAD_values = order_endmembers(endmembersGT.copy(), endmembers.copy())

    fig = plt.figure(figsize=(9, 9))
    title = f"mSAD: {SAD_values.mean():.4f} radians"
    plt.suptitle(title, fontsize=15)

    # 归一化
    e = endmembers.copy()
    egt = endmembersGT.copy()
    for i in range(num_endmembers):
        e[i] = e[i] / (e[i].max() + 1e-8)
        egt[i] = egt[i] / (egt[i].max() + 1e-8)

    for i in range(num_endmembers):
        ax = plt.subplot(2, n, i + 1)
        plt.plot(e[SAD_index[i]], 'r', linewidth=1.0, label='Estimated')
        plt.plot(egt[i], 'k', linewidth=1.0, label='GT')
        ax.set_title(f"SAD: {SAD_values[i]:.4f}")
        ax.get_xaxis().set_visible(False)
        if i == 0:
            ax.legend(fontsize=8)
        sad_list.append(SAD_values[i])

    sad_list.append(SAD_values.mean())
    plt.tight_layout()
    plt.savefig(save_path + '.png', dpi=150)
    plt.close()


def plot_abundances(abundances, abundanceGT, save_path, rmse_list, SAD_index=None):
    """绘制丰度对比图"""
    num_endmembers = abundances.shape[2]

    if SAD_index is not None:
        RMSE_index = SAD_index
        RMSE_values = np.zeros(num_endmembers)
        for i in range(num_endmembers):
            RMSE_values[i] = numpy_RMSE(abundances[:, :, RMSE_index[i]], abundanceGT[:, :, i])
    else:
        # 独立排序（不推荐，仅用于向后兼容）
        RMSE_index, RMSE_values = order_abundance(abundanceGT.copy(), abundances.copy())

    fig, axes = plt.subplots(num_endmembers, 2, figsize=(8, 3 * num_endmembers))
    cmap = 'jet'

    for i in range(num_endmembers):
        ax_pred = axes[i, 0] if num_endmembers > 1 else axes[0]
        im1 = ax_pred.imshow(abundances[:, :, RMSE_index[i]], cmap=cmap, vmin=0, vmax=1)
        ax_pred.set_title(f"Estimated (RMSE: {RMSE_values[i]:.4f})")
        ax_pred.axis('off')
        plt.colorbar(im1, ax=ax_pred, fraction=0.046, pad=0.04)

        ax_gt = axes[i, 1] if num_endmembers > 1 else axes[1]
        im2 = ax_gt.imshow(abundanceGT[:, :, i], cmap=cmap, vmin=0, vmax=1)
        ax_gt.set_title("Ground Truth")
        ax_gt.axis('off')
        plt.colorbar(im2, ax=ax_gt, fraction=0.046, pad=0.04)

        rmse_list.append(RMSE_values[i])

    rmse_list.append(RMSE_values.mean())
    plt.tight_layout()
    plt.savefig(save_path + '.png', dpi=150)
    plt.close()


def plot_alpha(alpha_values, save_path):
    """绘制alpha值分布"""
    plt.figure(figsize=(12, 4))
    plt.plot(alpha_values, 'b-', linewidth=1.0)
    plt.xlabel('Band Index')
    plt.ylabel('Alpha Value')
    plt.title(f'Learnable Alpha Distribution (mean={alpha_values.mean():.3f})')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path + '.png', dpi=150)
    plt.close()


def reconstruct(abundance, endmembers):
    """重构高光谱图像"""
    H, W, P = abundance.shape
    abundance_flat = abundance.reshape(H * W, P)  # [N, P]
    reconstructed = abundance_flat @ endmembers.T  # [N, C]
    return reconstructed


def plot_spatial_attention(attention_map, save_path):
    """绘制空间注意力图"""
    plt.figure(figsize=(8, 8))
    im = plt.imshow(attention_map, cmap='hot', vmin=0, vmax=1)
    plt.colorbar(im, fraction=0.046, pad=0.04)
    plt.title(f'Spatial Attention Map\n(mean={attention_map.mean():.3f}, std={attention_map.std():.3f})')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path + '.png', dpi=150)
    plt.close()


def generate_slic_segments(hsi_image, n_segments=100, compactness=10.0):
    """使用SLIC算法生成超像素分割"""
    from skimage.segmentation import slic
    from skimage.util import img_as_float

    if hsi_image.max() > 1.0:
        hsi_image = hsi_image / hsi_image.max()

    if hsi_image.shape[2] > 10:
        # 使用PCA降维到10个波段
        from sklearn.decomposition import PCA
        H, W, C = hsi_image.shape
        hsi_flat = hsi_image.reshape(-1, C)
        pca = PCA(n_components=min(10, C))
        hsi_pca = pca.fit_transform(hsi_flat)
        hsi_reduced = hsi_pca.reshape(H, W, -1)
    else:
        hsi_reduced = hsi_image

    # 运行SLIC
    segments = slic(hsi_reduced, n_segments=n_segments, compactness=compactness,
                    start_label=0, channel_axis=2)

    n_actual = len(np.unique(segments))

    return segments, n_actual


def superpixel_consistency_loss(abundance, superpixel_labels):
    """超像素一致性损失"""
    B, P, H, W = abundance.shape
    device = abundance.device

    # 展平空间维度
    abundance_flat = abundance.view(B, P, -1)  # [B, P, N]
    labels_flat = superpixel_labels.view(-1).long()  # [N]

    # 获取超像素数量
    num_sp = labels_flat.max().item() + 1

    ones = torch.ones(H * W, device=device)
    sp_counts = torch.zeros(num_sp, device=device)
    sp_counts.scatter_add_(0, labels_flat, ones)
    sp_counts = sp_counts.clamp(min=1)  # 避免除零

    total_loss = 0.0

    for b in range(B):
        for p in range(P):
            # 当前丰度通道
            abd = abundance_flat[b, p]  # [N]

            sp_sum = torch.zeros(num_sp, device=device)
            sp_sum.scatter_add_(0, labels_flat, abd)

            sp_mean = sp_sum / sp_counts  # [num_sp]

            pixel_sp_mean = sp_mean[labels_flat]  # [N]

            # 计算每个像素与其超像素均值的差异
            diff = abd - pixel_sp_mean

            sp_var_sum = torch.zeros(num_sp, device=device)
            sp_var_sum.scatter_add_(0, labels_flat, diff ** 2)
            sp_var = sp_var_sum / sp_counts  # 每个超像素的方差

            total_loss += sp_var.mean()

    return total_loss / (B * P)


def plot_superpixel_segments(segments, save_path):
    """绘制超像素分割结果"""
    from skimage.segmentation import mark_boundaries
    H, W = segments.shape
    background = np.ones((H, W, 3)) * 0.5

    marked = mark_boundaries(background, segments, color=(1, 0, 0), mode='thick')

    plt.figure(figsize=(8, 8))
    plt.imshow(marked)
    plt.title(f'SLIC Superpixel Segmentation\n(n_segments={len(np.unique(segments))})')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path + '.png', dpi=150)
    plt.close()
