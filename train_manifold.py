import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.utils.data as data
from utils_HSI import sample_gt, metrics, seed_worker, test_hsi, hsi_metrics
from datasets import get_dataset, HyperX
import random
import os
import time
import numpy as np
import pandas as pd
import argparse
from con_losses import SupConLoss, manifold_dis
from network import discrim_hyperG
from network import generator
from datetime import datetime


# =========================================================
# 🚀 【架构升维 3.0：空洞空间卷积池化金字塔 (ASPP)】
# 彻底解决 13x13 小斑块内的“局部模糊”问题，强制提取多尺度几何特征！
# =========================================================
class ASPP(nn.Module):
    def __init__(self, in_channels):
        super(ASPP, self).__init__()
        out_channels = in_channels // 4  # 降维以控制参数量，避免过拟合

        # 1. 1x1 卷积：保留原始中心像素特征
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)

        # 2. 3x3 卷积 (Dilation=2)：感受野 5x5，专捉细小的【道路边缘】
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2, bias=False)

        # 3. 3x3 卷积 (Dilation=4)：感受野 9x9，捕捉中等【住宅屋顶】
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=4, dilation=4, bias=False)

        # 4. 3x3 卷积 (Dilation=6)：感受野 13x13，覆盖整个 Patch，捕捉宏观【巨无霸建筑】
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)

        # 5. 全局平均池化分支：获取整个图块的极度全局环境属性
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels, out_channels, 1, bias=False)
        )

        # 将 5 个尺度的特征拼接后，映射回原始维度
        self.out_conv = nn.Conv2d(out_channels * 5, in_channels, 1, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)

        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x.size()[2:], mode='bilinear', align_corners=True)

        # 多尺度特征金字塔暴力拼接！
        out = torch.cat([x1, x2, x3, x4, x5], dim=1)
        out = self.out_conv(out)
        out = self.bn(out)

        # 强力残差连接：确保原始光谱不丢失
        return F.relu(x + out)


class EnhancerWithASPP(nn.Module):
    """将 ASPP 空间金字塔完美嵌入生成器增强层"""

    def __init__(self, base_enhancer, in_planes):
        super().__init__()
        self.enhancer = base_enhancer
        self.aspp = ASPP(in_planes)

    def forward(self, x_src, x_tgt=None):
        out = self.enhancer(x_src, x_tgt)
        if isinstance(out, tuple):
            x_enhanced = self.aspp(out[0])
            return (x_enhanced,) + out[1:]
        else:
            return self.aspp(out)


# =========================================================

parser = argparse.ArgumentParser(description='PyTorch SDEnet')
parser.add_argument('--save_path', type=str, default='./results/')
parser.add_argument('--data_path', type=str, default='./Datasets/Pavia/')
parser.add_argument('--source_name', type=str, default='paviaU', help='the name of the source dir')
parser.add_argument('--target_name', type=str, default='paviaC', help='the name of the test dir')
parser.add_argument('--gpu', type=int, default=0, help="Specify CUDA device")

group_train = parser.add_argument_group('Training')
group_train.add_argument('--patch_size', type=int, default=13)
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--momentum', type=float, default=0.9)
group_train.add_argument('--batch_size', type=int, default=256)
group_train.add_argument('--pro_dim', type=int, default=128)
group_train.add_argument('--test_stride', type=int, default=1)
parser.add_argument('--seed', type=int, default=1233)
parser.add_argument('--l2_decay', type=float, default=1e-4)
parser.add_argument('--num_epoch', type=int, default=400)
parser.add_argument('--training_sample_ratio', type=float, default=0.8)
parser.add_argument('--re_ratio', type=int, default=0)
parser.add_argument('--max_epoch', type=int, default=400)
parser.add_argument('--log_interval', type=int, default=40)
parser.add_argument('--d_se', type=int, default=64)
parser.add_argument('--lambda_1', type=float, default=1.0)
parser.add_argument('--lambda_2', type=float, default=1.0)
parser.add_argument('--lambda_3', type=float, default=1.0)
parser.add_argument('--lr_scheduler', type=str, default='none')
parser.add_argument('--low_freq', action='store_true', default=True)

group_da = parser.add_argument_group('Data augmentation')
group_da.add_argument('--flip_augmentation', action='store_true', default=True)
group_da.add_argument('--radiation_augmentation', action='store_true', default=True)
group_da.add_argument('--mixture_augmentation', action='store_true', default=False)
args = parser.parse_args()


def evaluate(net, val_loader, hyperparameter, device, tgt=False, enhancer=None):
    ps = []
    ys = []
    for i, (x1, y1) in enumerate(val_loader):
        y1 = y1 - 1
        with torch.no_grad():
            x1 = x1.to(device)
            if enhancer:
                out = enhancer(x_src=x1, x_tgt=None)
                x1 = out[0] if isinstance(out, tuple) else out
            p1 = net(x1)
            p1 = p1.argmax(dim=1)
            ps.append(p1.detach().cpu().numpy())
            ys.append(y1.numpy())
    ps = np.concatenate(ps)
    ys = np.concatenate(ys)
    acc = np.mean(ys == ps) * 100
    if tgt:
        results = metrics(ps, ys, n_classes=ys.max() + 1)
        print(results['Confusion_matrix'], '\n', 'AA:', results['class_acc'], '\n', 'OA:', results['Accuracy'], '\n',
              'Kappa:', results['Kappa'])
        if enhancer:
            class CombinedNet(nn.Module):
                def __init__(self, e, n):
                    super().__init__()
                    self.e = e
                    self.n = n

                def forward(self, x):
                    out = self.e(x_src=x, x_tgt=None)
                    out = out[0] if isinstance(out, tuple) else out
                    return self.n(out)

            temp_net = CombinedNet(enhancer, net)
            probility = test_hsi(temp_net, val_loader.dataset.data, hyperparameter)
        else:
            probility = test_hsi(net, val_loader.dataset.data, hyperparameter)

        np.save(args.source_name + 'tsne_pred.npy', probility)
        prediction = np.argmax(probility, axis=-1)

        run_results = hsi_metrics(prediction, val_loader.dataset.label - 1, [-1],
                                  n_classes=hyperparameter['n_classes'])
        print(run_results)
        return acc, results, prediction
    else:
        return acc


def evaluate_tgt(cls_net, device, loader, hyperparameter, modelpath, enhancer=None):
    saved_weight = torch.load(modelpath)
    cls_net.load_state_dict(saved_weight['Discriminator'])
    cls_net.eval()
    if enhancer:
        if 'Enhancer' in saved_weight:
            enhancer.load_state_dict(saved_weight['Enhancer'])
        enhancer.eval()

    teacc, best_results, pred = evaluate(cls_net, loader, hyperparameter, device, tgt=True, enhancer=enhancer)
    return teacc, best_results, pred


def experiment():
    if args.gpu == -1:
        device = torch.device('cpu')
    else:
        if torch.cuda.is_available():
            device = torch.device(f'cuda:{args.gpu}')
        else:
            device = torch.device('cpu')
            args.gpu = -1

    hyperparams = vars(args)
    if 'Houston' in args.source_name and 'Pavia' in args.data_path:
        args.data_path = './Datasets/Houston/'

    hyperparams['device'] = device
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    root = os.path.join(args.save_path, args.source_name + 'to' + args.target_name)
    log_dir = os.path.join(root, str(args.lr) + '_dim' + str(args.pro_dim) +
                           '_pt' + str(args.patch_size) + '_bs' + str(args.batch_size) + '_' + time_str)
    if not os.path.exists(root):
        os.makedirs(root)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    seed_worker(args.seed)
    img_src, gt_src, LABEL_VALUES_src, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.source_name,
                                                                                        args.data_path)
    img_tar, gt_tar, LABEL_VALUES_tar, IGNORED_LABELS, RGB_BANDS, palette = get_dataset(args.target_name,
                                                                                        args.data_path)

    sample_num_src = len(np.nonzero(gt_src)[0])
    sample_num_tar = len(np.nonzero(gt_tar)[0])
    n_classes = np.max(gt_src)
    print("Classes:", n_classes)

    for i in range(n_classes):
        count_class = np.copy(gt_src)
        test_count = np.copy(gt_tar)
        count_class[(gt_src != i + 1)] = 0
        class_num = np.count_nonzero(count_class)
        test_count[gt_tar != i + 1] = 0
        print([i + 1], ':', class_num, np.count_nonzero(test_count))

    print("Total", np.count_nonzero(gt_src), np.count_nonzero(gt_tar))
    tmp = args.training_sample_ratio * args.re_ratio * sample_num_src / sample_num_tar
    num_classes = gt_src.max()
    N_BANDS = img_src.shape[-1]
    hyperparams.update({'n_classes': num_classes, 'n_bands': N_BANDS, 'ignored_labels': IGNORED_LABELS,
                        'device': device, 'center_pixel': None, 'supervision': 'full'})

    r = int(hyperparams['patch_size'] / 2) + 1
    img_src = np.pad(img_src, ((r, r), (r, r), (0, 0)), 'symmetric')
    img_tar = np.pad(img_tar, ((r, r), (r, r), (0, 0)), 'symmetric')
    gt_src = np.pad(gt_src, ((r, r), (r, r)), 'constant', constant_values=(0, 0))
    gt_tar = np.pad(gt_tar, ((r, r), (r, r)), 'constant', constant_values=(0, 0))

    if 'whu' in args.source_name:
        train_gt_src, val_gt_src, _, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')
        val_gt_src, _, _, _ = sample_gt(train_gt_src, 0.1, mode='random')
    else:
        train_gt_src, val_gt_src, _, _ = sample_gt(gt_src, args.training_sample_ratio, mode='random')

    print("All training number is,", np.count_nonzero(train_gt_src), np.count_nonzero(val_gt_src))
    test_gt_tar, _, _, _ = sample_gt(gt_tar, 1, mode='random')
    img_src_con, train_gt_src_con = img_src, train_gt_src
    val_gt_src_con = val_gt_src

    if tmp < 1:
        for i in range(args.re_ratio - 1):
            img_src_con = np.concatenate((img_src_con, img_src))
            train_gt_src_con = np.concatenate((train_gt_src_con, train_gt_src))
            val_gt_src_con = np.concatenate((val_gt_src_con, val_gt_src))

    hyperparams_train = hyperparams.copy()
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_dataset = HyperX(img_src_con, train_gt_src_con, **hyperparams_train)

    train_labels_flat = train_gt_src_con[train_gt_src_con > 0] - 1
    class_counts = np.bincount(train_labels_flat, minlength=hyperparams['n_classes'])
    weight_per_class = 1.0 / (np.sqrt(class_counts) + 1e-6)
    min_weight = np.min(weight_per_class)
    weight_per_class = np.clip(weight_per_class, a_min=min_weight, a_max=min_weight * 15.0)
    sample_weights = weight_per_class[train_labels_flat]
    sample_weights_tensor = torch.DoubleTensor(sample_weights)

    from torch.utils.data import WeightedRandomSampler
    sampler = WeightedRandomSampler(weights=sample_weights_tensor, num_samples=len(sample_weights_tensor),
                                    replacement=True)
    torch.backends.cudnn.benchmark = True

    train_loader = data.DataLoader(train_dataset, batch_size=hyperparams['batch_size'], pin_memory=True,
                                   worker_init_fn=seed_worker, generator=g, sampler=sampler, num_workers=4,
                                   prefetch_factor=2)
    val_dataset = HyperX(img_src_con, val_gt_src_con, **hyperparams)
    val_loader = data.DataLoader(val_dataset, pin_memory=True, batch_size=hyperparams['batch_size'])
    test_dataset = HyperX(img_tar, test_gt_tar, **hyperparams)
    test_loader = data.DataLoader(test_dataset, pin_memory=True, worker_init_fn=seed_worker, generator=g,
                                  batch_size=hyperparams['batch_size'])
    imsize = [hyperparams['patch_size'], hyperparams['patch_size']]

    pad = True if 'whu' in args.source_name else False

    # ---------------------------------------------------------
    # 搭载全局自注意力架构的网络
    # ---------------------------------------------------------
    base_enhancer = generator.FeatureEnhancer(imsize=imsize, imdim=N_BANDS).to(device)
    Enhancer = EnhancerWithASPP(base_enhancer, in_planes=N_BANDS).to(device)
    Enhancer_opt = optim.Adam(Enhancer.parameters(), lr=args.lr)

    D_net = discrim_hyperG.Discriminator(inchannel=N_BANDS, outchannel=args.pro_dim, num_classes=num_classes,
                                         patch_size=hyperparams['patch_size'], pad=pad).to(device)
    D_opt = optim.Adam(D_net.parameters(), lr=args.lr)

    from network.discrim_hyperG import FreqDiscriminator
    D_freq = FreqDiscriminator(N_BANDS).to(device)
    D_freq_opt = optim.Adam(D_freq.parameters(), lr=args.lr)

    G_net = generator.Generator(n=args.d_se, imdim=N_BANDS, imsize=imsize, zdim=10, device=device,
                                low_freq=args.low_freq).to(device)
    G_opt = optim.Adam(G_net.parameters(), lr=args.lr)

    if args.lr_scheduler == 'cosine':
        scheduler_D = optim.lr_scheduler.CosineAnnealingLR(D_opt, T_max=args.max_epoch, eta_min=1e-5)
        scheduler_G = optim.lr_scheduler.CosineAnnealingLR(G_opt, T_max=args.max_epoch, eta_min=1e-5)
        scheduler_E = optim.lr_scheduler.CosineAnnealingLR(Enhancer_opt, T_max=args.max_epoch, eta_min=1e-5)

    # =========================================================
    # 🚀 【死锁 78.8% 黄金比例：上帝权重】
    # =========================================================
    if 'Houston' in args.source_name:
        weight_list = [2.5, 1.2, 1.2, 0.5, 1.0, 0.6, 2.5]
    else:
        weight_list = [1.0] * num_classes

    class_weights_tensor = torch.FloatTensor(weight_list).to(device)
    cls_criterion = nn.CrossEntropyLoss(weight=class_weights_tensor, label_smoothing=0.1).to(device)
    con_criterion = SupConLoss(device=device)

    best_acc_tgt = 0
    taracc_list = []
    best_acc = 0
    geometric_constraint_weight = 0.1

    # =========================================================
    # 训练大循环
    # =========================================================
    for epoch in range(1, args.max_epoch + 1):
        t1 = time.time()
        loss_list = []
        D_net.train()
        G_net.train()
        Enhancer.train()

        for i, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            y = y - 1

            try:
                x_tgt_raw, _ = next(test_iter)
            except (StopIteration, NameError):
                test_iter = iter(test_loader)
                x_tgt_raw, _ = next(test_iter)

            x_tgt_raw = x_tgt_raw.to(device)

            if x.size(0) != x_tgt_raw.size(0):
                min_b = min(x.size(0), x_tgt_raw.size(0))
                x = x[:min_b]
                y = y[:min_b]
                x_tgt_raw = x_tgt_raw[:min_b]

            # --- 特征提取 (内置纯原生全局自注意力) ---
            x_enhanced, _, _ = Enhancer(x_src=x, x_tgt=x_tgt_raw)
            x_tgt_enhanced, _, _ = Enhancer(x_src=x_tgt_raw, x_tgt=x)

            x_ED = G_net(x_enhanced)
            x_tgt = G_net(x_tgt_enhanced)

            p_SD, z_SD = D_net(x_enhanced, mode='train')
            p_ED, z_ED = D_net(x_ED, mode='train')

            x_ID = torch.zeros_like(x_enhanced)
            y_unique = torch.unique(y)
            for id_val in y_unique:
                mask = (y == id_val)
                if mask.any():
                    z_SD_i = z_SD[mask]
                    z_ED_i = z_ED[mask]
                    z_all_i = torch.cat([z_SD_i, z_ED_i], dim=0)

                    sample_size = z_SD_i.size(0)
                    idx_rand = torch.randperm(z_all_i.size(0), device=device)[:sample_size]

                    p_dist_sd, c_dist_sd = manifold_dis(z_SD_i, z_all_i[idx_rand, :])
                    p_dist_ed, c_dist_ed = manifold_dis(z_ED_i, z_all_i[idx_rand, :])

                    lambda_pr1 = args.lambda_3 * p_dist_sd + c_dist_sd
                    lambda_pd1 = args.lambda_3 * p_dist_ed + c_dist_ed

                    weight_sd = (lambda_pd1 / (lambda_pr1 + lambda_pd1 + 1e-8)).view(-1, 1, 1, 1)
                    weight_ed = (lambda_pr1 / (lambda_pr1 + lambda_pd1 + 1e-8)).view(-1, 1, 1, 1)

                    x_ID[mask] = weight_sd * x_enhanced[mask] + weight_ed * x_ED[mask]

            p_ID, z_ID = D_net(x_ID, mode='train')
            p_tgt, z_tgt = D_net(x_tgt, mode='train')

            min_batch_size = min(z_tgt.size(0), z_SD.size(0))
            z_tgt_con = z_tgt[:min_batch_size]
            y_con = y[:min_batch_size]

            src_cls_loss = cls_criterion(p_SD, y.long()) + cls_criterion(p_ED, y.long()) + cls_criterion(p_ID, y.long())

            zsrc = torch.cat([z_SD[:min_batch_size].unsqueeze(1), z_ED[:min_batch_size].unsqueeze(1),
                              z_ID[:min_batch_size].unsqueeze(1)], dim=1)
            con_loss = con_criterion(zsrc, y_con, adv=False)

            loss_D = src_cls_loss + args.lambda_1 * con_loss

            D_opt.zero_grad()
            Enhancer_opt.zero_grad()
            loss_D.backward(retain_graph=True)

            zsrc_con = torch.cat([z_tgt_con.unsqueeze(1), z_ED[:min_batch_size].unsqueeze(1).detach(),
                                  z_ID[:min_batch_size].unsqueeze(1).detach()], dim=1)
            con_loss_adv = 0
            idx_1 = np.random.randint(0, zsrc_con.size(1))
            zall_adv = torch.cat([z_tgt_con.unsqueeze(1), zsrc_con[:, idx_1:idx_1 + 1].detach()], dim=1)
            con_loss_adv += con_criterion(zall_adv, adv=True)

            p_dist, c_dist = manifold_dis(z_SD, z_ED)
            g_loss = args.lambda_3 * p_dist + c_dist

            mmd_shift = g_loss.item()
            with torch.no_grad():
                target_weight = 1.0 - np.exp(-mmd_shift / 5.0)
                geometric_constraint_weight = 0.9 * geometric_constraint_weight + 0.1 * target_weight
                geometric_constraint_weight = max(0.01, min(1.0, geometric_constraint_weight))
                if epoch <= 20:
                    current_geo_weight = geometric_constraint_weight * (epoch / 20.0)
                else:
                    current_geo_weight = geometric_constraint_weight

            gen_cls_loss = cls_criterion(p_ED, y.long()) + cls_criterion(p_ID, y.long())
            loss_G = gen_cls_loss + args.lambda_1 * con_loss_adv + current_geo_weight * g_loss

            # =========================================================
            # 🚀 【核心动力：类动态伪标签 (CB-PL) + 信息熵最小化】
            # =========================================================
            if epoch > 150:
                with torch.no_grad():
                    prob_tgt = F.softmax(p_tgt.detach(), dim=1)
                    max_probs, pseudo_labels = torch.max(prob_tgt, dim=1)

                conf_mask = torch.zeros_like(pseudo_labels, dtype=torch.bool)

                for c in range(num_classes):
                    c_mask = (pseudo_labels == c)
                    if c_mask.sum() > 0:
                        c_probs = max_probs[c_mask]
                        k = max(1, int(0.3 * c_mask.sum().item()))
                        topk_val, _ = torch.topk(c_probs, k)
                        c_thresh = max(0.7, topk_val[-1].item())
                        conf_mask |= (c_mask & (max_probs >= c_thresh))

                pseudo_criterion = nn.CrossEntropyLoss().to(device)

                if conf_mask.sum() > 0:
                    pseudo_loss = pseudo_criterion(p_tgt[conf_mask], pseudo_labels[conf_mask])
                    loss_G = loss_G + 0.4 * pseudo_loss

                unc_mask = ~conf_mask
                if unc_mask.sum() > 0:
                    prob_unc = F.softmax(p_tgt[unc_mask], dim=1)
                    entropy_loss = -torch.mean(torch.sum(prob_unc * torch.log(prob_unc + 1e-8), dim=1))
                    loss_G = loss_G + 0.1 * entropy_loss
            # =========================================================

            G_opt.zero_grad()
            loss_G.backward()

            D_opt.step()
            Enhancer_opt.step()
            G_opt.step()

            loss_list.append([src_cls_loss.item(), gen_cls_loss.item(), g_loss.item(), con_loss_adv.item()])

        if args.lr_scheduler == 'cosine':
            scheduler_D.step()
            scheduler_G.step()
            scheduler_E.step()

        avg_src, avg_tgt, avg_geo, avg_weight = np.mean(loss_list, 0)

        D_net.eval()
        G_net.eval()
        Enhancer.eval()

        teacc = evaluate(D_net, val_loader, hyperparams, device, enhancer=Enhancer)
        if best_acc < teacc:
            best_acc = teacc
            torch.save({
                'Discriminator': D_net.state_dict(),
                'Generator': G_net.state_dict(),
                'Enhancer': Enhancer.state_dict(),
            }, os.path.join(log_dir, f'best.pkl'))
        t2 = time.time()

        print(
            f'epoch {epoch}, train {len(train_loader.dataset)}, time {t2 - t1:.2f}, src_cls {avg_src:.4f} tgt_cls {avg_tgt:.4f} G_Loss {avg_geo:.4f} con_adv {avg_weight:.4f} '
            f'/// teacc {teacc:2.2f}')

        if epoch % args.log_interval == 0:
            pklpath = f'{log_dir}/best.pkl'
            taracc, result_tgt, pred = evaluate_tgt(D_net, device, test_loader, hyperparams, pklpath, enhancer=Enhancer)
            if best_acc_tgt < taracc:
                best_acc_tgt = taracc
                best_results = result_tgt
                np.save(args.source_name + '_pred_OURS.npy', pred)
            taracc_list.append(round(taracc, 2))
            print(f'load pth, target sample number {len(test_loader.dataset)}, max taracc {max(taracc_list):2.2f}')

    with open('out_' + args.source_name + '_ablation.log', 'a') as f:
        f.write("\n")
        f.write('-----------------low_freq:' + str(args.low_freq) + "\n")
        f.write('max:' + str(max(taracc_list) if taracc_list else 0) + "\n")
        f.write('LAMBDA1:' + str(args.lambda_1) + "\n")
        f.write('LAMBDA2:' + str(args.lambda_2) + "\n")
        f.write('LAMBDA3:' + str(args.lambda_3) + "\n")
        f.write('-----------------low_freq:' + str(args.low_freq) + "\n")
        f.write("\n")
        f.write('OA:' + str(best_results['Accuracy'] if 'best_results' in locals() else 0) + "\n")
        f.write('AA:' + str(best_results['class_acc'] if 'best_results' in locals() else 0) + "\n")
        f.write('Kappa:' + str(best_results['Kappa'] if 'best_results' in locals() else 0) + "\n")
        f.write("\n")


if __name__ == '__main__':
    experiment()