"""
VQGAN 微调脚本 - 适配动漫数据集

用途：
    在动漫数据集上微调预训练的 VQGAN，提高重建质量

运行方式：
    # 完整微调（推荐，需要较长时间）
    python scripts/finetune_vqgan.py --config configs/finetune_vqgan.yaml
    
    # 快速测试（验证流程）
    python scripts/finetune_vqgan.py --config configs/finetune_vqgan.yaml --max_steps 100

策略说明：
    1. decoder_only: 冻结 Encoder，只微调 Decoder + Codebook（推荐，显存友好）
    2. full: 全部微调（效果更好，但需要更多显存）

优化特性：
    - 混合精度训练（节省显存）
    - 梯度裁剪（稳定训练）
    - 修正的PSNR计算
    - 残差可视化
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
import math
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
from tqdm import tqdm
import yaml
from datetime import datetime

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.VQGAN.vqgan import VQModel
from model.VQGAN.lpips import LPIPS


class AnimeImageDataset(torch.utils.data.Dataset):
    """简单的动漫图像数据集（支持多种目录结构）"""
    def __init__(self, image_dir, image_size=256, split='train'):
        self.image_size = image_size
        self.image_paths = []
        
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        
        # 支持多种目录结构
        if os.path.exists(os.path.join(image_dir, split)):
            search_dir = os.path.join(image_dir, split)
        else:
            search_dir = image_dir
        
        # 递归搜索图像
        for root, dirs, files in os.walk(search_dir):
            for f in files:
                if Path(f).suffix.lower() in exts:
                    self.image_paths.append(os.path.join(root, f))
        
        print(f"[{split}] Found {len(self.image_paths)} images")
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip() if split == 'train' else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """加载单张图像（带异常处理）"""
        try:
            img = Image.open(self.image_paths[idx]).convert('RGB')
            # 如果是RGBA，转换为RGB（白色背景）
            if img.mode == 'RGBA':
                background = Image.new('RGB', img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background
            return self.transform(img)
        except Exception as e:
            print(f"[WARNING] 加载图像失败 {self.image_paths[idx]}: {e}")
            # 返回黑色图像作为占位符
            return torch.zeros(3, self.image_size, self.image_size)


def get_vqgan_config():
    """返回 VQGAN-f8 的默认配置"""
    from argparse import Namespace
    
    ddconfig = Namespace(
        double_z=False,
        z_channels=4,
        resolution=256,
        in_channels=3,
        out_ch=3,
        ch=128,
        ch_mult=(1, 2, 2, 4),
        num_res_blocks=2,
        attn_resolutions=(32,),
        dropout=0.0
    )
    
    lossconfig = Namespace(
        target='torch.nn.Identity'
    )
    
    return ddconfig, lossconfig


def load_vqgan(ckpt_path, device='cuda'):
    """加载 VQGAN 模型"""
    ddconfig, lossconfig = get_vqgan_config()
    
    model = VQModel(
        ddconfig=ddconfig,
        lossconfig=lossconfig,
        n_embed=16384,
        embed_dim=4,
        ckpt_path=ckpt_path
    )
    model = model.to(device)
    return model


class VQGANFineTuner:
    def __init__(self, config):
        self.config = config
        self.device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        # 混合精度训练
        self.use_amp = config.get('use_amp', True) and self.device.startswith('cuda')
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        
        # 加载模型
        print(f"[INFO] 加载预训练 VQGAN: {config['ckpt_path']}")
        self.model = load_vqgan(config['ckpt_path'], self.device)
        
        # 微调策略
        self.finetune_mode = config.get('finetune_mode', 'decoder_only')
        self._setup_trainable_params()
        
        # LPIPS 感知损失
        self.lpips = LPIPS().eval().to(self.device)
        for p in self.lpips.parameters():
            p.requires_grad = False
        
        # 优化器 (使用AdamW以获得更好的权重衰减)
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.get('lr', 1e-5),
            betas=(0.9, 0.999),
            weight_decay=config.get('weight_decay', 0.01)
        )
        
        # 学习率调度
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=config.get('max_steps', 50000),
            eta_min=config.get('lr', 1e-5) * 0.1
        )
        
        # 损失权重
        self.l1_weight = config.get('l1_weight', 1.0)
        self.lpips_weight = config.get('lpips_weight', 0.5)
        self.codebook_weight = config.get('codebook_weight', 0.25)
        
        # 梯度裁剪阈值
        self.grad_clip = config.get('grad_clip', 1.0)
        
        # 输出目录
        self.output_dir = config.get('output_dir', 'result/VQGAN-f8-finetuned')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'samples'), exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'checkpoints'), exist_ok=True)
        
        # 记录最佳指标
        self.best_psnr = 0
        self.best_lpips = float('inf')
    
    def _setup_trainable_params(self):
        """根据微调模式设置可训练参数"""
        if self.finetune_mode == 'decoder_only':
            # 冻结 Encoder，只训练 Decoder + Codebook
            print("[INFO] 微调模式: decoder_only (冻结 Encoder)")
            for p in self.model.encoder.parameters():
                p.requires_grad = False
            for p in self.model.quant_conv.parameters():
                p.requires_grad = False
            # Decoder + post_quant_conv + quantize 可训练
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"[INFO] 可训练参数: {trainable/1e6:.2f}M / {total/1e6:.2f}M")
            
        elif self.finetune_mode == 'codebook_only':
            # 只训练 Codebook
            print("[INFO] 微调模式: codebook_only")
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model.quantize.parameters():
                p.requires_grad = True
                
        elif self.finetune_mode == 'full':
            # 全部微调
            print("[INFO] 微调模式: full (全部参数)")
            trainable = sum(p.numel() for p in self.model.parameters())
            print(f"[INFO] 可训练参数: {trainable/1e6:.2f}M")
        
        else:
            raise ValueError(f"Unknown finetune_mode: {self.finetune_mode}")
    
    def compute_loss(self, x, x_rec, codebook_loss):
        """计算总损失"""
        # L1 重建损失
        l1_loss = F.l1_loss(x_rec, x)
        
        # LPIPS 感知损失
        lpips_loss = self.lpips(x_rec, x).mean()
        
        # 总损失
        total_loss = (
            self.l1_weight * l1_loss +
            self.lpips_weight * lpips_loss +
            self.codebook_weight * codebook_loss
        )
        
        return {
            'total': total_loss,
            'l1': l1_loss,
            'lpips': lpips_loss,
            'codebook': codebook_loss
        }
    
    def train_step(self, batch):
        """单步训练（支持混合精度和梯度裁剪）"""
        self.model.train()
        x = batch.to(self.device)
        
        # 混合精度训练
        with torch.cuda.amp.autocast(enabled=self.use_amp):
            # 前向传播
            x_rec, codebook_loss = self.model(x)
            
            # 计算损失
            losses = self.compute_loss(x, x_rec, codebook_loss)
        
        # 反向传播
        self.optimizer.zero_grad()
        
        if self.use_amp:
            self.scaler.scale(losses['total']).backward()
            if self.grad_clip > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            losses['total'].backward()
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            self.optimizer.step()
        
        self.scheduler.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    @torch.no_grad()
    def evaluate(self, dataloader, num_batches=10):
        """评估（修正的PSNR计算）"""
        self.model.eval()
        metrics = {'mse': [], 'psnr': [], 'l1': [], 'lpips': []}
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            x = batch.to(self.device)
            x_rec, _ = self.model(x)
            
            # 计算各项指标
            mse = F.mse_loss(x_rec, x).item()
            metrics['mse'].append(mse)
            
            # 修正的PSNR计算：data_range = 2.0 (从-1到1)
            if mse < 1e-10:
                psnr = 100.0
            else:
                psnr = 10 * math.log10(4.0 / mse)  # 4 = 2^2
            
            metrics['psnr'].append(psnr)
            metrics['l1'].append(F.l1_loss(x_rec, x).item())
            metrics['lpips'].append(self.lpips(x_rec, x).mean().item())
        
        return {k: np.mean(v) for k, v in metrics.items()}
    
    @torch.no_grad()
    def save_samples(self, dataloader, step, num_samples=4):
        """保存样本对比图（原图+重建+残差）"""
        self.model.eval()
        batch = next(iter(dataloader))[:num_samples].to(self.device)
        x_rec, _ = self.model(batch)
        
        # 反归一化到[0, 1]
        def denorm(t):
            return ((t + 1) / 2).clamp(0, 1)
        
        # 计算残差（绝对值）
        residual = torch.abs(batch - x_rec)
        residual = residual / (residual.max() + 1e-8)  # 归一化到[0, 1]
        
        # 创建对比网格：原图 | 重建 | 残差
        comparison = []
        for i in range(num_samples):
            comparison.extend([
                denorm(batch[i]),
                denorm(x_rec[i]),
                residual[i]
            ])
        
        # 保存
        from torchvision.utils import save_image, make_grid
        grid = make_grid(comparison, nrow=3, padding=2, pad_value=1.0)
        save_image(grid, os.path.join(self.output_dir, 'samples', f'step_{step:06d}.png'))
    
    def save_checkpoint(self, step, metrics=None, is_best=False):
        """保存 checkpoint"""
        ckpt = {
            'step': step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'scaler': self.scaler.state_dict() if self.use_amp else None,
            'config': self.config,
            'metrics': metrics,
            'best_psnr': self.best_psnr,
            'best_lpips': self.best_lpips
        }
        
        # 保存完整checkpoint
        ckpt_path = os.path.join(self.output_dir, 'checkpoints', f'checkpoint_step{step:06d}.pth')
        torch.save(ckpt, ckpt_path)
        
        # 保存为兼容格式（与原始 VQGAN 一致）
        ckpt_compat = {'state_dict': self.model.state_dict()}
        torch.save(ckpt_compat, os.path.join(self.output_dir, 'model_latest.ckpt'))
        
        # 如果是最好模型，额外保存
        if is_best:
            torch.save(ckpt_compat, os.path.join(self.output_dir, 'model_best.ckpt'))
            print(f"[INFO] ⭐ Best model saved at step {step}")
        
        print(f"[INFO] Checkpoint saved: step {step}")
        
        # 保留最近5个checkpoint
        self._cleanup_old_checkpoints(keep=5)
    
    def _cleanup_old_checkpoints(self, keep=5):
        """清理旧的checkpoint文件，节省磁盘空间"""
        checkpoint_dir = os.path.join(self.output_dir, 'checkpoints')
        if not os.path.exists(checkpoint_dir):
            return
        
        checkpoints = []
        for f in os.listdir(checkpoint_dir):
            if f.startswith('checkpoint_step') and f.endswith('.pth'):
                try:
                    step = int(f[15:-4])  # 提取步数
                    checkpoints.append((step, os.path.join(checkpoint_dir, f)))
                except:
                    pass
        
        # 按步数排序，删除最旧的
        checkpoints.sort(key=lambda x: x[0])
        while len(checkpoints) > keep:
            _, old_path = checkpoints.pop(0)
            try:
                os.remove(old_path)
            except:
                pass
    
    def train(self, train_loader, val_loader, max_steps, eval_interval=1000, save_interval=5000):
        """训练循环"""
        print(f"\n[INFO] 开始微调 VQGAN")
        print(f"[INFO] 最大步数: {max_steps}, 评估间隔: {eval_interval}, 保存间隔: {save_interval}")
        print(f"[INFO] 混合精度训练: {self.use_amp}")
        
        # 初始评估
        print("\n[INFO] 初始评估...")
        init_metrics = self.evaluate(val_loader)
        print(f"[INIT] PSNR: {init_metrics['psnr']:.2f} dB, LPIPS: {init_metrics['lpips']:.4f}")
        
        step = 0
        running_losses = {'total': 0, 'l1': 0, 'lpips': 0, 'codebook': 0}
        running_count = 0
        
        pbar = tqdm(total=max_steps, desc="微调 VQGAN")
        
        while step < max_steps:
            for batch in train_loader:
                if step >= max_steps:
                    break
                
                losses = self.train_step(batch)
                
                # 累积损失
                for k, v in losses.items():
                    if k in running_losses:
                        running_losses[k] += v
                running_count += 1
                
                step += 1
                pbar.update(1)
                
                # 日志输出
                if step % 100 == 0 and running_count > 0:
                    avg_losses = {k: v / running_count for k, v in running_losses.items()}
                    pbar.set_postfix({
                        'loss': f"{avg_losses['total']:.4f}",
                        'l1': f"{avg_losses['l1']:.4f}",
                        'lpips': f"{avg_losses['lpips']:.4f}",
                        'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                    })
                    # 重置累计
                    running_losses = {k: 0 for k in running_losses}
                    running_count = 0
                
                # 评估
                if step % eval_interval == 0:
                    self.model.eval()
                    metrics = self.evaluate(val_loader)
                    
                    print(f"\n{'='*60}")
                    print(f"[Step {step}]")
                    print(f"  PSNR:  {metrics['psnr']:.2f} dB (Δ{metrics['psnr']-init_metrics['psnr']:+.2f})")
                    print(f"  LPIPS: {metrics['lpips']:.4f} (Δ{metrics['lpips']-init_metrics['lpips']:+.4f})")
                    print(f"  L1:    {metrics['l1']:.6f}")
                    print(f"{'='*60}")
                    
                    # 保存样本
                    self.save_samples(val_loader, step)
                    
                    # 检查是否是最佳模型
                    is_best = False
                    if metrics['psnr'] > self.best_psnr:
                        self.best_psnr = metrics['psnr']
                        is_best = True
                    if metrics['lpips'] < self.best_lpips:
                        self.best_lpips = metrics['lpips']
                        is_best = True
                    
                    # 保存checkpoint
                    if is_best or step % save_interval == 0:
                        self.save_checkpoint(step, metrics, is_best=is_best)
                
                # 定期保存
                if step % save_interval == 0:
                    self.save_checkpoint(step)
        
        pbar.close()
        
        # 最终评估
        print("\n" + "="*70)
        print("VQGAN 微调完成!")
        print("="*70)
        
        final_metrics = self.evaluate(val_loader)
        print(f"最终指标:")
        print(f"  PSNR:  {final_metrics['psnr']:.2f} dB (初始: {init_metrics['psnr']:.2f} dB, 提升: {final_metrics['psnr']-init_metrics['psnr']:+.2f})")
        print(f"  LPIPS: {final_metrics['lpips']:.4f} (初始: {init_metrics['lpips']:.4f}, 改善: {init_metrics['lpips']-final_metrics['lpips']:+.4f})")
        print(f"  L1:    {final_metrics['l1']:.6f}")

        print(f"\n最佳指标:")
        print(f"  PSNR:  {self.best_psnr:.2f} dB")
        print(f"  LPIPS: {self.best_lpips:.4f}")
        
        # 保存最终模型
        self.save_checkpoint(step, final_metrics, is_best=True)
        
        print(f"\n模型保存至:")
        print(f"  {self.output_dir}/model_best.ckpt (最佳模型)")
        print(f"  {self.output_dir}/model_latest.ckpt (最新模型)")
        print(f"  {self.output_dir}/checkpoints/ (完整checkpoints)")
        print(f"  {self.output_dir}/samples/ (重建样本)")
        print("="*70)


def main():
    parser = argparse.ArgumentParser(description='微调 VQGAN')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--ckpt', type=str, default='result/VQGAN-f8/model.ckpt', help='预训练权重')
    parser.add_argument('--data_dir', type=str, default='datasets', help='数据目录')
    parser.add_argument('--output_dir', type=str, default='result/VQGAN-f8-anime', help='输出目录')
    parser.add_argument('--finetune_mode', type=str, default='decoder_only', 
                        choices=['decoder_only', 'codebook_only', 'full'], help='微调模式')
    parser.add_argument('--max_steps', type=int, default=20000, help='最大训练步数')
    parser.add_argument('--batch_size', type=int, default=2, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    parser.add_argument('--use_amp', action='store_true', default=True, help='使用混合精度训练')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪阈值')
    args = parser.parse_args()
    
    # 构建配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        config = {
            'ckpt_path': args.ckpt,
            'data_dir': args.data_dir,
            'output_dir': args.output_dir,
            'finetune_mode': args.finetune_mode,
            'max_steps': args.max_steps,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'device': args.device if torch.cuda.is_available() else 'cpu',
            'use_amp': args.use_amp and torch.cuda.is_available(),
            'grad_clip': args.grad_clip,
            'l1_weight': 1.0,
            'lpips_weight': 0.5,
            'codebook_weight': 0.25,
            'weight_decay': 0.01,
        }
    
    print("="*50)
    print("VQGAN 微调配置")
    print("="*50)
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("="*50)
    
    # 数据集
    train_dataset = AnimeImageDataset(config['data_dir'], split='train')
    
    # 验证集：优先使用val目录，否则从训练集中分割
    val_path = os.path.join(config['data_dir'], 'val')
    if os.path.exists(val_path):
        val_dataset = AnimeImageDataset(config['data_dir'], split='val')
        if len(val_dataset) == 0:
            print("[WARNING] val目录存在但为空，将从训练集中划分20%作为验证集")
            train_size = int(0.8 * len(train_dataset))
            val_size = len(train_dataset) - train_size
            train_dataset, val_dataset = torch.utils.data.random_split(
                train_dataset, [train_size, val_size]
            )
            print(f"[INFO] 数据集划分: 训练集 {train_size}, 验证集 {val_size}")
    else:
        print("[INFO] 未找到val目录，从训练集中划分20%作为验证集")
        if len(train_dataset) < 10:
            print("[ERROR] 训练集图像数量过少（< 10），无法有效训练")
            sys.exit(1)
        train_size = int(0.8 * len(train_dataset))
        val_size = len(train_dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [train_size, val_size]
        )
        print(f"[INFO] 数据集划分: 训练集 {train_size}, 验证集 {val_size}")
    
    # Windows平台num_workers兼容性处理
    num_workers = 0 if os.name == 'nt' else min(4, os.cpu_count() or 1)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=max(1, num_workers // 2) if num_workers > 0 else 0,
        pin_memory=torch.cuda.is_available()
    )
    
    # 微调器
    finetuner = VQGANFineTuner(config)
    
    # 训练
    try:
        finetuner.train(
            train_loader,
            val_loader,
            max_steps=config['max_steps'],
            eval_interval=min(1000, config['max_steps'] // 10),
            save_interval=min(5000, config['max_steps'] // 4)
        )
    except KeyboardInterrupt:
        print("\n[INFO] 训练被用户中断 (Ctrl+C)")
        print("[INFO] 保存当前状态...")
        # 保存当前模型状态
        import time
        timestamp = int(time.time())
        finetuner.save_checkpoint(timestamp, is_best=False)
        print("[INFO] 模型已保存，可稍后继续训练")
    except Exception as e:
        print(f"\n[ERROR] 训练出错: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
