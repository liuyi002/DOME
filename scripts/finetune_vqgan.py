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
    1. 冻结 Encoder，只微调 Decoder + Codebook（快速，推荐先尝试）
    2. 全部微调（效果更好，但需要更多时间和显存）
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
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
    """简单的动漫图像数据集"""
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
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return self.transform(img)


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
        self.device = config.get('device', 'cuda')
        
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
        
        # 优化器
        self.optimizer = Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.get('lr', 1e-5),
            betas=(0.9, 0.999)
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
        
        # 输出目录
        self.output_dir = config.get('output_dir', 'result/VQGAN-f8-finetuned')
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(os.path.join(self.output_dir, 'samples'), exist_ok=True)
    
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
        """单步训练"""
        self.model.train()
        x = batch.to(self.device)
        
        # 前向传播
        x_rec, codebook_loss = self.model(x)
        
        # 计算损失
        losses = self.compute_loss(x, x_rec, codebook_loss)
        
        # 反向传播
        self.optimizer.zero_grad()
        losses['total'].backward()
        self.optimizer.step()
        self.scheduler.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    @torch.no_grad()
    def evaluate(self, dataloader, num_batches=10):
        """评估"""
        self.model.eval()
        metrics = {'mse': [], 'psnr': [], 'l1': [], 'lpips': []}
        
        for i, batch in enumerate(dataloader):
            if i >= num_batches:
                break
            
            x = batch.to(self.device)
            x_rec, _ = self.model(x)
            
            mse = F.mse_loss(x_rec, x).item()
            metrics['mse'].append(mse)
            metrics['psnr'].append(10 * np.log10(1.0 / mse) if mse > 0 else 40)
            metrics['l1'].append(F.l1_loss(x_rec, x).item())
            metrics['lpips'].append(self.lpips(x_rec, x).mean().item())
        
        return {k: np.mean(v) for k, v in metrics.items()}
    
    @torch.no_grad()
    def save_samples(self, dataloader, step, num_samples=4):
        """保存样本对比图"""
        self.model.eval()
        batch = next(iter(dataloader))[:num_samples].to(self.device)
        x_rec, _ = self.model(batch)
        
        # 反归一化
        def denorm(t):
            return ((t + 1) / 2).clamp(0, 1)
        
        # 拼接对比图
        comparison = torch.cat([denorm(batch), denorm(x_rec)], dim=3)
        
        # 保存
        from torchvision.utils import save_image
        save_image(comparison, os.path.join(self.output_dir, 'samples', f'step_{step:06d}.png'), nrow=1)
    
    def save_checkpoint(self, step, metrics=None):
        """保存 checkpoint"""
        ckpt = {
            'step': step,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'config': self.config,
            'metrics': metrics
        }
        
        # 保存为兼容格式（与原始 VQGAN 一致）
        ckpt_compat = {'state_dict': self.model.state_dict()}
        
        torch.save(ckpt, os.path.join(self.output_dir, f'checkpoint_step{step}.pth'))
        torch.save(ckpt_compat, os.path.join(self.output_dir, 'model.ckpt'))
        print(f"[INFO] Checkpoint saved at step {step}")
    
    def train(self, train_loader, val_loader, max_steps, eval_interval=1000, save_interval=5000):
        """训练循环"""
        print(f"\n[INFO] 开始微调 VQGAN")
        print(f"[INFO] 最大步数: {max_steps}, 评估间隔: {eval_interval}, 保存间隔: {save_interval}")
        
        step = 0
        best_psnr = 0
        running_losses = {'total': 0, 'l1': 0, 'lpips': 0, 'codebook': 0}
        
        pbar = tqdm(total=max_steps, desc="微调 VQGAN")
        
        while step < max_steps:
            for batch in train_loader:
                if step >= max_steps:
                    break
                
                losses = self.train_step(batch)
                
                for k, v in losses.items():
                    running_losses[k] += v
                
                step += 1
                pbar.update(1)
                
                # 日志
                if step % 100 == 0:
                    avg_losses = {k: v / 100 for k, v in running_losses.items()}
                    pbar.set_postfix({
                        'loss': f"{avg_losses['total']:.4f}",
                        'l1': f"{avg_losses['l1']:.4f}",
                        'lpips': f"{avg_losses['lpips']:.4f}",
                        'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
                    })
                    running_losses = {k: 0 for k in running_losses}
                
                # 评估
                if step % eval_interval == 0:
                    metrics = self.evaluate(val_loader)
                    print(f"\n[Step {step}] Val PSNR: {metrics['psnr']:.2f} dB, LPIPS: {metrics['lpips']:.4f}")
                    
                    # 保存样本
                    self.save_samples(val_loader, step)
                    
                    # 保存最佳模型
                    if metrics['psnr'] > best_psnr:
                        best_psnr = metrics['psnr']
                        self.save_checkpoint(step, metrics)
                        print(f"[INFO] 新最佳 PSNR: {best_psnr:.2f} dB")
                
                # 定期保存
                if step % save_interval == 0:
                    self.save_checkpoint(step)
        
        pbar.close()
        
        # 最终保存
        final_metrics = self.evaluate(val_loader)
        self.save_checkpoint(step, final_metrics)
        
        print("\n" + "="*50)
        print("VQGAN 微调完成!")
        print("="*50)
        print(f"最终 PSNR: {final_metrics['psnr']:.2f} dB")
        print(f"最终 LPIPS: {final_metrics['lpips']:.4f}")
        print(f"最佳 PSNR: {best_psnr:.2f} dB")
        print(f"模型保存至: {self.output_dir}/model.ckpt")


def main():
    parser = argparse.ArgumentParser(description='微调 VQGAN')
    parser.add_argument('--config', type=str, default=None, help='配置文件路径')
    parser.add_argument('--ckpt', type=str, default='result/VQGAN-f8/model.ckpt', help='预训练权重')
    parser.add_argument('--data_dir', type=str, default='datasets', help='数据目录')
    parser.add_argument('--output_dir', type=str, default='result/VQGAN-f8-anime', help='输出目录')
    parser.add_argument('--finetune_mode', type=str, default='decoder_only', 
                        choices=['decoder_only', 'codebook_only', 'full'], help='微调模式')
    parser.add_argument('--max_steps', type=int, default=20000, help='最大训练步数')
    parser.add_argument('--batch_size', type=int, default=4, help='批次大小')
    parser.add_argument('--lr', type=float, default=1e-5, help='学习率')
    parser.add_argument('--device', type=str, default='cuda', help='设备')
    args = parser.parse_args()
    
    # 构建配置
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r') as f:
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
            'l1_weight': 1.0,
            'lpips_weight': 0.5,
            'codebook_weight': 0.25,
        }
    
    print("="*50)
    print("VQGAN 微调配置")
    print("="*50)
    for k, v in config.items():
        print(f"  {k}: {v}")
    print("="*50)
    
    # 数据集
    train_dataset = AnimeImageDataset(config['data_dir'], split='train')
    val_dataset = AnimeImageDataset(config['data_dir'], split='val')
    
    if len(train_dataset) == 0:
        print("[ERROR] 训练集为空，请检查 data_dir 路径")
        sys.exit(1)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset if len(val_dataset) > 0 else train_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=2
    )
    
    # 微调器
    finetuner = VQGANFineTuner(config)
    
    # 训练
    finetuner.train(
        train_loader,
        val_loader,
        max_steps=config['max_steps'],
        eval_interval=min(1000, config['max_steps'] // 10),
        save_interval=min(5000, config['max_steps'] // 4)
    )


if __name__ == '__main__':
    main()
