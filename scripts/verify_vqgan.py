"""
VQGAN 预训练模型验证脚本

用途：
1. 验证 VQGAN-f8 权重能否正常加载
2. 测试编码-解码重建质量
3. 评估是否需要针对动漫数据集微调

运行方式：
    python scripts/verify_vqgan.py --ckpt result/VQGAN-f8/model.ckpt --image_dir datasets/test/reference --output_dir result/vqgan_verify
"""

import argparse
import os
import sys
import torch
import torch.nn.functional as F
import numpy as np
import math
from PIL import Image
from pathlib import Path
import torchvision.transforms as transforms
from tqdm import tqdm

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.VQGAN.vqgan import VQModel


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
    model.eval()
    return model


def compute_metrics(original, reconstructed):
    """计算重建质量指标（修正的PSNR计算）"""
    # MSE
    mse = F.mse_loss(reconstructed, original).item()
    
    # PSNR（修正版）- data_range = 2.0 (从-1到1)
    if mse < 1e-10:
        psnr = 100.0  # 避免除以零
    else:
        psnr = 10 * math.log10(4.0 / mse)  # 4 = 2^2 (data_range^2)
    
    # L1
    l1 = F.l1_loss(reconstructed, original).item()
    
    return {
        'mse': mse,
        'psnr': psnr,
        'l1': l1
    }


def verify_single_image(model, image_path, output_dir, device='cuda'):
    """验证单张图像的重建"""
    # 加载图像
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
    ])
    
    image = Image.open(image_path).convert('RGB')
    x = transform(image).unsqueeze(0).to(device)
    
    # 编码-解码
    with torch.no_grad():
        # 编码
        h = model.encoder(x)
        h = model.quant_conv(h)
        quant, emb_loss, info = model.quantize(h)
        
        # 解码
        x_rec = model.decode(quant)
    
    # 计算指标
    metrics = compute_metrics(x, x_rec)
    
    # 保存结果
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # 反归一化
        def denorm(t):
            return ((t + 1) / 2).clamp(0, 1)
        
        # 保存原图和重建图
        name = Path(image_path).stem
        
        # 拼接对比图
        comparison = torch.cat([denorm(x), denorm(x_rec)], dim=3)  # 水平拼接
        comparison = comparison.squeeze(0).permute(1, 2, 0).cpu().numpy()
        comparison = (comparison * 255).astype(np.uint8)
        Image.fromarray(comparison).save(os.path.join(output_dir, f'{name}_compare.png'))
    
    return metrics, h.shape


def main():
    parser = argparse.ArgumentParser(description='验证 VQGAN 预训练模型')
    parser.add_argument('--ckpt', type=str, default='result/VQGAN-f8/model.ckpt',
                        help='VQGAN checkpoint 路径')
    parser.add_argument('--image_dir', type=str, default=None,
                        help='测试图像目录')
    parser.add_argument('--image', type=str, default=None,
                        help='单张测试图像路径')
    parser.add_argument('--output_dir', type=str, default='result/vqgan_verify',
                        help='输出目录')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--max_images', type=int, default=20,
                        help='最大测试图像数量')
    args = parser.parse_args()
    
    # 检查 checkpoint 是否存在
    if not os.path.exists(args.ckpt):
        print(f"[ERROR] Checkpoint 不存在: {args.ckpt}")
        sys.exit(1)
    
    # 设备
    device = args.device if torch.cuda.is_available() else 'cpu'
    print(f"[INFO] 使用设备: {device}")
    
    # 加载模型
    print(f"[INFO] 加载 VQGAN 模型: {args.ckpt}")
    try:
        model = load_vqgan(args.ckpt, device)
        print("[OK] VQGAN 模型加载成功!")
    except Exception as e:
        print(f"[ERROR] 模型加载失败: {e}")
        sys.exit(1)
    
    # 打印模型信息
    total_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] 模型参数量: {total_params / 1e6:.2f}M")
    
    # 收集测试图像
    image_paths = []
    if args.image:
        image_paths.append(args.image)
    elif args.image_dir and os.path.exists(args.image_dir):
        exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        for f in os.listdir(args.image_dir):
            if Path(f).suffix.lower() in exts:
                image_paths.append(os.path.join(args.image_dir, f))
        image_paths = image_paths[:args.max_images]
    
    if not image_paths:
        print("[WARN] 未找到测试图像，仅验证模型加载")
        print("\n" + "="*50)
        print("✅ VQGAN 模型验证通过（仅加载测试）")
        print("="*50)
        return
    
    # 测试重建
    print(f"\n[INFO] 测试 {len(image_paths)} 张图像的重建质量...")
    all_metrics = {'mse': [], 'psnr': [], 'l1': []}
    
    for img_path in tqdm(image_paths, desc="验证中"):
        try:
            metrics, latent_shape = verify_single_image(model, img_path, args.output_dir, device)
            for k, v in metrics.items():
                all_metrics[k].append(v)
        except Exception as e:
            print(f"[WARN] 处理失败 {img_path}: {e}")
    
    # 汇总结果
    print("\n" + "="*50)
    print("VQGAN 重建质量评估结果")
    print("="*50)
    print(f"测试图像数: {len(all_metrics['mse'])}")
    print(f"Latent 形状: {latent_shape} (压缩比 f8: 256x256 -> 32x32)")
    print("-"*50)
    print(f"平均 MSE:  {np.mean(all_metrics['mse']):.6f}")
    print(f"平均 PSNR: {np.mean(all_metrics['psnr']):.2f} dB")
    print(f"平均 L1:   {np.mean(all_metrics['l1']):.6f}")
    print("-"*50)
    
    # 判断是否需要微调
    avg_psnr = np.mean(all_metrics['psnr'])
    print("\n[评估建议]")
    if avg_psnr >= 30:
        print("✅ PSNR >= 30dB，重建质量优秀，无需微调 VQGAN")
        print("   可以直接用于 Latent BBDM 训练")
    elif avg_psnr >= 25:
        print("⚠️  PSNR 在 25-30dB，重建质量一般")
        print("   建议: 可以先尝试训练，如果生成结果有明显失真再考虑微调")
    else:
        print("❌ PSNR < 25dB，重建质量较差")
        print("   强烈建议: 在动漫数据集上微调 VQGAN")
    
    print(f"\n[INFO] 对比图已保存至: {args.output_dir}")


if __name__ == '__main__':
    main()
