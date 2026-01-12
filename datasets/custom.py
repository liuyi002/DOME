import random
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from Register import Registers
from datasets.base import ImagePathDataset
from datasets.utils import get_image_paths_from_dir
from PIL import Image
import cv2
import os

from datasets.tps_warp import warp_image, get_tps_scale_for_stage


@Registers.datasets.register_with_name('custom_single')
class CustomSingleDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal

        self.imgs = ImagePathDataset(image_paths, self.image_size, flip=self.flip, to_normal=self.to_normal)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i], self.imgs[i]


@Registers.datasets.register_with_name('custom_aligned')
class CustomAlignedDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        image_paths_ori = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/B'))
        image_paths_cond = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, f'{stage}/A'))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal

        self.imgs_ori = ImagePathDataset(image_paths_ori, self.image_size, flip=self.flip, to_normal=self.to_normal)
        self.imgs_cond = ImagePathDataset(image_paths_cond, self.image_size, flip=self.flip, to_normal=self.to_normal)

    def __len__(self):
        return len(self.imgs_ori)

    def __getitem__(self, i):
        return self.imgs_ori[i], self.imgs_cond[i]


@Registers.datasets.register_with_name('custom_colorization_LAB')
class CustomColorizationLABDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = False
        if index >= self._length:
            index = index - self._length
            p = True

        img_path = self.image_paths[index]
        image = None
        try:
            image = cv2.imread(img_path)
            if self.to_lab:
                image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        except BaseException as e:
            print(img_path)

        if p:
            image = cv2.flip(image, 1)
        image = cv2.resize(image, self.image_size, interpolation=cv2.INTER_LINEAR)
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1).contiguous()

        if self.to_normal:
            image = (image - 127.5) / 127.5
            image.clamp_(-1., 1.)

        L = image[0:1, :, :]
        ab = image[1:, :, :]
        cond = torch.cat((L, L, L), dim=0)
        return image, cond


@Registers.datasets.register_with_name('custom_colorization_RGB')
class CustomColorizationRGBDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = False
        if index >= self._length:
            index = index - self._length
            p = True

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        cond_image = image.convert('L')
        cond_image = cond_image.convert('RGB')

        image = transform(image)
        cond_image = transform(cond_image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)
            cond_image = (cond_image - 0.5) * 2.
            cond_image.clamp_(-1., 1.)

        image_name = Path(img_path).stem
        return (image, image_name), (cond_image, image_name)


@Registers.datasets.register_with_name('custom_inpainting')
class CustomInpaintingDataset(Dataset):
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.image_paths = get_image_paths_from_dir(os.path.join(dataset_config.dataset_path, stage))
        self.flip = dataset_config.flip if stage == 'train' else False
        self.to_normal = dataset_config.to_normal
        self._length = len(self.image_paths)

    def __len__(self):
        if self.flip:
            return self._length * 2
        return self._length

    def __getitem__(self, index):
        p = 0.
        if index >= self._length:
            index = index - self._length
            p = 1.

        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(p=p),
            transforms.Resize(self.image_size),
            transforms.ToTensor()
        ])

        img_path = self.image_paths[index]
        image = None
        try:
            image = Image.open(img_path)
        except BaseException as e:
            print(img_path)

        if not image.mode == 'RGB':
            image = image.convert('RGB')

        image = transform(image)

        if self.to_normal:
            image = (image - 0.5) * 2.
            image.clamp_(-1., 1.)

        height, width = self.image_size
        mask_width = random.randint(128, 180)
        mask_height = random.randint(128, 180)
        mask_pos_x = random.randint(0, height - mask_height)
        mask_pos_y = random.randint(0, width - mask_width)
        mask = torch.ones_like(image)
        mask[:, mask_pos_x:mask_pos_x+mask_height, mask_pos_y:mask_pos_y+mask_width] = 0

        cond_image = image * mask

        image_name = Path(img_path).stem
        return (image, image_name), (cond_image, image_name)


@Registers.datasets.register_with_name('custom_lineart_colorization')
class LineartColorizationDataset(Dataset):
    """
    动漫线稿上色数据集
    支持三元组数据：线稿(L)、参考彩图(I_warped)、真值彩图(I_gt)
    
    支持特性:
    - TPS 课程学习：通过 tps_scale 或 curriculum_stage 控制变形强度
    - 自参考增强：use_self_reference=True 时，对 reference 应用变形作为 warped
    
    目录结构:
        dataset_path/
        ├── train/
        │   ├── sketch/      # 线稿图像
        │   └── reference/   # 参考彩图（同时作为 ground_truth）
        ├── val/ ...
        └── test/ ...
    """
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.dataset_path = dataset_config.dataset_path
        self.stage = stage
        self.flip = getattr(dataset_config, 'flip', False) if stage == 'train' else False
        self.to_normal = getattr(dataset_config, 'to_normal', True)
        self.distorted_condition = getattr(dataset_config, 'distorted_condition', True)
        
        # TPS 课程学习配置
        # 方式1: 直接指定 tps_scale (0.0 ~ 1.0)
        # 方式2: 指定 curriculum_stage (1, 2, 3) 自动映射到预设强度
        self.tps_scale = getattr(dataset_config, 'tps_scale', None)
        self.curriculum_stage = getattr(dataset_config, 'curriculum_stage', None)
        
        # 确定最终的 TPS 变形强度
        if self.tps_scale is not None:
            self._tps_distortion_scale = self.tps_scale
        elif self.curriculum_stage is not None:
            self._tps_distortion_scale = get_tps_scale_for_stage(self.curriculum_stage)
        else:
            self._tps_distortion_scale = 0.25  # 默认值

        # 设置三个数据路径
        self.sketch_path = os.path.join(self.dataset_path, stage, 'sketch')
        self.reference_path = os.path.join(self.dataset_path, stage, 'reference')
        # self.reference_warped_path = os.path.join(self.dataset_path, stage, 'reference_warped')
        
        # 验证路径是否存在
        for path in [self.sketch_path, self.reference_path]:  # Removed reference_warped_path
            if not os.path.exists(path):
                raise FileNotFoundError(f"Path does not exist: {path}")
        
        # 获取各文件夹中的文件列表
        sketch_files = set(os.listdir(self.sketch_path))
        reference_files = set(os.listdir(self.reference_path))
        # warped_files = set(os.listdir(self.reference_warped_path))
        
        # 检查文件名一致性
        if sketch_files != reference_files:
            # 找出不一致的文件
            missing_in_reference = sketch_files - reference_files
            extra_in_reference = reference_files - sketch_files

            
            error_msg = "文件名不一致:\n"
            if missing_in_reference:
                error_msg += f"在reference文件夹中缺少的文件: {missing_in_reference}\n"
            if extra_in_reference:
                error_msg += f"在reference文件夹中多余的文件: {extra_in_reference}\n"
            
            raise ValueError(error_msg)
            
        # 使用交集，确保所有文件夹都有相同的文件名
        self.file_list = sorted(list(sketch_files.intersection(reference_files)))
        print(f"Found {len(self.file_list)} samples in {stage} split with consistent filenames")
        print(f"TPS distortion scale: {self._tps_distortion_scale}")
        
        # 创建图像转换
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])
        
        if self.to_normal:
            # 归一化到[-1, 1]
            self.transform = transforms.Compose([
                self.transform,
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        """
        返回三元组数据
        Returns:
            dict: {
                'L': 线稿图 (条件输入),
                'I_warped': 参考彩图 (主输入), 
                'I_gt': 真值彩图 (监督信号),
                'filename': 文件名
            }
        """
        filename = self.file_list[index]
        
        # 加载三张图像
        sketch_path = os.path.join(self.sketch_path, filename)
        reference_path = os.path.join(self.reference_path, filename)
        
        sketch_img = Image.open(sketch_path).convert('RGB')
        reference_img = Image.open(reference_path).convert('RGB')
        if self.distorted_condition:
            # 使用课程学习的 TPS 强度
            warped_img = warp_image(reference_img, distortion_scale=self._tps_distortion_scale)
        else:
            warped_img = reference_img
        
        # 应用随机翻转增强（训练时）
        if self.flip and self.stage == 'train' and random.random() > 0.5:
            sketch_img = TF.hflip(sketch_img)
            reference_img = TF.hflip(reference_img)
            warped_img = TF.hflip(warped_img)
        
        # 应用变换
        sketch_tensor = self.transform(sketch_img)
        reference_tensor = self.transform(reference_img)
        warped_tensor = self.transform(warped_img)
        
        return {
            'L': sketch_tensor,        # 线稿图 (条件输入)
            'I_warped': warped_tensor, # 参考彩图 (主输入)
            'I_gt': reference_tensor,  # 真值彩图 (监督信号)
            'filename': filename
        }
    
    def set_tps_scale(self, scale: float):
        """
        动态更新 TPS 变形强度（供 Runner 在阶段切换时调用）
        
        :param scale: 新的变形强度 (0.0 ~ 1.0)
        """
        self._tps_distortion_scale = scale
        print(f"[Dataset] TPS distortion scale updated to: {scale}")
    
    def set_curriculum_stage(self, stage: int):
        """
        根据训练阶段更新 TPS 强度
        
        :param stage: 训练阶段 (1, 2, 3)
        """
        self._tps_distortion_scale = get_tps_scale_for_stage(stage)
        print(f"[Dataset] Curriculum stage {stage}, TPS scale: {self._tps_distortion_scale}")


@Registers.datasets.register_with_name('custom_lineart_self_reference')
class LineartSelfReferenceDataset(Dataset):
    """
    自参考增强数据集（用于阶段1训练）
    
    与 LineartColorizationDataset 不同，该数据集:
    - 只需要 ground_truth 彩图
    - 自动从彩图提取线稿（使用边缘检测）
    - reference 直接使用 ground_truth（或轻微变形版本）
    
    目录结构:
        dataset_path/
        ├── train/
        │   └── images/   # 彩色图像（作为 ground_truth）
        ├── val/ ...
        └── test/ ...
    """
    def __init__(self, dataset_config, stage='train'):
        super().__init__()
        self.image_size = (dataset_config.image_size, dataset_config.image_size)
        self.dataset_path = dataset_config.dataset_path
        self.stage = stage
        self.flip = getattr(dataset_config, 'flip', False) if stage == 'train' else False
        self.to_normal = getattr(dataset_config, 'to_normal', True)
        
        # 自参考时使用轻微变形
        self.tps_scale = getattr(dataset_config, 'tps_scale', 0.05)
        
        # 线稿提取配置
        self.canny_low = getattr(dataset_config, 'canny_low', 50)
        self.canny_high = getattr(dataset_config, 'canny_high', 150)
        
        # 数据路径
        self.images_path = os.path.join(self.dataset_path, stage, 'images')
        if not os.path.exists(self.images_path):
            # 兼容：尝试 reference 文件夹
            self.images_path = os.path.join(self.dataset_path, stage, 'reference')
        
        if not os.path.exists(self.images_path):
            raise FileNotFoundError(f"Path does not exist: {self.images_path}")
        
        self.file_list = sorted(os.listdir(self.images_path))
        print(f"[SelfRef] Found {len(self.file_list)} samples in {stage} split")
        print(f"[SelfRef] TPS scale: {self.tps_scale}")
        
        # 图像转换
        self.transform = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])
        
        if self.to_normal:
            self.transform = transforms.Compose([
                self.transform,
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
    
    def _extract_lineart(self, image: Image.Image) -> Image.Image:
        """
        从彩色图像提取线稿（使用 Canny 边缘检测）
        
        :param image: PIL Image (RGB)
        :return: PIL Image (RGB, 白底黑线)
        """
        # 转为灰度
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        # Canny 边缘检测
        edges = cv2.Canny(gray, self.canny_low, self.canny_high)
        # 反转：白底黑线
        lineart = 255 - edges
        # 转回 RGB
        lineart_rgb = cv2.cvtColor(lineart, cv2.COLOR_GRAY2RGB)
        return Image.fromarray(lineart_rgb)
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, index):
        filename = self.file_list[index]
        image_path = os.path.join(self.images_path, filename)
        
        # 加载 ground_truth
        gt_img = Image.open(image_path).convert('RGB')
        
        # 提取线稿
        sketch_img = self._extract_lineart(gt_img)
        
        # 生成参考图（轻微变形）
        if self.tps_scale > 0:
            warped_img = warp_image(gt_img, distortion_scale=self.tps_scale)
        else:
            warped_img = gt_img
        
        # 随机翻转
        if self.flip and self.stage == 'train' and random.random() > 0.5:
            sketch_img = TF.hflip(sketch_img)
            gt_img = TF.hflip(gt_img)
            warped_img = TF.hflip(warped_img)
        
        # 应用变换
        sketch_tensor = self.transform(sketch_img)
        gt_tensor = self.transform(gt_img)
        warped_tensor = self.transform(warped_img)
        
        return {
            'L': sketch_tensor,
            'I_warped': warped_tensor,
            'I_gt': gt_tensor,
            'filename': filename
        }
    
    def set_tps_scale(self, scale: float):
        """动态更新 TPS 变形强度"""
        self.tps_scale = scale
        print(f"[SelfRef Dataset] TPS scale updated to: {scale}")