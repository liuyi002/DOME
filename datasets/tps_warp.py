import math
import random
import numpy as np
import cv2
from PIL import Image
import datasets.thinplate as tps


# TPS 课程学习强度配置（对应四阶段训练）
TPS_CURRICULUM_STAGES = {
    1: 0.05,   # 阶段1: 轻微变形，模拟自参考
    2: 0.15,   # 阶段2: 中等变形，引入对齐挑战
    3: 0.30,   # 阶段3: 较强变形，增强鲁棒性
}


def warp_image(image, distortion_scale=0.25):
    """
    对图像应用 TPS (Thin Plate Spline) 变形
    
    :param[in]  image             type: PIL Image
    :param[in]  distortion_scale  变形强度，范围 [0, 1]，默认 0.25
                                  - 0.05: 轻微变形（阶段1）
                                  - 0.15: 中等变形（阶段2）
                                  - 0.30: 较强变形（阶段3）
    :return     distorted image   type: PIL Image
    """
    p1, r1, p2, r2 = 0.0, 0.0, 0.0, 0.0
    while math.isclose(p1 + r1, p2 + r2):
        p1 = round(random.uniform(0.3, 0.7), 2)
        p2 = round(random.uniform(0.3, 0.7), 2)
        # 使用 distortion_scale 控制变形强度
        r1 = round(random.uniform(-distortion_scale, distortion_scale), 2)
        r2 = round(random.uniform(-distortion_scale, distortion_scale), 2)

    c_src = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [p1, p1], [p2, p2]])
    c_dst = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [p1 + r1, p1 + r1], [p2 + r2, p2 + r2]])
    arr = np.array(image)
    theta = tps.tps_theta_from_points(c_src, c_dst, reduced=True)
    grids = tps.tps_grid(theta, c_dst, arr.shape)
    mapx, mapy = tps.tps_grid_to_remap(grids, arr.shape)
    warped_arr = cv2.remap(arr, mapx, mapy, cv2.INTER_CUBIC)
    return Image.fromarray(warped_arr)


def get_tps_scale_for_stage(stage: int) -> float:
    """
    根据训练阶段获取对应的 TPS 变形强度
    
    :param stage: 训练阶段 (1, 2, 3)
    :return: 变形强度
    """
    return TPS_CURRICULUM_STAGES.get(stage, 0.25)