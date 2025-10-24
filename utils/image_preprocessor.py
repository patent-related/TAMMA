import os
import numpy as np
import cv2
import random
from typing import List, Dict, Tuple, Optional, Any, Callable
import logging
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ImagePreprocessor:
    """
    图像预处理工具类
    
    提供图像加载、调整大小、增强、变换等功能
    """
    
    @staticmethod
    def load_image(image_path: str,
                  target_size: Optional[Tuple[int, int]] = None,
                  color_mode: str = 'rgb',
                  normalize: bool = False) -> Optional[np.ndarray]:
        """
        加载图像
        
        Args:
            image_path: 图像路径
            target_size: 目标大小 (width, height)
            color_mode: 颜色模式 (rgb/bgr/grayscale)
            normalize: 是否归一化到[0, 1]
            
        Returns:
            图像数组
        """
        try:
            # 使用PIL加载图像
            with Image.open(image_path) as img:
                # 转换颜色模式
                if color_mode == 'rgb':
                    img = img.convert('RGB')
                elif color_mode == 'bgr':
                    img = img.convert('RGB')
                    # 后续会使用cv2的cvtColor转换
                elif color_mode == 'grayscale':
                    img = img.convert('L')
                else:
                    raise ValueError(f"不支持的颜色模式: {color_mode}")
                
                # 调整大小
                if target_size:
                    img = img.resize(target_size, Image.LANCZOS)
                
                # 转换为numpy数组
                image_array = np.array(img)
                
                # 如果是BGR模式，转换
                if color_mode == 'bgr' and len(image_array.shape) == 3:
                    image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
                
                # 归一化
                if normalize:
                    image_array = image_array.astype(np.float32) / 255.0
            
            return image_array
            
        except Exception as e:
            logger.error(f"加载图像失败 {image_path}: {e}")
            return None
    
    @staticmethod
    def load_images(image_paths: List[str],
                   target_size: Optional[Tuple[int, int]] = None,
                   color_mode: str = 'rgb',
                   normalize: bool = False) -> List[np.ndarray]:
        """
        批量加载图像
        
        Args:
            image_paths: 图像路径列表
            target_size: 目标大小
            color_mode: 颜色模式
            normalize: 是否归一化
            
        Returns:
            图像数组列表
        """
        images = []
        
        for i, image_path in enumerate(image_paths):
            img = ImagePreprocessor.load_image(image_path, target_size, color_mode, normalize)
            if img is not None:
                images.append(img)
            else:
                # 如果加载失败，创建一个空白图像
                h, w = target_size if target_size else (256, 256)
                channels = 3 if color_mode != 'grayscale' else 1
                blank_img = np.zeros((h, w, channels), dtype=np.uint8)
                images.append(blank_img)
            
            if (i + 1) % 100 == 0:
                logger.info(f"已加载 {i + 1}/{len(image_paths)} 张图像")
        
        return images
    
    @staticmethod
    def resize_image(image: np.ndarray,
                    target_size: Tuple[int, int],
                    keep_aspect_ratio: bool = False,
                    pad: bool = False) -> np.ndarray:
        """
        调整图像大小
        
        Args:
            image: 图像数组
            target_size: 目标大小 (width, height)
            keep_aspect_ratio: 是否保持长宽比
            pad: 是否填充到目标大小
            
        Returns:
            调整大小后的图像
        """
        try:
            h, w = image.shape[:2]
            target_w, target_h = target_size
            
            if keep_aspect_ratio:
                # 计算缩放比例
                scale = min(target_w / w, target_h / h)
                new_w = int(w * scale)
                new_h = int(h * scale)
                
                # 缩放图像
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                
                if pad:
                    # 创建目标大小的空白图像
                    result = np.zeros((target_h, target_w, *image.shape[2:]), dtype=image.dtype)
                    
                    # 计算填充位置
                    top = (target_h - new_h) // 2
                    left = (target_w - new_w) // 2
                    
                    # 放置缩放后的图像
                    result[top:top + new_h, left:left + new_w] = resized
                    return result
                else:
                    return resized
            else:
                # 直接调整大小
                return cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
                
        except Exception as e:
            logger.error(f"调整图像大小失败: {e}")
            return image
    
    @staticmethod
    def normalize_image(image: np.ndarray,
                       mean: Optional[List[float]] = None,
                       std: Optional[List[float]] = None,
                       min_max: bool = False) -> np.ndarray:
        """
        归一化图像
        
        Args:
            image: 图像数组
            mean: 均值
            std: 标准差
            min_max: 是否进行Min-Max归一化
            
        Returns:
            归一化后的图像
        """
        try:
            # 转换为float32
            img = image.astype(np.float32)
            
            if min_max:
                # Min-Max归一化到[0, 1]
                min_val = img.min()
                max_val = img.max()
                if max_val > min_val:
                    img = (img - min_val) / (max_val - min_val)
            else:
                # 归一化到[0, 1]
                img = img / 255.0
                
                # 减去均值，除以标准差
                if mean is not None:
                    mean = np.array(mean, dtype=np.float32)
                    if len(img.shape) == 3 and len(mean) == img.shape[2]:
                        img = img - mean
                    else:
                        img = img - mean[0]
                
                if std is not None:
                    std = np.array(std, dtype=np.float32)
                    if len(img.shape) == 3 and len(std) == img.shape[2]:
                        img = img / std
                    else:
                        img = img / std[0]
            
            return img
            
        except Exception as e:
            logger.error(f"归一化图像失败: {e}")
            return image
    
    @staticmethod
    def augment_image(image: np.ndarray,
                     augmentation_config: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        增强图像
        
        Args:
            image: 图像数组
            augmentation_config: 增强配置
            
        Returns:
            增强后的图像
        """
        try:
            # 默认配置
            default_config = {
                'brightness': 0.2,
                'contrast': 0.2,
                'saturation': 0.2,
                'hue': 0.1,
                'rotation': 15,
                'scale': 0.2,
                'shear': 10,
                'horizontal_flip': True,
                'vertical_flip': False,
                'blur': 0.5,
                'noise_std': 0.02
            }
            
            config = default_config.copy()
            if augmentation_config:
                config.update(augmentation_config)
            
            # 转换为PIL图像
            if len(image.shape) == 3 and image.shape[2] == 3:
                img = Image.fromarray(image, mode='RGB')
            elif len(image.shape) == 3 and image.shape[2] == 1:
                img = Image.fromarray(image.squeeze(), mode='L')
            else:
                img = Image.fromarray(image, mode='L')
            
            # 亮度调整
            if random.random() < 0.5:
                enhancer = ImageEnhance.Brightness(img)
                img = enhancer.enhance(1 + random.uniform(-config['brightness'], config['brightness']))
            
            # 对比度调整
            if random.random() < 0.5:
                enhancer = ImageEnhance.Contrast(img)
                img = enhancer.enhance(1 + random.uniform(-config['contrast'], config['contrast']))
            
            # 饱和度调整（仅对彩色图像）
            if random.random() < 0.5 and img.mode == 'RGB':
                enhancer = ImageEnhance.Color(img)
                img = enhancer.enhance(1 + random.uniform(-config['saturation'], config['saturation']))
            
            # 旋转
            if random.random() < 0.5:
                angle = random.uniform(-config['rotation'], config['rotation'])
                img = img.rotate(angle, fillcolor=0)
            
            # 保存原始尺寸
            original_w, original_h = img.size
            
            # 缩放
            if random.random() < 0.5:
                scale_factor = 1 + random.uniform(-config['scale'], config['scale'])
                new_size = tuple(int(dim * scale_factor) for dim in img.size)
                img = img.resize(new_size, Image.LANCZOS)
                # 随机裁剪回原大小
                if scale_factor > 1:
                    w, h = img.size
                    x = random.randint(0, max(0, w - original_w))
                    y = random.randint(0, max(0, h - original_h))
                    img = img.crop((x, y, x + original_w, y + original_h))
            
            # 水平翻转
            if config['horizontal_flip'] and random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            
            # 垂直翻转
            if config['vertical_flip'] and random.random() < 0.5:
                img = img.transpose(Image.FLIP_TOP_BOTTOM)
            
            # 模糊
            if random.random() < 0.3:
                blur_radius = random.uniform(0.1, config['blur'])
                img = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
            
            # 转换回numpy数组
            augmented = np.array(img)
            
            # 添加高斯噪声
            if random.random() < 0.3 and config['noise_std'] > 0:
                noise = np.random.normal(0, config['noise_std'] * 255, augmented.shape)
                augmented = np.clip(augmented + noise, 0, 255).astype(np.uint8)
            
            return augmented
            
        except Exception as e:
            logger.error(f"增强图像失败: {e}")
            return image
    
    @staticmethod
    def create_augmentations_pipeline(config: Optional[Dict[str, Any]] = None) -> A.Compose:
        """
        创建增强管道（使用albumentations）
        
        Args:
            config: 增强配置
            
        Returns:
            增强管道
        """
        try:
            # 默认配置
            default_config = {
                'enable_rotation': True,
                'rotation_limit': 15,
                'enable_scale': True,
                'scale_limit': 0.2,
                'enable_shift': True,
                'shift_limit': 0.1,
                'enable_shear': True,
                'shear_limit': 10,
                'enable_horizontal_flip': True,
                'enable_vertical_flip': False,
                'enable_brightness_contrast': True,
                'brightness_limit': 0.2,
                'contrast_limit': 0.2,
                'enable_hue_saturation': True,
                'hue_limit': 10,
                'saturation_limit': 0.2,
                'enable_blur': True,
                'blur_limit': 3,
                'enable_noise': True,
                'noise_std': 0.02,
                'p': 0.5
            }
            
            config = default_config.copy()
            if config:
                config.update(config)
            
            # 创建变换列表
            transforms = []
            
            # 几何变换
            geo_transforms = []
            
            if config['enable_rotation']:
                geo_transforms.append(A.Rotate(limit=config['rotation_limit'], p=0.7))
            
            if config['enable_scale'] or config['enable_shift'] or config['enable_shear']:
                geo_transforms.append(A.ShiftScaleRotate(
                    shift_limit=config['shift_limit'] if config['enable_shift'] else 0,
                    scale_limit=config['scale_limit'] if config['enable_scale'] else 0,
                    rotate_limit=0,
                    shear_limit=config['shear_limit'] if config['enable_shear'] else 0,
                    p=0.7
                ))
            
            if geo_transforms:
                transforms.append(A.OneOf(geo_transforms, p=0.7))
            
            # 翻转
            if config['enable_horizontal_flip']:
                transforms.append(A.HorizontalFlip(p=0.5))
            
            if config['enable_vertical_flip']:
                transforms.append(A.VerticalFlip(p=0.3))
            
            # 亮度对比度
            if config['enable_brightness_contrast']:
                transforms.append(A.RandomBrightnessContrast(
                    brightness_limit=config['brightness_limit'],
                    contrast_limit=config['contrast_limit'],
                    p=0.5
                ))
            
            # 色调饱和度
            if config['enable_hue_saturation']:
                transforms.append(A.HueSaturationValue(
                    hue_shift_limit=config['hue_limit'],
                    sat_shift_limit=config['saturation_limit'],
                    val_shift_limit=0,
                    p=0.5
                ))
            
            # 模糊
            if config['enable_blur']:
                transforms.append(A.OneOf([
                    A.GaussianBlur(blur_limit=config['blur_limit']),
                    A.MedianBlur(blur_limit=config['blur_limit'])
                ], p=0.3))
            
            # 噪声
            if config['enable_noise']:
                transforms.append(A.GaussNoise(var_limit=(10, 50), p=0.3))
            
            # 裁剪
            transforms.append(A.RandomCrop(height=224, width=224, p=0.2))
            
            # 确保尺寸一致
            transforms.append(A.Resize(height=224, width=224))
            
            # 创建管道
            pipeline = A.Compose(transforms, p=config['p'])
            
            return pipeline
            
        except Exception as e:
            logger.error(f"创建增强管道失败: {e}")
            # 返回最小管道
            return A.Compose([A.Resize(height=224, width=224)])
    
    @staticmethod
    def apply_augmentations_pipeline(image: np.ndarray,
                                    pipeline: A.Compose) -> np.ndarray:
        """
        应用增强管道
        
        Args:
            image: 图像数组
            pipeline: 增强管道
            
        Returns:
            增强后的图像
        """
        try:
            # 应用增强
            augmented = pipeline(image=image)['image']
            return augmented
            
        except Exception as e:
            logger.error(f"应用增强管道失败: {e}")
            return image
    
    @staticmethod
    def convert_color_space(image: np.ndarray,
                           src_space: str = 'rgb',
                           dst_space: str = 'hsv') -> np.ndarray:
        """
        转换颜色空间
        
        Args:
            image: 图像数组
            src_space: 源颜色空间
            dst_space: 目标颜色空间
            
        Returns:
            转换后的图像
        """
        try:
            # 构建转换键
            key = (src_space.lower(), dst_space.lower())
            
            # 定义转换映射
            conversion_map = {
                ('rgb', 'hsv'): cv2.COLOR_RGB2HSV,
                ('rgb', 'lab'): cv2.COLOR_RGB2LAB,
                ('rgb', 'gray'): cv2.COLOR_RGB2GRAY,
                ('rgb', 'ycrcb'): cv2.COLOR_RGB2YCrCb,
                ('hsv', 'rgb'): cv2.COLOR_HSV2RGB,
                ('hsv', 'lab'): cv2.COLOR_HSV2BGR,  # 需要中间步骤
                ('lab', 'rgb'): cv2.COLOR_LAB2RGB,
                ('lab', 'hsv'): cv2.COLOR_LAB2BGR,  # 需要中间步骤
                ('bgr', 'rgb'): cv2.COLOR_BGR2RGB,
                ('bgr', 'hsv'): cv2.COLOR_BGR2HSV,
                ('bgr', 'lab'): cv2.COLOR_BGR2LAB,
                ('bgr', 'gray'): cv2.COLOR_BGR2GRAY,
                ('gray', 'rgb'): cv2.COLOR_GRAY2RGB,
                ('gray', 'bgr'): cv2.COLOR_GRAY2BGR,
            }
            
            if key in conversion_map:
                return cv2.cvtColor(image, conversion_map[key])
            elif key[0] == key[1]:
                # 相同颜色空间，直接返回
                return image.copy()
            else:
                # 尝试通过RGB作为中间步骤
                if key[0] != 'rgb':
                    # 先转换到RGB
                    img_rgb = ImagePreprocessor.convert_color_space(image, key[0], 'rgb')
                    # 再转换到目标空间
                    return ImagePreprocessor.convert_color_space(img_rgb, 'rgb', key[1])
                else:
                    logger.warning(f"不支持的颜色空间转换: {src_space} -> {dst_space}")
                    return image.copy()
                    
        except Exception as e:
            logger.error(f"转换颜色空间失败: {e}")
            return image.copy()
    
    @staticmethod
    def denoise_image(image: np.ndarray,
                     method: str = 'gaussian',
                     params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        去噪图像
        
        Args:
            image: 图像数组
            method: 去噪方法 (gaussian/median/bilateral/nlmeans)
            params: 去噪参数
            
        Returns:
            去噪后的图像
        """
        try:
            default_params = {
                'gaussian': {'ksize': (5, 5), 'sigmaX': 1.5},
                'median': {'ksize': 5},
                'bilateral': {'d': 9, 'sigmaColor': 75, 'sigmaSpace': 75},
                'nlmeans': {'h': 5}
            }
            
            # 获取参数
            method_params = default_params.get(method, {})
            if params:
                method_params.update(params)
            
            if method == 'gaussian':
                return cv2.GaussianBlur(image, **method_params)
            elif method == 'median':
                return cv2.medianBlur(image, **method_params)
            elif method == 'bilateral':
                return cv2.bilateralFilter(image, **method_params)
            elif method == 'nlmeans':
                if len(image.shape) == 3:
                    return cv2.fastNlMeansDenoisingColored(image, h=method_params['h'])
                else:
                    return cv2.fastNlMeansDenoising(image, h=method_params['h'])
            else:
                logger.warning(f"不支持的去噪方法: {method}")
                return image.copy()
                
        except Exception as e:
            logger.error(f"去噪图像失败: {e}")
            return image.copy()
    
    @staticmethod
    def enhance_edge(image: np.ndarray,
                    method: str = 'canny',
                    params: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        边缘增强
        
        Args:
            image: 图像数组
            method: 边缘检测方法 (canny/sobel/laplacian)
            params: 边缘检测参数
            
        Returns:
            边缘图像
        """
        try:
            default_params = {
                'canny': {'threshold1': 100, 'threshold2': 200},
                'sobel': {'ksize': 3},
                'laplacian': {'ksize': 3}
            }
            
            # 获取参数
            method_params = default_params.get(method, {})
            if params:
                method_params.update(params)
            
            # 转换为灰度图（如果需要）
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            if method == 'canny':
                edges = cv2.Canny(gray, **method_params)
            elif method == 'sobel':
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, **method_params)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, **method_params)
                edges = np.sqrt(sobelx**2 + sobely**2)
                edges = np.uint8(edges / np.max(edges) * 255)
            elif method == 'laplacian':
                edges = cv2.Laplacian(gray, cv2.CV_64F, **method_params)
                edges = np.uint8(np.abs(edges) / np.max(np.abs(edges)) * 255)
            else:
                logger.warning(f"不支持的边缘检测方法: {method}")
                edges = gray
            
            return edges
            
        except Exception as e:
            logger.error(f"边缘增强失败: {e}")
            # 返回灰度图
            if len(image.shape) == 3:
                return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                return image.copy()
    
    @staticmethod
    def create_mask(image: np.ndarray,
                   threshold: int = 128,
                   invert: bool = False) -> np.ndarray:
        """
        创建二值掩码
        
        Args:
            image: 图像数组
            threshold: 阈值
            invert: 是否反转
            
        Returns:
            掩码
        """
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            
            # 二值化
            if invert:
                _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
            else:
                _, mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
            
            # 形态学操作
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            return mask
            
        except Exception as e:
            logger.error(f"创建掩码失败: {e}")
            return np.ones_like(image[:, :, 0]) * 255 if len(image.shape) == 3 else np.ones_like(image) * 255
    
    @staticmethod
    def apply_mask(image: np.ndarray,
                  mask: np.ndarray,
                  fill_color: Tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
        """
        应用掩码
        
        Args:
            image: 图像数组
            mask: 掩码
            fill_color: 填充颜色
            
        Returns:
            应用掩码后的图像
        """
        try:
            # 确保掩码是二值的
            if mask.max() > 1:
                mask = mask // 255
            
            # 扩展掩码维度
            if len(mask.shape) == 2 and len(image.shape) == 3:
                mask = np.expand_dims(mask, axis=2)
                mask = np.repeat(mask, 3, axis=2)
            
            # 创建填充图像
            fill_image = np.full_like(image, fill_color, dtype=image.dtype)
            
            # 应用掩码
            result = image * mask + fill_image * (1 - mask)
            
            return result.astype(image.dtype)
            
        except Exception as e:
            logger.error(f"应用掩码失败: {e}")
            return image.copy()
    
    @staticmethod
    def crop_center(image: np.ndarray,
                   crop_size: Tuple[int, int]) -> np.ndarray:
        """
        裁剪图像中心区域
        
        Args:
            image: 图像数组
            crop_size: 裁剪大小
            
        Returns:
            裁剪后的图像
        """
        try:
            h, w = image.shape[:2]
            crop_h, crop_w = crop_size
            
            # 计算裁剪位置
            start_h = (h - crop_h) // 2
            start_w = (w - crop_w) // 2
            end_h = start_h + crop_h
            end_w = start_w + crop_w
            
            # 确保不越界
            start_h = max(0, start_h)
            start_w = max(0, start_w)
            end_h = min(h, end_h)
            end_w = min(w, end_w)
            
            return image[start_h:end_h, start_w:end_w]
            
        except Exception as e:
            logger.error(f"裁剪图像中心失败: {e}")
            return image.copy()
    
    @staticmethod
    def equalize_histogram(image: np.ndarray,
                          adaptive: bool = False) -> np.ndarray:
        """
        直方图均衡化
        
        Args:
            image: 图像数组
            adaptive: 是否使用自适应直方图均衡化
            
        Returns:
            均衡化后的图像
        """
        try:
            # 转换为灰度图
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                is_color = True
            else:
                gray = image.copy()
                is_color = False
            
            if adaptive:
                # 自适应直方图均衡化
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                equalized = clahe.apply(gray)
            else:
                # 普通直方图均衡化
                equalized = cv2.equalizeHist(gray)
            
            # 如果是彩色图像，将均衡化后的通道放回
            if is_color:
                result = image.copy()
                # 转换为HSV空间，只均衡化V通道
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                hsv[:, :, 2] = equalized
                result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                return result
            else:
                return equalized
                
        except Exception as e:
            logger.error(f"直方图均衡化失败: {e}")
            return image.copy()
    
    @staticmethod
    def calculate_image_stats(image: np.ndarray) -> Dict[str, Any]:
        """
        计算图像统计信息
        
        Args:
            image: 图像数组
            
        Returns:
            统计信息字典
        """
        try:
            stats = {
                'shape': image.shape,
                'dtype': str(image.dtype),
                'min_value': float(np.min(image)),
                'max_value': float(np.max(image)),
                'mean_value': float(np.mean(image)),
                'std_value': float(np.std(image)),
            }
            
            # 如果是彩色图像，计算每个通道的统计信息
            if len(image.shape) == 3:
                channel_stats = []
                for i in range(image.shape[2]):
                    channel_stats.append({
                        'min': float(np.min(image[:, :, i])),
                        'max': float(np.max(image[:, :, i])),
                        'mean': float(np.mean(image[:, :, i])),
                        'std': float(np.std(image[:, :, i])),
                    })
                stats['channel_stats'] = channel_stats
            
            # 计算清晰度（拉普拉斯方差）
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image.copy()
            stats['sharpness'] = float(cv2.Laplacian(gray, cv2.CV_64F).var())
            
            # 计算亮度
            if len(image.shape) == 3:
                hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
                stats['brightness'] = float(np.mean(hsv[:, :, 2]))
            
            return stats
            
        except Exception as e:
            logger.error(f"计算图像统计信息失败: {e}")
            return {'error': str(e)}

# 示例用法
if __name__ == '__main__':
    # 示例：加载并预处理图像
    # image_path = 'example.jpg'
    # image = ImagePreprocessor.load_image(image_path, target_size=(256, 256))
    # 
    # # 应用增强
    # augmented = ImagePreprocessor.augment_image(image)
    # 
    # # 创建掩码
    # mask = ImagePreprocessor.create_mask(image)
    # masked_image = ImagePreprocessor.apply_mask(image, mask)
    # 
    # # 计算统计信息
    # stats = ImagePreprocessor.calculate_image_stats(image)
    # 
    logger.info("图像预处理工具类已加载")