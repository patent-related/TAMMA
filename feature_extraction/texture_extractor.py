import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
from sklearn.feature_extraction import image as skimage

# 首先初始化logger
logger = logging.getLogger(__name__)

# 尝试导入cupy，用于GPU加速
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    logger.warning("未找到cupy库，GPU加速功能不可用")
    CUPY_AVAILABLE = False

class TextureFeatureExtractor:
    """
    纹理特征提取器
    
    支持多种纹理特征类型：
    - LBP (Local Binary Pattern)
    - GLCM (Gray-Level Co-occurrence Matrix)
    - Gabor 滤波器
    - HOG (Histogram of Oriented Gradients)
    - 多种特征组合
    """
    
    def __init__(self, 
                 feature_types: List[str] = ['lbp', 'glcm', 'gabor', 'hog'],
                 lbp_radius: int = 1,
                 lbp_n_points: int = 8,
                 glcm_distances: List[int] = [1],
                 glcm_angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4],
                 gabor_scales: List[int] = [1, 2, 4, 8],
                 gabor_orientations: List[int] = [0, 45, 90, 135],
                 hog_cell_size: Tuple[int, int] = (8, 8),
                 hog_block_size: Tuple[int, int] = (2, 2),
                 hog_orientations: int = 9,
                 spatial_pyramid_levels: int = 2,
                 normalize: bool = True,
                 use_gpu: bool = False,
                 gpu_device: int = 0):
        """
        Args:
            feature_types: 使用的特征类型列表
            lbp_radius: LBP半径
            lbp_n_points: LBP采样点数量
            glcm_distances: GLCM距离列表
            glcm_angles: GLCM角度列表
            gabor_scales: Gabor滤波器尺度列表
            gabor_orientations: Gabor滤波器方向列表
            hog_cell_size: HOG单元格大小
            hog_block_size: HOG块大小
            hog_orientations: HOG方向数量
            spatial_pyramid_levels: 空间金字塔层级
            normalize: 是否归一化特征
        """
        # 验证特征类型
        valid_types = ['lbp', 'glcm', 'gabor', 'hog']
        for ft in feature_types:
            if ft not in valid_types:
                raise ValueError(f"不支持的特征类型: {ft}. 有效类型: {valid_types}")
        
        self.feature_types = feature_types
        self.lbp_radius = lbp_radius
        self.lbp_n_points = lbp_n_points
        self.glcm_distances = glcm_distances
        self.glcm_angles = glcm_angles
        self.gabor_scales = gabor_scales
        self.gabor_orientations = gabor_orientations
        self.hog_cell_size = hog_cell_size
        self.hog_block_size = hog_block_size
        self.hog_orientations = hog_orientations
        self.spatial_pyramid_levels = spatial_pyramid_levels
        self.normalize = normalize
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        self.gpu_device = gpu_device
        
        if self.use_gpu:
            logger.info(f"启用GPU加速的纹理特征提取器（设备：{gpu_device}）")
        else:
            if use_gpu and not CUPY_AVAILABLE:
                logger.warning("cupy库不可用，回退到CPU模式")
            logger.info("使用CPU模式的纹理特征提取器")
        
        # 初始化Gabor滤波器
        self.gabor_kernels = self._create_gabor_kernels()
        
        logger.info(f"初始化纹理特征提取器: {feature_types}, "
                   f"空间金字塔: {spatial_pyramid_levels}")
    
    def _create_gabor_kernels(self) -> List[cv2.Mat]:
        """创建Gabor滤波器核"""
        kernels = []
        for scale in self.gabor_scales:
            for theta in self.gabor_orientations:
                theta_rad = np.deg2rad(theta)
                kernel = cv2.getGaborKernel(
                    ksize=(scale*4+1, scale*4+1),  # 核大小
                    sigma=scale*2,                # 标准差
                    theta=theta_rad,              # 方向
                    lambd=scale*3,                # 波长
                    gamma=0.5,                    # 空间纵横比
                    psi=0,                        # 相位偏移
                    ktype=cv2.CV_32F              # 核类型
                )
                kernels.append(kernel)
        return kernels
    
    def extract(self, image) -> np.ndarray:
        """
        提取纹理特征
        
        Args:
            image: BGR格式图像
            
        Returns:
            feature: 组合纹理特征向量
        """
        # 确保输入是numpy数组，而不是cupy数组
        if CUPY_AVAILABLE and isinstance(image, cp.ndarray):
            image = cp.asnumpy(image)
        
        # 确保输入是有效的numpy数组
        if not isinstance(image, np.ndarray):
            raise TypeError(f"期望numpy数组，得到: {type(image)}")
        
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 空间金字塔特征
        all_features = []
        
        for level in range(self.spatial_pyramid_levels + 1):
            num_blocks = 2 ** level
            block_size = (gray.shape[0] // num_blocks, gray.shape[1] // num_blocks)
            
            level_features = []
            
            for i in range(num_blocks):
                for j in range(num_blocks):
                    # 计算块的坐标
                    y_start = i * block_size[0]
                    y_end = (i + 1) * block_size[0] if i < num_blocks - 1 else gray.shape[0]
                    x_start = j * block_size[1]
                    x_end = (j + 1) * block_size[1] if j < num_blocks - 1 else gray.shape[1]
                    
                    # 提取块
                    block = gray[y_start:y_end, x_start:x_end]
                    
                    # 计算块的特征
                    block_feature = self._extract_block_features(block)
                    level_features.append(block_feature)
            
            # 聚合该层级的特征
            if level_features:
                level_features = np.concatenate(level_features)
                
                # 金字塔权重
                if level == 0:
                    weight = 1.0 / 4
                elif level == 1:
                    weight = 1.0 / 4
                else:
                    weight = 1.0 / (2 ** (level + 1))
                
                all_features.append(weight * level_features)
        
        # 组合所有层级特征
        if all_features:
            combined_feature = np.concatenate(all_features)
        else:
            combined_feature = np.array([])
        
        # 归一化
        if self.normalize and len(combined_feature) > 0:
            norm = np.linalg.norm(combined_feature)
            if norm > 0:
                combined_feature /= norm
        
        return combined_feature
    
    def _extract_block_features(self, block: np.ndarray) -> np.ndarray:
        """
        提取单个块的特征
        """
        features = []
        
        # LBP特征
        if 'lbp' in self.feature_types:
            lbp_feature = self._extract_lbp(block)
            features.append(lbp_feature)
        
        # GLCM特征
        if 'glcm' in self.feature_types:
            glcm_feature = self._extract_glcm(block)
            features.append(glcm_feature)
        
        # Gabor特征
        if 'gabor' in self.feature_types:
            gabor_feature = self._extract_gabor(block)
            features.append(gabor_feature)
        
        # HOG特征
        if 'hog' in self.feature_types:
            hog_feature = self._extract_hog(block)
            features.append(hog_feature)
        
        # 组合特征
        if features:
            return np.concatenate(features)
        else:
            return np.array([])
    
    def _extract_lbp(self, block: np.ndarray) -> np.ndarray:
        """
        提取LBP特征
        """
        try:
            # 计算LBP
            lbp = self._compute_lbp(block, self.lbp_radius, self.lbp_n_points)
            
            # 计算直方图
            hist, _ = np.histogram(lbp.ravel(), 
                                 bins=np.arange(0, self.lbp_n_points + 3),
                                 range=(0, self.lbp_n_points + 2))
            
            # 归一化
            hist = hist / (np.sum(hist) + 1e-7)
            
            return hist
        except Exception as e:
            logger.warning(f"LBP特征提取失败: {e}")
            return np.zeros(256)  # 返回默认零向量
    
    def _compute_lbp(self, image: np.ndarray, radius: int, n_points: int) -> np.ndarray:
        """
        计算LBP图像
        """
        # 如果启用GPU并且图像足够大，使用GPU加速
        if self.use_gpu and image.size > 10000:  # 只有当图像足够大时才使用GPU
            return self._compute_lbp_gpu(image, radius, n_points)
        else:
            return self._compute_lbp_cpu(image, radius, n_points)
    
    def _compute_lbp_cpu(self, image: np.ndarray, radius: int, n_points: int) -> np.ndarray:
        """
        CPU版本的LBP计算
        """
        height, width = image.shape
        lbp = np.zeros((height - 2*radius, width - 2*radius), dtype=np.uint8)
        
        for i in range(radius, height - radius):
            for j in range(radius, width - radius):
                center = image[i, j]
                code = 0
                
                for k in range(n_points):
                    # 计算采样点坐标
                    theta = 2 * np.pi * k / n_points
                    x = int(i + radius * np.sin(theta))
                    y = int(j + radius * np.cos(theta))
                    
                    # 比较并计算二进制码
                    if image[x, y] >= center:
                        code |= (1 << k)
                
                lbp[i - radius, j - radius] = code
        
        return lbp
    
    def _compute_lbp_gpu(self, image: np.ndarray, radius: int, n_points: int) -> np.ndarray:
        """
        GPU版本的LBP计算，使用cupy加速
        """
        try:
            # 将图像转移到GPU
            gpu_image = cp.array(image)
            height, width = gpu_image.shape
            
            # 创建结果数组
            gpu_lbp = cp.zeros((height - 2*radius, width - 2*radius), dtype=cp.uint8)
            
            # 预计算采样点的偏移
            thetas = cp.array([2 * cp.pi * k / n_points for k in range(n_points)])
            dx = (radius * cp.sin(thetas)).astype(cp.int32)
            dy = (radius * cp.cos(thetas)).astype(cp.int32)
            
            # 为每个像素计算LBP
            for i in range(radius, height - radius):
                for j in range(radius, width - radius):
                    center = gpu_image[i, j]
                    code = 0
                    
                    # 使用向量化操作处理采样点
                    x_coords = i + dx
                    y_coords = j + dy
                    
                    # 获取采样点的值
                    sample_values = gpu_image[x_coords, y_coords]
                    
                    # 比较并计算二进制码
                    for k in range(n_points):
                        if sample_values[k] >= center:
                            code |= (1 << k)
                    
                    gpu_lbp[i - radius, j - radius] = code
            
            # 将结果转回CPU
            return cp.asnumpy(gpu_lbp)
        except Exception as e:
            logger.error(f"GPU LBP计算失败: {e}，回退到CPU计算")
            # 回退到CPU计算
            return self._compute_lbp_cpu(image, radius, n_points)
    
    def _extract_glcm(self, block: np.ndarray) -> np.ndarray:
        """
        提取GLCM特征
        """
        try:
            # 确保灰度级别在0-255范围内
            if block.max() > 0:
                block_normalized = (block / block.max() * 255).astype(np.uint8)
            else:
                block_normalized = block.astype(np.uint8)
            
            # 计算GLCM矩阵
            from skimage.feature import graycomatrix, graycoprops
            
            glcm = graycomatrix(
                block_normalized,
                distances=self.glcm_distances,
                angles=self.glcm_angles,
                symmetric=True,
                normed=True
            )
            
            # 提取GLCM属性
            properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
            features = []
            
            for prop in properties:
                prop_values = graycoprops(glcm, prop).ravel()
                features.extend(prop_values)
            
            return np.array(features)
        except Exception as e:
            logger.warning(f"GLCM特征提取失败: {e}")
            # 返回默认零向量
            return np.zeros(len(self.glcm_distances) * len(self.glcm_angles) * 5)
    
    def _extract_gabor(self, block: np.ndarray) -> np.ndarray:
        """
        提取Gabor特征
        """
        features = []
        
        for kernel in self.gabor_kernels:
            # 应用Gabor滤波器
            filtered = cv2.filter2D(block.astype(np.float32), cv2.CV_32F, kernel)
            
            # 计算统计特征
            mean_val = np.mean(filtered)
            var_val = np.var(filtered)
            features.extend([mean_val, var_val])
        
        return np.array(features)
    
    def _extract_hog(self, block: np.ndarray) -> np.ndarray:
        """
        提取HOG特征
        """
        try:
            # 确保块大小足够大
            if block.shape[0] < self.hog_cell_size[0] * self.hog_block_size[0] or \
               block.shape[1] < self.hog_cell_size[1] * self.hog_block_size[1]:
                # 如果块太小，进行缩放
                scale = max(
                    (self.hog_cell_size[0] * self.hog_block_size[0]) / block.shape[0],
                    (self.hog_cell_size[1] * self.hog_block_size[1]) / block.shape[1]
                )
                new_size = (int(block.shape[1] * scale), int(block.shape[0] * scale))
                block = cv2.resize(block, new_size)
            
            # 使用skimage的HOG实现
            from skimage.feature import hog
            
            features = hog(
                block,
                orientations=self.hog_orientations,
                pixels_per_cell=self.hog_cell_size,
                cells_per_block=self.hog_block_size,
                block_norm='L2-Hys',
                feature_vector=True
            )
            
            return features
        except Exception as e:
            logger.warning(f"HOG特征提取失败: {e}")
            return np.zeros(128)  # 返回默认零向量
    
    def compute_similarity(self, 
                          feature1: np.ndarray, 
                          feature2: np.ndarray, 
                          method: str = 'cosine') -> float:
        """
        计算两个纹理特征之间的相似度
        
        Args:
            feature1: 特征向量1
            feature2: 特征向量2
            method: 相似度计算方法 ('cosine', 'l2', 'chi2', 'bhattacharyya')
            
        Returns:
            similarity: 相似度得分 (0-1)
        """
        # 根据是否使用GPU选择不同的计算方法
        if self.use_gpu:
            return self._compute_similarity_gpu(feature1, feature2, method)
        else:
            return self._compute_similarity_cpu(feature1, feature2, method)
    
    def _compute_similarity_cpu(self, 
                               feature1: np.ndarray, 
                               feature2: np.ndarray, 
                               method: str = 'cosine') -> float:
        """
        CPU版本的相似度计算
        """
        method = method.lower()
        
        # 处理特征长度不匹配的情况
        if len(feature1) != len(feature2):
            logger.warning(f"特征长度不匹配: {len(feature1)} vs {len(feature2)}")
            # 使用较长的特征长度
            max_len = max(len(feature1), len(feature2))
            padded1 = np.zeros(max_len)
            padded2 = np.zeros(max_len)
            padded1[:len(feature1)] = feature1
            padded2[:len(feature2)] = feature2
            feature1, feature2 = padded1, padded2
        
        if method == 'cosine':
            # 余弦相似度
            dot_product = np.dot(feature1, feature2)
            norm_product = np.linalg.norm(feature1) * np.linalg.norm(feature2) + 1e-7
            similarity = dot_product / norm_product
        
        elif method == 'l2':
            # L2距离 -> 相似度
            l2 = np.linalg.norm(feature1 - feature2)
            # 使用指数衰减将距离转换为相似度
            similarity = np.exp(-l2)
        
        elif method == 'chi2':
            # 卡方距离 -> 相似度
            numerator = (feature1 - feature2) ** 2
            denominator = feature1 + feature2 + 1e-7
            chi2 = np.sum(numerator / denominator)
            similarity = np.exp(-chi2)
        
        elif method == 'bhattacharyya':
            # Bhattacharyya距离 -> 相似度
            # 确保特征为非负数
            feature1 = np.maximum(0, feature1)
            feature2 = np.maximum(0, feature2)
            
            # 归一化
            sum1 = np.sum(feature1)
            sum2 = np.sum(feature2)
            
            if sum1 > 0:
                feature1 = feature1 / sum1
            if sum2 > 0:
                feature2 = feature2 / sum2
            
            # 计算Bhattacharyya距离
            bc = np.sum(np.sqrt(feature1 * feature2))
            bc = np.maximum(0, np.minimum(bc, 1.0))  # 确保在0-1范围内
            similarity = bc
        
        else:
            raise ValueError(f"不支持的相似度方法: {method}")
        
        # 确保在0-1范围内
        similarity = max(0.0, min(1.0, similarity))
        
        return similarity
    
    def _compute_similarity_gpu(self, 
                               feature1: np.ndarray, 
                               feature2: np.ndarray, 
                               method: str = 'cosine') -> float:
        """
        GPU版本的相似度计算，使用cupy加速
        """
        try:
            method = method.lower()
            
            # 处理特征长度不匹配的情况
            if len(feature1) != len(feature2):
                logger.warning(f"特征长度不匹配: {len(feature1)} vs {len(feature2)}")
                # 使用较长的特征长度
                max_len = max(len(feature1), len(feature2))
                padded1 = cp.zeros(max_len)
                padded2 = cp.zeros(max_len)
                padded1[:len(feature1)] = cp.array(feature1)
                padded2[:len(feature2)] = cp.array(feature2)
                f1_gpu = padded1
                f2_gpu = padded2
            else:
                # 转移数据到GPU
                f1_gpu = cp.array(feature1)
                f2_gpu = cp.array(feature2)
            
            if method == 'cosine':
                # 余弦相似度
                dot_product = cp.dot(f1_gpu, f2_gpu)
                norm_product = cp.linalg.norm(f1_gpu) * cp.linalg.norm(f2_gpu) + 1e-7
                similarity = dot_product / norm_product
            
            elif method == 'l2':
                # L2距离 -> 相似度
                l2 = cp.linalg.norm(f1_gpu - f2_gpu)
                # 使用指数衰减将距离转换为相似度
                similarity = cp.exp(-l2)
            
            elif method == 'chi2':
                # 卡方距离 -> 相似度
                numerator = (f1_gpu - f2_gpu) ** 2
                denominator = f1_gpu + f2_gpu + 1e-7
                chi2 = cp.sum(numerator / denominator)
                similarity = cp.exp(-chi2)
            
            elif method == 'bhattacharyya':
                # Bhattacharyya距离 -> 相似度
                # 确保特征为非负数
                f1_gpu = cp.maximum(0, f1_gpu)
                f2_gpu = cp.maximum(0, f2_gpu)
                
                # 归一化
                sum1 = cp.sum(f1_gpu)
                sum2 = cp.sum(f2_gpu)
                
                if sum1 > 0:
                    f1_gpu = f1_gpu / sum1
                if sum2 > 0:
                    f2_gpu = f2_gpu / sum2
                
                # 计算Bhattacharyya距离
                bc = cp.sum(cp.sqrt(f1_gpu * f2_gpu))
                bc = cp.maximum(0, cp.minimum(bc, 1.0))  # 确保在0-1范围内
                similarity = bc
            
            else:
                raise ValueError(f"不支持的相似度方法: {method}")
            
            # 将结果转回CPU并确保在0-1范围内
            similarity_cpu = float(similarity)
            similarity_cpu = max(0.0, min(1.0, similarity_cpu))
            
            return similarity_cpu
        except Exception as e:
            logger.error(f"GPU相似度计算失败: {e}，回退到CPU计算")
            # 回退到CPU计算
            return self._compute_similarity_cpu(feature1, feature2, method)