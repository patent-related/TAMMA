import cv2
import numpy as np
from typing import Optional, Tuple
import logging

# 首先初始化logger
logger = logging.getLogger(__name__)

# 尝试导入cupy用于GPU加速
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    logger.warning("cupy库未安装，无法使用GPU加速颜色特征提取")

class ColorFeatureExtractor:
    """
    颜色特征提取器
    
    支持多种颜色空间和特征表示方法
    """
    
    def __init__(self, 
                 color_space: str = 'HSV',
                 h_bins: int = 32,
                 s_bins: int = 32,
                 v_bins: int = 32,
                 use_spatial_pyramid: bool = True,
                 pyramid_levels: int = 3,
                 use_gpu: bool = False):
        """
        Args:
            color_space: 颜色空间 ('HSV', 'RGB', 'LAB', 'YCrCb')
            h_bins: H通道bin数
            s_bins: S通道bin数
            v_bins: V通道bin数
            use_spatial_pyramid: 是否使用空间金字塔
            pyramid_levels: 金字塔层数
        """
        self.color_space = color_space.upper()
        self.h_bins = h_bins
        self.s_bins = s_bins
        self.v_bins = v_bins
        self.use_spatial_pyramid = use_spatial_pyramid
        self.pyramid_levels = pyramid_levels
        self.use_gpu = use_gpu and CUPY_AVAILABLE
        
        # 验证颜色空间
        valid_spaces = ['HSV', 'RGB', 'LAB', 'YCrCb']
        if self.color_space not in valid_spaces:
            raise ValueError(f"不支持的颜色空间: {color_space}. 支持的选项: {valid_spaces}")
        
        logger.info(f"初始化颜色特征提取器: {color_space}, "
                   f"bins=({h_bins}, {s_bins}, {v_bins}), "
                   f"pyramid={use_spatial_pyramid}({pyramid_levels}), "
                   f"GPU={'已启用' if self.use_gpu else '未启用'}")
    
    def extract(self, image: np.ndarray) -> np.ndarray:
        """
        提取颜色特征
        
        Args:
            image: BGR格式图像
            
        Returns:
            feature: 颜色特征向量
        """
        # 转换颜色空间
        converted = self._convert_color_space(image)
        
        if self.use_gpu:
            # GPU版本特征提取
            if self.use_spatial_pyramid:
                feature = self._extract_spatial_pyramid_gpu(converted)
            else:
                feature = self._extract_histogram_gpu(converted)
        else:
            # CPU版本特征提取
            if self.use_spatial_pyramid:
                feature = self._extract_spatial_pyramid(converted)
            else:
                feature = self._extract_histogram(converted)
        
        return feature
    
    def _convert_color_space(self, image) -> np.ndarray:
        """转换颜色空间"""
        # 确保输入是numpy数组，而不是cupy数组
        if CUPY_AVAILABLE and isinstance(image, cp.ndarray):
            image = cp.asnumpy(image)
        
        # 确保输入是有效的numpy数组
        if not isinstance(image, np.ndarray):
            raise TypeError(f"期望numpy数组，得到: {type(image)}")
        
        if self.color_space == 'HSV':
            return cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.color_space == 'RGB':
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.color_space == 'LAB':
            return cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        elif self.color_space == 'YCrCb':
            return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    
    def _extract_histogram(self, image: np.ndarray) -> np.ndarray:
        """提取单通道直方图"""
        # 设置通道范围
        if self.color_space == 'HSV':
            ranges = [0, 180, 0, 256, 0, 256]
        else:
            ranges = [0, 256, 0, 256, 0, 256]
        
        # 计算直方图
        hist = cv2.calcHist([image], [0, 1, 2], None, 
                          [self.h_bins, self.s_bins, self.v_bins], 
                          ranges)
        
        # 展平和归一化
        hist = hist.flatten()
        hist = hist / (np.sum(hist) + 1e-7)
        
        return hist
    
    def _extract_histogram_gpu(self, image: np.ndarray) -> np.ndarray:
        """使用GPU提取单通道直方图"""
        # 将图像传输到GPU
        img_gpu = cp.array(image)
        
        # 计算每个通道的直方图
        h_bins = self.h_bins
        s_bins = self.s_bins
        v_bins = self.v_bins
        
        # 设置通道范围
        if self.color_space == 'HSV':
            h_range = (0, 180)
        else:
            h_range = (0, 256)
        s_range = (0, 256)
        v_range = (0, 256)
        
        # 计算3D直方图 (使用分箱策略)
        h_indices = cp.clip(((img_gpu[:,:,0] - h_range[0]) * h_bins / (h_range[1] - h_range[0])).astype(cp.int32), 0, h_bins - 1)
        s_indices = cp.clip(((img_gpu[:,:,1] - s_range[0]) * s_bins / (s_range[1] - s_range[0])).astype(cp.int32), 0, s_bins - 1)
        v_indices = cp.clip(((img_gpu[:,:,2] - v_range[0]) * v_bins / (v_range[1] - v_range[0])).astype(cp.int32), 0, v_bins - 1)
        
        # 计算一维索引
        indices = h_indices * (s_bins * v_bins) + s_indices * v_bins + v_indices
        
        # 使用bincount计算直方图
        hist = cp.bincount(indices.flatten(), minlength=h_bins * s_bins * v_bins)
        
        # 归一化
        hist = hist / (cp.sum(hist) + 1e-7)
        
        # 转回CPU
        return cp.asnumpy(hist)
    
    def _extract_spatial_pyramid(self, image: np.ndarray) -> np.ndarray:
        """
        使用空间金字塔提取颜色特征
        
        实现空间金字塔匹配 (SPM) 算法
        """
        h, w = image.shape[:2]
        features = []
        
        # 权重计算 (根据论文)
        weights = [1.0 / (2 ** (2 * (self.pyramid_levels - l - 1))) 
                  for l in range(self.pyramid_levels)]
        
        for level in range(self.pyramid_levels):
            # 计算网格大小
            grid_size = 2 ** level
            cell_h = h // grid_size
            cell_w = w // grid_size
            
            level_features = []
            
            for i in range(grid_size):
                for j in range(grid_size):
                    # 计算单元格区域
                    start_h = i * cell_h
                    start_w = j * cell_w
                    end_h = start_h + cell_h if i < grid_size - 1 else h
                    end_w = start_w + cell_w if j < grid_size - 1 else w
                    
                    # 提取单元格
                    cell = image[start_h:end_h, start_w:end_w]
                    
                    # 计算单元格直方图
                    cell_hist = self._extract_histogram(cell)
                    level_features.append(cell_hist)
            
            # 合并当前层级的特征并应用权重
            level_feature = np.concatenate(level_features) * weights[level]
            features.append(level_feature)
        
        # 合并所有层级的特征
        final_feature = np.concatenate(features)
        final_feature = final_feature / (np.linalg.norm(final_feature) + 1e-7)
        
        return final_feature
    
    def _extract_spatial_pyramid_gpu(self, image: np.ndarray) -> np.ndarray:
        """
        使用GPU加速的空间金字塔提取颜色特征
        """
        h, w = image.shape[:2]
        
        # 将图像传输到GPU
        img_gpu = cp.array(image)
        
        # 权重计算 (根据论文)
        weights = [1.0 / (2 ** (2 * (self.pyramid_levels - l - 1))) 
                  for l in range(self.pyramid_levels)]
        
        # 预分配所有层级的特征空间
        total_bins = self.h_bins * self.s_bins * self.v_bins
        total_features = 0
        for level in range(self.pyramid_levels):
            grid_size = 2 ** level
            total_features += grid_size * grid_size * total_bins
        
        all_features_gpu = cp.zeros(total_features, dtype=cp.float32)
        
        start_idx = 0
        for level in range(self.pyramid_levels):
            # 计算网格大小
            grid_size = 2 ** level
            cell_h = h // grid_size
            cell_w = w // grid_size
            
            # 为当前层级分配特征空间
            level_feature_size = grid_size * grid_size * total_bins
            level_features_gpu = all_features_gpu[start_idx:start_idx + level_feature_size]
            
            cell_idx = 0
            for i in range(grid_size):
                for j in range(grid_size):
                    # 计算单元格区域
                    start_h = i * cell_h
                    start_w = j * cell_w
                    end_h = start_h + cell_h if i < grid_size - 1 else h
                    end_w = start_w + cell_w if j < grid_size - 1 else w
                    
                    # 提取单元格
                    cell_gpu = img_gpu[start_h:end_h, start_w:end_w]
                    
                    # 在GPU上计算单元格直方图
                    # 设置通道范围
                    if self.color_space == 'HSV':
                        h_range = (0, 180)
                    else:
                        h_range = (0, 256)
                    s_range = (0, 256)
                    v_range = (0, 256)
                    
                    # 计算3D直方图 (使用分箱策略)
                    h_indices = cp.clip(((cell_gpu[:,:,0] - h_range[0]) * self.h_bins / (h_range[1] - h_range[0])).astype(cp.int32), 0, self.h_bins - 1)
                    s_indices = cp.clip(((cell_gpu[:,:,1] - s_range[0]) * self.s_bins / (s_range[1] - s_range[0])).astype(cp.int32), 0, self.s_bins - 1)
                    v_indices = cp.clip(((cell_gpu[:,:,2] - v_range[0]) * self.v_bins / (v_range[1] - v_range[0])).astype(cp.int32), 0, self.v_bins - 1)
                    
                    # 计算一维索引
                    indices = h_indices * (self.s_bins * self.v_bins) + s_indices * self.v_bins + v_indices
                    
                    # 使用bincount计算直方图
                    cell_hist_gpu = cp.bincount(indices.flatten(), minlength=total_bins)
                    
                    # 归一化
                    cell_hist_gpu = cell_hist_gpu / (cp.sum(cell_hist_gpu) + 1e-7)
                    
                    # 存储到层级特征中
                    start = cell_idx * total_bins
                    end = start + total_bins
                    level_features_gpu[start:end] = cell_hist_gpu
                    cell_idx += 1
            
            # 应用权重
            level_features_gpu *= weights[level]
            start_idx += level_feature_size
        
        # 归一化最终特征
        all_features_gpu = all_features_gpu / (cp.linalg.norm(all_features_gpu) + 1e-7)
        
        # 转回CPU
        return cp.asnumpy(all_features_gpu)
    
    def compute_similarity(self, 
                          feature1: np.ndarray, 
                          feature2: np.ndarray, 
                          method: str = 'bhattacharyya') -> float:
        """
        计算两个颜色特征之间的相似度
        
        Args:
            feature1: 特征向量1
            feature2: 特征向量2
            method: 相似度计算方法 ('bhattacharyya', 'cosine', 'chi2', 'l1', 'l2')
            
        Returns:
            similarity: 相似度得分 (0-1)
        """
        if self.use_gpu:
            return self._compute_similarity_gpu(feature1, feature2, method)
        else:
            return self._compute_similarity_cpu(feature1, feature2, method)
    
    def _compute_similarity_cpu(self, 
                               feature1: np.ndarray, 
                               feature2: np.ndarray, 
                               method: str = 'bhattacharyya') -> float:
        """
        CPU版本的相似度计算
        """
        method = method.lower()
        
        if method == 'bhattacharyya':
            # 巴氏距离 -> 相似度
            bc = cv2.compareHist(feature1.astype(np.float32), 
                               feature2.astype(np.float32), 
                               cv2.HISTCMP_BHATTACHARYYA)
            # 转换为相似度 (0-1范围)
            similarity = np.exp(-bc * bc / 0.5)
        
        elif method == 'cosine':
            # 余弦相似度
            dot_product = np.dot(feature1, feature2)
            norm_product = np.linalg.norm(feature1) * np.linalg.norm(feature2) + 1e-7
            similarity = dot_product / norm_product
        
        elif method == 'chi2':
            # 卡方距离 -> 相似度
            chi2 = cv2.compareHist(feature1.astype(np.float32), 
                                 feature2.astype(np.float32), 
                                 cv2.HISTCMP_CHISQR)
            # 归一化并转换为相似度
            similarity = 1 / (1 + chi2)
        
        elif method == 'l1':
            # L1距离 -> 相似度
            l1 = np.sum(np.abs(feature1 - feature2))
            similarity = 1 / (1 + l1)
        
        elif method == 'l2':
            # L2距离 -> 相似度
            l2 = np.linalg.norm(feature1 - feature2)
            similarity = 1 / (1 + l2)
        
        else:
            raise ValueError(f"不支持的相似度方法: {method}")
        
        # 确保在0-1范围内
        similarity = max(0.0, min(1.0, similarity))
        
        return similarity
    
    def _compute_similarity_gpu(self, 
                               feature1: np.ndarray, 
                               feature2: np.ndarray, 
                               method: str = 'bhattacharyya') -> float:
        """
        GPU版本的相似度计算
        """
        method = method.lower()
        
        # 将特征传输到GPU
        f1_gpu = cp.array(feature1, dtype=cp.float32)
        f2_gpu = cp.array(feature2, dtype=cp.float32)
        
        if method == 'cosine':
            # 余弦相似度
            dot_product = cp.sum(f1_gpu * f2_gpu)
            norm_product = cp.linalg.norm(f1_gpu) * cp.linalg.norm(f2_gpu) + 1e-7
            similarity = float(dot_product / norm_product)
        
        elif method == 'l1':
            # L1距离 -> 相似度
            l1 = cp.sum(cp.abs(f1_gpu - f2_gpu))
            similarity = 1.0 / (1.0 + float(l1))
        
        elif method == 'l2':
            # L2距离 -> 相似度
            l2 = cp.linalg.norm(f1_gpu - f2_gpu)
            similarity = 1.0 / (1.0 + float(l2))
        
        elif method == 'chi2':
            # 卡方距离 -> 相似度
            diff = f1_gpu - f2_gpu
            sum_f = f1_gpu + f2_gpu
            # 避免除以零
            sum_f = cp.maximum(sum_f, 1e-7)
            chi2 = cp.sum((diff * diff) / sum_f)
            similarity = 1.0 / (1.0 + float(chi2))
        
        elif method == 'bhattacharyya':
            # 巴氏距离 -> 相似度
            # 使用numpy计算，因为cv2的GPU版本不可直接访问
            # 注意：这里仍需在CPU上执行，但使用更高效的实现
            bc = cv2.compareHist(feature1.astype(np.float32), 
                               feature2.astype(np.float32), 
                               cv2.HISTCMP_BHATTACHARYYA)
            # 转换为相似度 (0-1范围)
            similarity = np.exp(-bc * bc / 0.5)
        
        else:
            raise ValueError(f"不支持的相似度方法: {method}")
        
        # 确保在0-1范围内
        similarity = max(0.0, min(1.0, similarity))
        
        return similarity