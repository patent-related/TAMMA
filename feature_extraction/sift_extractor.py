import cv2
import numpy as np
from typing import List, Optional, Tuple
from sklearn.cluster import KMeans
import pickle
import logging
from pathlib import Path

# 初始化logger
logger = logging.getLogger(__name__)

# 尝试导入cupy，用于GPU加速
CUPY_AVAILABLE = False
CUML_AVAILABLE = False

try:
    import cupy as cp
    # 简单测试cupy是否正常工作
    _ = cp.array([1])
    CUPY_AVAILABLE = True
    logger.info(f"成功导入cupy，版本: {cp.__version__}")
except Exception as e:
    logger.warning(f"cupy导入失败: {str(e)}")

# 尝试导入cuml
try:
    from cuml.cluster import KMeans as cuKMeans
    CUML_AVAILABLE = True
    logger.info("成功导入cuml")
except ImportError:
    logger.warning("未找到cuml库，GPU K-means功能不可用，但仍可使用cupy进行其他GPU加速")

# 记录GPU加速状态
if CUPY_AVAILABLE:
    logger.info("GPU加速功能部分可用")
else:
    logger.warning("GPU加速功能不可用，将使用CPU模式")

class SIFTFeatureExtractor:
    """
    SIFT特征提取器
    
    支持多种特征编码方法：
    - BoVW (Bag of Visual Words)
    - VLAD (Vector of Locally Aggregated Descriptors)
    - Fisher Vector
    """
    
    def __init__(self, 
                 n_features: int = 500,
                 n_octave_layers: int = 3,
                 contrast_threshold: float = 0.04,
                 edge_threshold: float = 10,
                 sigma: float = 1.6,
                 encoding_method: str = 'bovw',
                 codebook_size: int = 512,
                 codebook_path: Optional[str] = None,
                 use_gpu: bool = False,
                 gpu_device: int = 0):
        """
        Args:
            n_features: 最大特征点数
            n_octave_layers: 每八度的层数
            contrast_threshold: 对比度阈值
            edge_threshold: 边缘阈值
            sigma: 初始高斯模糊的sigma值
            encoding_method: 编码方法 ('bovw', 'vlad', 'fisher')
            codebook_size: 码本大小
            codebook_path: 预训练码本路径
        """
        # SIFT参数
        self.n_features = n_features
        self.n_octave_layers = n_octave_layers
        self.contrast_threshold = contrast_threshold
        self.edge_threshold = edge_threshold
        self.sigma = sigma
        
        # 编码参数
        self.encoding_method = encoding_method.lower()
        self.codebook_size = codebook_size
        self.codebook = None
        self.codebook_path = codebook_path
        
        # GPU相关设置
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        
        # CUDA加速检查
        self.use_cuda = False
        if self.use_gpu:
            # 检查OpenCV CUDA支持
            try:
                if hasattr(cv2.cuda, 'SURF_CUDA'):
                    self.use_cuda = True
                    logger.info(f"启用OpenCV CUDA加速的SIFT特征提取器（设备：{gpu_device}）")
                else:
                    logger.warning("OpenCV CUDA支持不可用，使用CPU模式的SIFT")
            except Exception as e:
                logger.warning(f"检查CUDA支持时出错: {e}，使用CPU模式")
        
        # 初始化SIFT
        self.sift = cv2.SIFT_create(
            nfeatures=n_features,
            nOctaveLayers=n_octave_layers,
            contrastThreshold=contrast_threshold,
            edgeThreshold=edge_threshold,
            sigma=sigma
        )
        
        # 加载码本（如果有）
        if codebook_path:
            self._load_codebook(codebook_path)
        
        logger.info(f"初始化SIFT特征提取器: {encoding_method}, "
                   f"codebook_size={codebook_size}, "
                   f"n_features={n_features}, "
                   f"GPU模式={'开启' if self.use_cuda else '关闭'}")
    
    def _load_codebook(self, path: str):
        """加载预训练码本"""
        try:
            with open(path, 'rb') as f:
                loaded_data = pickle.load(f)
            
            # 检查并转换码本类型
            if isinstance(loaded_data, dict):
                # 如果加载的是字典，尝试从中提取'codebook'键
                if 'codebook' in loaded_data:
                    self.codebook = np.array(loaded_data['codebook'])
                    logger.info(f"已从字典格式加载码本: {path}")
                else:
                    logger.error(f"码本字典中缺少'codebook'键: {path}")
                    self.codebook = None
            elif isinstance(loaded_data, np.ndarray):
                self.codebook = loaded_data
                logger.info(f"已加载numpy数组格式码本: {path}")
            else:
                # 尝试转换为numpy数组
                try:
                    self.codebook = np.array(loaded_data)
                    logger.info(f"已转换并加载码本: {path}")
                except Exception as e:
                    logger.error(f"码本类型错误 ({type(loaded_data)}), 无法转换为numpy数组: {e}")
                    self.codebook = None
        except Exception as e:
            logger.error(f"加载码本失败: {e}")
            self.codebook = None
    
    def build_codebook(self, 
                       images: List[np.ndarray], 
                       max_descriptors: int = 100000,
                       save_path: Optional[str] = None) -> np.ndarray:
        """
        使用K-means构建码本
        
        Args:
            images: 训练图像列表
            max_descriptors: 最大使用的描述子数量
            save_path: 保存码本的路径
            
        Returns:
            codebook: 学习到的码本
        """
        logger.info(f"开始构建码本: {self.codebook_size}个聚类, {len(images)}张图像")
        
        all_descriptors = []
        
        # 提取所有图像的描述子
        for i, image in enumerate(images):
            if len(all_descriptors) >= max_descriptors:
                break
                
            try:
                # 转换为灰度图
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                else:
                    gray = image.copy()
                
                # 提取SIFT特征
                _, descriptors = self.sift.detectAndCompute(gray, None)
                
                if descriptors is not None and len(descriptors) > 0:
                    all_descriptors.append(descriptors)
            except Exception as e:
                logger.warning(f"处理图像 {i} 失败: {e}")
            
            if (i + 1) % 50 == 0:
                logger.info(f"  已处理 {i+1}/{len(images)} 张图像, 收集了 {sum(len(d) for d in all_descriptors)} 个描述子")
        
        # 合并所有描述子
        if not all_descriptors:
            raise RuntimeError("无法收集足够的描述子来构建码本")
        
        all_descriptors = np.vstack(all_descriptors)
        
        # 随机采样以限制数量
        if len(all_descriptors) > max_descriptors:
            indices = np.random.choice(len(all_descriptors), max_descriptors, replace=False)
            all_descriptors = all_descriptors[indices]
        
        logger.info(f"开始K-means聚类: {len(all_descriptors)} 个描述子, {self.codebook_size} 个聚类")
        
        # 选择K-means实现
        if self.use_gpu and CUML_AVAILABLE and len(all_descriptors) > 10000:
            # 使用GPU版本的K-means（对于大数据集更高效）
            logger.info("使用GPU加速的K-means聚类")
            try:
                # 转移数据到GPU
                if CUPY_AVAILABLE:
                    descriptors_gpu = cp.array(all_descriptors)
                
                # 使用cuml的K-means
                kmeans = cuKMeans(n_clusters=self.codebook_size, 
                                random_state=42, 
                                n_init=5,  # GPU版本通常需要更少的初始化次数
                                verbose=True)
                kmeans.fit(all_descriptors)
                
                self.codebook = kmeans.cluster_centers_
            except Exception as e:
                logger.error(f"GPU K-means失败: {e}，回退到CPU K-means")
                # 回退到CPU K-means
                kmeans = KMeans(n_clusters=self.codebook_size, 
                              random_state=42, 
                              n_init=10, 
                              verbose=1)
                kmeans.fit(all_descriptors)
                self.codebook = kmeans.cluster_centers_
        else:
            # 使用CPU版本的K-means
            kmeans = KMeans(n_clusters=self.codebook_size, 
                          random_state=42, 
                          n_init=10, 
                          verbose=1)
            kmeans.fit(all_descriptors)
            self.codebook = kmeans.cluster_centers_
        
        # 保存码本
        if save_path:
            save_dir = Path(save_path).parent
            save_dir.mkdir(parents=True, exist_ok=True)
            
            with open(save_path, 'wb') as f:
                pickle.dump(self.codebook, f)
            logger.info(f"码本已保存到: {save_path}")
        
        logger.info("码本构建完成")
        return self.codebook
    
    def extract(self, image) -> np.ndarray:
        """
        提取SIFT特征
        
        Args:
            image: BGR格式图像
            
        Returns:
            feature: 编码后的特征向量
        """
        # 确保输入是numpy数组，而不是cupy数组
        if self.use_gpu and hasattr(image, '__module__') and image.__module__ == 'cupy':
            import cupy as cp
            image = cp.asnumpy(image)
        
        # 确保输入是有效的numpy数组
        if not isinstance(image, np.ndarray):
            raise TypeError(f"期望numpy数组，得到: {type(image)}")
        
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 提取SIFT特征
        keypoints, descriptors = self.sift.detectAndCompute(gray, None)
        
        # 处理没有特征点的情况
        if descriptors is None or len(descriptors) == 0:
            logger.warning("未检测到SIFT特征点")
            # 返回零向量
            if self.encoding_method == 'bovw':
                return np.zeros(self.codebook_size)
            elif self.encoding_method == 'vlad':
                return np.zeros(self.codebook_size * 128)
            else:  # fisher
                return np.zeros(self.codebook_size * 256)
        
        # 编码特征
        if self.encoding_method == 'bovw':
            return self._encode_bovw(descriptors)
        elif self.encoding_method == 'vlad':
            return self._encode_vlad(descriptors)
        elif self.encoding_method == 'fisher':
            return self._encode_fisher(descriptors)
        else:
            raise ValueError(f"不支持的编码方法: {self.encoding_method}")
    
    def _encode_bovw(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Bag of Visual Words 编码
        """
        if self.codebook is None:
            raise RuntimeError("需要先构建或加载码本")
        
        # 计算每个描述子到最近聚类中心的距离
        distances = np.linalg.norm(descriptors[:, np.newaxis] - self.codebook, axis=2)
        
        # 找到最近的聚类中心
        nearest_indices = np.argmin(distances, axis=1)
        
        # 计算直方图
        histogram = np.bincount(nearest_indices, minlength=self.codebook_size)
        
        # 归一化
        histogram = histogram / (np.sum(histogram) + 1e-7)
        
        return histogram
    
    def _encode_vlad(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Vector of Locally Aggregated Descriptors 编码
        """
        if self.codebook is None:
            raise RuntimeError("需要先构建或加载码本")
        
        # 计算每个描述子到所有聚类中心的距离
        distances = np.linalg.norm(descriptors[:, np.newaxis] - self.codebook, axis=2)
        
        # 找到最近的聚类中心
        nearest_indices = np.argmin(distances, axis=1)
        
        # 初始化VLAD向量
        vlad = np.zeros((self.codebook_size, descriptors.shape[1]))
        
        # 累加残差
        for i, idx in enumerate(nearest_indices):
            vlad[idx] += descriptors[i] - self.codebook[idx]
        
        # 归一化 (intra-normalization)
        for i in range(self.codebook_size):
            vlad[i] /= (np.linalg.norm(vlad[i]) + 1e-7)
        
        # 展平
        vlad = vlad.flatten()
        
        # L2归一化
        vlad /= (np.linalg.norm(vlad) + 1e-7)
        
        return vlad
    
    def _encode_fisher(self, descriptors: np.ndarray) -> np.ndarray:
        """
        Fisher Vector 编码 (简化版)
        
        使用聚类中心作为均值，计算梯度
        """
        if self.codebook is None:
            raise RuntimeError("需要先构建或加载码本")
        
        # 计算每个描述子到所有聚类中心的距离
        distances = np.linalg.norm(descriptors[:, np.newaxis] - self.codebook, axis=2)
        
        # 计算soft assignment权重
        gamma = np.exp(-distances / (2.0 * 0.1))  # 使用固定的sigma
        gamma /= np.sum(gamma, axis=1, keepdims=True) + 1e-7
        
        # 初始化Fisher向量
        n_clusters, n_dims = self.codebook.shape
        fisher = np.zeros(n_clusters * 2 * n_dims)
        
        # 计算均值和方差的梯度
        for i in range(n_clusters):
            weight = gamma[:, i][:, np.newaxis]
            
            # 均值梯度
            mean_gradient = np.sum(weight * (descriptors - self.codebook[i]), axis=0)
            
            # 方差梯度 (简化为均值的平方)
            var_gradient = np.sum(weight * ((descriptors - self.codebook[i])**2 - 1), axis=0) / np.sqrt(2)
            
            # 存储到Fisher向量
            start_idx = i * 2 * n_dims
            fisher[start_idx:start_idx + n_dims] = mean_gradient
            fisher[start_idx + n_dims:start_idx + 2 * n_dims] = var_gradient
        
        # Power normalization 和 L2归一化
        fisher = np.sign(fisher) * np.abs(fisher)**0.5
        fisher /= (np.linalg.norm(fisher) + 1e-7)
        
        return fisher
    
    def compute_similarity(self, 
                          feature1: np.ndarray, 
                          feature2: np.ndarray, 
                          method: str = 'cosine') -> float:
        """
        计算两个SIFT特征之间的相似度
        
        Args:
            feature1: 特征向量1
            feature2: 特征向量2
            method: 相似度计算方法 ('cosine', 'l1', 'l2')
            
        Returns:
            similarity: 相似度得分 (0-1)
        """
        # 添加类型检查和错误处理
        if not isinstance(feature1, np.ndarray) or not isinstance(feature2, np.ndarray):
            logger.warning(f"SIFT特征类型错误: feature1={type(feature1)}, feature2={type(feature2)}")
            return 0.0
        
        # 检查维度是否匹配
        if feature1.shape != feature2.shape:
            logger.warning(f"SIFT特征维度不匹配: {feature1.shape} vs {feature2.shape}")
            return 0.0
        
        # 检查是否为空
        if feature1.size == 0 or feature2.size == 0:
            logger.warning("SIFT特征为空")
            return 0.0
        
        method = method.lower()
        
        try:
            if method == 'cosine':
                # 余弦相似度
                norm1 = np.linalg.norm(feature1)
                norm2 = np.linalg.norm(feature2)
                
                # 避免除零错误
                if norm1 < 1e-7 or norm2 < 1e-7:
                    return 0.0
                
                dot_product = np.dot(feature1, feature2)
                similarity = dot_product / (norm1 * norm2)
                # 确保值在有效范围内
                similarity = max(0.0, min(1.0, similarity))
            
            elif method == 'l1':
                # L1距离 -> 相似度
                l1 = np.sum(np.abs(feature1 - feature2))
                similarity = 1 / (1 + l1)
                # 确保值在有效范围内
                similarity = max(0.0, min(1.0, similarity))
            
            elif method == 'l2':
                # L2距离 -> 相似度
                l2 = np.linalg.norm(feature1 - feature2)
                similarity = 1 / (1 + l2)
                # 确保值在有效范围内
                similarity = max(0.0, min(1.0, similarity))
            
            else:
                logger.warning(f"不支持的相似度方法: {method}")
                return 0.0
            
        except Exception as e:
            logger.error(f"计算SIFT特征相似度时出错: {e}")
            return 0.0
        
        return similarity