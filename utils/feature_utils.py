import os
import json
import numpy as np
import cv2
import pickle
from typing import List, Dict, Tuple, Optional, Any, Callable
import logging
from sklearn.preprocessing import normalize
import faiss

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FeatureUtils:
    """
    特征处理工具类
    
    提供特征提取、存储、加载、索引构建等功能
    """
    
    @staticmethod
    def extract_sift_features(images: List[np.ndarray], 
                            n_features: int = 200,
                            normalize_features: bool = True) -> List[np.ndarray]:
        """
        提取SIFT特征
        
        Args:
            images: 图像列表
            n_features: 最大特征点数量
            normalize_features: 是否归一化特征
            
        Returns:
            特征列表
        """
        try:
            # 初始化SIFT
            sift = cv2.SIFT_create(nfeatures=n_features)
            
            features_list = []
            for i, image in enumerate(images):
                # 转换为灰度图
                if len(image.shape) == 3:
                    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    gray = image
                
                # 提取特征
                keypoints, descriptors = sift.detectAndCompute(gray, None)
                
                if descriptors is None:
                    # 如果没有提取到特征，使用零向量
                    features = np.zeros((1, 128), dtype=np.float32)
                else:
                    features = descriptors[:n_features]  # 限制数量
                    
                    # 归一化
                    if normalize_features:
                        features = normalize(features)
                
                features_list.append(features)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"已提取 {i + 1}/{len(images)} 张图像的SIFT特征")
            
            return features_list
            
        except Exception as e:
            logger.error(f"提取SIFT特征失败: {e}")
            return [np.zeros((1, 128), dtype=np.float32) for _ in images]
    
    @staticmethod
    def build_bow_histogram(descriptors_list: List[np.ndarray],
                          codebook: np.ndarray,
                          num_words: int = 1024) -> List[np.ndarray]:
        """
        构建词袋直方图
        
        Args:
            descriptors_list: 描述符列表
            codebook: 码本
            num_words: 单词数量
            
        Returns:
            词袋直方图列表
        """
        try:
            # 创建FAISS索引进行快速最近邻搜索
            index = faiss.IndexFlatL2(codebook.shape[1])
            index.add(codebook.astype(np.float32))
            
            histograms = []
            
            for i, descriptors in enumerate(descriptors_list):
                if descriptors is None or len(descriptors) == 0:
                    # 如果没有特征，返回零向量
                    histogram = np.zeros(num_words, dtype=np.float32)
                else:
                    # 查找最近邻
                    _, indices = index.search(descriptors.astype(np.float32), 1)
                    
                    # 构建直方图
                    histogram = np.bincount(indices.flatten(), minlength=num_words).astype(np.float32)
                    
                    # 归一化
                    if np.sum(histogram) > 0:
                        histogram = histogram / np.sum(histogram)
                
                histograms.append(histogram)
                
                if (i + 1) % 100 == 0:
                    logger.info(f"已构建 {i + 1}/{len(descriptors_list)} 个词袋直方图")
            
            return histograms
            
        except Exception as e:
            logger.error(f"构建词袋直方图失败: {e}")
            return [np.zeros(num_words, dtype=np.float32) for _ in descriptors_list]
    
    @staticmethod
    def extract_color_histogram(image: np.ndarray,
                              bins: int = 8,
                              color_space: str = 'hsv',
                              normalize: bool = True) -> np.ndarray:
        """
        提取颜色直方图
        
        Args:
            image: 输入图像
            bins: 直方图分箱数
            color_space: 颜色空间 (hsv/rgb/lab)
            normalize: 是否归一化
            
        Returns:
            颜色直方图
        """
        try:
            # 转换颜色空间
            if color_space == 'hsv':
                img = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'lab':
                img = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
            else:  # rgb
                img = image.copy()
            
            # 计算每个通道的直方图
            histograms = []
            for i in range(3):
                if color_space == 'hsv' and i == 0:  # H通道范围是0-179
                    hist_range = [0, 180]
                else:  # 其他通道是0-255
                    hist_range = [0, 256]
                
                hist = cv2.calcHist([img], [i], None, [bins], hist_range)
                histograms.append(hist.flatten())
            
            # 合并直方图
            histogram = np.concatenate(histograms)
            
            # 归一化
            if normalize and np.sum(histogram) > 0:
                histogram = histogram / np.sum(histogram)
            
            return histogram
            
        except Exception as e:
            logger.error(f"提取颜色直方图失败: {e}")
            return np.zeros(3 * bins, dtype=np.float32)
    
    @staticmethod
    def extract_multiple_color_histograms(image: np.ndarray,
                                        bins: List[int] = None,
                                        color_spaces: List[str] = None) -> Dict[str, np.ndarray]:
        """
        提取多种颜色直方图
        
        Args:
            image: 输入图像
            bins: 不同分箱数
            color_spaces: 不同颜色空间
            
        Returns:
            直方图字典
        """
        if bins is None:
            bins = [4, 8, 16]
        
        if color_spaces is None:
            color_spaces = ['hsv', 'rgb']
        
        histograms = {}
        
        for color_space in color_spaces:
            for bin_size in bins:
                key = f"{color_space}_bins_{bin_size}"
                histograms[key] = FeatureUtils.extract_color_histogram(
                    image, bin_size, color_space
                )
        
        return histograms
    
    @staticmethod
    def save_features(features: List[np.ndarray],
                     output_path: str,
                     ids: Optional[List[str]] = None) -> bool:
        """
        保存特征到文件
        
        Args:
            features: 特征列表
            output_path: 输出路径
            ids: 特征ID列表
            
        Returns:
            是否保存成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 准备数据
            data = {
                'features': features,
                'ids': ids or [str(i) for i in range(len(features))],
                'metadata': {
                    'num_features': len(features),
                    'feature_dim': features[0].shape[0] if features else 0
                }
            }
            
            # 保存到文件
            with open(output_path, 'wb') as f:
                pickle.dump(data, f)
            
            logger.info(f"特征已保存到: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存特征失败: {e}")
            return False
    
    @staticmethod
    def load_features(input_path: str) -> Tuple[List[np.ndarray], List[str], Dict[str, Any]]:
        """
        从文件加载特征
        
        Args:
            input_path: 输入路径
            
        Returns:
            (特征列表, ID列表, 元数据)
        """
        try:
            with open(input_path, 'rb') as f:
                data = pickle.load(f)
            
            features = data.get('features', [])
            ids = data.get('ids', [])
            metadata = data.get('metadata', {})
            
            logger.info(f"从 {input_path} 加载了 {len(features)} 个特征")
            return features, ids, metadata
            
        except Exception as e:
            logger.error(f"加载特征失败: {e}")
            return [], [], {}
    
    @staticmethod
    def build_faiss_index(features: List[np.ndarray],
                        index_type: str = 'flat',
                        nlist: int = 100,
                        m: int = 16,
                        efConstruction: int = 200,
                        use_gpu: bool = False,
                        gpu_device: int = 0) -> Any:
        """
        构建FAISS索引
        
        Args:
            features: 特征列表
            index_type: 索引类型 (flat/ivf/hnsw)
            nlist: IVF索引的聚类数量
            m: HNSW的每个节点的邻居数
            efConstruction: HNSW的构建参数
            use_gpu: 是否使用GPU加速
            gpu_device: GPU设备ID
            
        Returns:
            FAISS索引
        """
        try:
            # 类型检查和预处理
            processed_features = []
            for i, feat in enumerate(features):
                if not isinstance(feat, np.ndarray):
                    logger.warning(f"特征 {i} 不是numpy数组，类型: {type(feat)}")
                    # 跳过无效特征
                    continue
                if feat.size == 0:
                    logger.warning(f"特征 {i} 是空数组")
                    continue
                # 转换为float32
                processed_features.append(feat.astype(np.float32))
            
            if not processed_features:
                logger.error("没有有效特征用于构建索引")
                return None
            
            # 检查特征维度是否一致
            feature_dims = {f.shape[0] for f in processed_features}
            if len(feature_dims) > 1:
                logger.warning(f"特征维度不一致: {feature_dims}")
                # 使用第一个有效特征的维度
                d = processed_features[0].shape[0]
                # 对维度不一致的特征进行处理（这里简单地截断或填充）
                for i in range(len(processed_features)):
                    if processed_features[i].shape[0] != d:
                        if processed_features[i].shape[0] > d:
                            # 截断
                            processed_features[i] = processed_features[i][:d]
                        else:
                            # 填充零
                            pad = np.zeros(d - processed_features[i].shape[0], dtype=np.float32)
                            processed_features[i] = np.concatenate([processed_features[i], pad])
            else:
                d = processed_features[0].shape[0]
            
            # 转换为numpy数组
            feature_array = np.array(processed_features).astype(np.float32)
            
            # 创建索引
            if index_type == 'flat':
                index = faiss.IndexFlatL2(d)
            elif index_type == 'ivf':
                quantizer = faiss.IndexFlatL2(d)
                index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
                # IVF需要训练
                index.train(feature_array)
            elif index_type == 'hnsw':
                index = faiss.IndexHNSWFlat(d, m)
                index.hnsw.efConstruction = efConstruction
            else:
                raise ValueError(f"不支持的索引类型: {index_type}")
            
            # 添加特征
            index.add(feature_array)
            
            # GPU加速
            if use_gpu:
                try:
                    # 检查是否有GPU可用
                    if faiss.get_num_gpus() > 0:
                        # 将索引转移到GPU
                        res = faiss.StandardGpuResources()
                        if index_type == 'flat':
                            index = faiss.index_cpu_to_gpu(res, gpu_device, index)
                        elif index_type == 'ivf':
                            # IVF索引的GPU转移需要特殊处理
                            index = faiss.index_cpu_to_gpu(res, gpu_device, index)
                            # 对IVF索引设置nprobe参数以加速搜索
                            index.nprobe = min(nlist, 10)  # 设置适当的nprobe值
                        elif index_type == 'hnsw':
                            # HNSW索引的GPU转移
                            index = faiss.index_cpu_to_gpu(res, gpu_device, index)
                        logger.info(f"已将索引转移到GPU {gpu_device}")
                    else:
                        logger.warning("没有可用的GPU，继续使用CPU索引")
                except Exception as gpu_error:
                    logger.warning(f"GPU加速失败: {gpu_error}，继续使用CPU索引")
            
            logger.info(f"已构建 {index_type} 索引，包含 {index.ntotal} 个特征，使用{'GPU' if use_gpu and hasattr(index, 'device') else 'CPU'}")
            return index
            
        except Exception as e:
            logger.error(f"构建FAISS索引失败: {e}")
            return None
    
    @staticmethod
    def search_faiss_index(index: Any,
                         query_features: List[np.ndarray],
                         k: int = 10,
                         batch_size: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        """
        在FAISS索引中搜索
        
        Args:
            index: FAISS索引
            query_features: 查询特征
            k: 返回的邻居数量
            batch_size: 批处理大小，对于GPU加速尤为重要
            
        Returns:
            (距离矩阵, 索引矩阵)
        """
        try:
            # 转换为numpy数组
            query_array = np.array(query_features).astype(np.float32)
            num_queries = query_array.shape[0]
            
            # 初始化结果数组
            all_distances = np.zeros((num_queries, k), dtype=np.float32)
            all_indices = np.zeros((num_queries, k), dtype=np.int64)
            
            # 批处理搜索，避免内存问题并优化GPU性能
            for i in range(0, num_queries, batch_size):
                end_idx = min(i + batch_size, num_queries)
                batch_queries = query_array[i:end_idx]
                
                # 搜索
                distances, indices = index.search(batch_queries, k)
                
                # 保存结果
                all_distances[i:end_idx] = distances
                all_indices[i:end_idx] = indices
            
            # 检查是否使用GPU
            is_gpu = hasattr(index, 'device')
            logger.debug(f"FAISS搜索完成，查询数: {num_queries}, 结果数: {k}, 使用{'GPU' if is_gpu else 'CPU'}")
            
            return all_distances, all_indices
            
        except Exception as e:
            logger.error(f"FAISS搜索失败: {e}")
            # 返回空结果
            num_queries = len(query_features)
            return np.zeros((num_queries, k)), np.zeros((num_queries, k), dtype=np.int64)
    
    @staticmethod
    def normalize_features(features: List[np.ndarray],
                         norm: str = 'l2') -> List[np.ndarray]:
        """
        归一化特征
        
        Args:
            features: 特征列表
            norm: 归一化类型 (l1/l2/max)
            
        Returns:
            归一化后的特征
        """
        try:
            normalized = []
            
            for feat in features:
                if norm == 'l2':
                    norm_val = np.linalg.norm(feat)
                    if norm_val > 0:
                        normalized_feat = feat / norm_val
                    else:
                        normalized_feat = feat.copy()
                elif norm == 'l1':
                    norm_val = np.sum(np.abs(feat))
                    if norm_val > 0:
                        normalized_feat = feat / norm_val
                    else:
                        normalized_feat = feat.copy()
                elif norm == 'max':
                    norm_val = np.max(np.abs(feat))
                    if norm_val > 0:
                        normalized_feat = feat / norm_val
                    else:
                        normalized_feat = feat.copy()
                else:
                    raise ValueError(f"不支持的归一化类型: {norm}")
                
                normalized.append(normalized_feat)
            
            return normalized
            
        except Exception as e:
            logger.error(f"归一化特征失败: {e}")
            return features
    
    @staticmethod
    def fuse_features(features_list: List[List[np.ndarray]],
                    method: str = 'concatenate',
                    weights: Optional[List[float]] = None) -> List[np.ndarray]:
        """
        融合多个特征列表
        
        Args:
            features_list: 特征列表的列表
            method: 融合方法 (concatenate/weighted_addition)
            weights: 权重列表
            
        Returns:
            融合后的特征列表
        """
        try:
            # 检查特征数量是否一致
            num_features = len(features_list[0])
            for i, feats in enumerate(features_list[1:], 2):
                if len(feats) != num_features:
                    raise ValueError(f"特征列表 {i} 的长度不一致")
            
            # 设置默认权重
            if weights is None:
                weights = [1.0 / len(features_list)] * len(features_list)
            
            # 检查权重数量
            if len(weights) != len(features_list):
                raise ValueError("权重数量与特征列表数量不匹配")
            
            fused_features = []
            
            for i in range(num_features):
                # 获取当前样本的所有特征
                sample_features = [feats[i] for feats in features_list]
                
                if method == 'concatenate':
                    # 连接特征
                    fused = np.concatenate(sample_features)
                elif method == 'weighted_addition':
                    # 加权相加
                    # 确保所有特征维度相同
                    first_dim = sample_features[0].shape[0]
                    for feat in sample_features[1:]:
                        if feat.shape[0] != first_dim:
                            raise ValueError("加权相加需要所有特征维度相同")
                    
                    fused = np.zeros_like(sample_features[0])
                    for j, (feat, weight) in enumerate(zip(sample_features, weights)):
                        fused += feat * weight
                else:
                    raise ValueError(f"不支持的融合方法: {method}")
                
                fused_features.append(fused)
            
            logger.info(f"已融合 {len(features_list)} 个特征列表，方法: {method}")
            return fused_features
            
        except Exception as e:
            logger.error(f"融合特征失败: {e}")
            # 返回第一个特征列表
            return features_list[0] if features_list else []
    
    @staticmethod
    def compute_feature_distances(query_features: np.ndarray,
                                 gallery_features: List[np.ndarray],
                                 metric: str = 'euclidean') -> np.ndarray:
        """
        计算特征距离
        
        Args:
            query_features: 查询特征
            gallery_features: 图库特征
            metric: 距离度量 (euclidean/cosine/manhattan)
            
        Returns:
            距离数组
        """
        try:
            distances = np.zeros(len(gallery_features), dtype=np.float32)
            
            if metric == 'euclidean':
                for i, feat in enumerate(gallery_features):
                    distances[i] = np.linalg.norm(query_features - feat)
            elif metric == 'cosine':
                # 计算余弦相似度，转换为距离 (1 - 相似度)
                for i, feat in enumerate(gallery_features):
                    if np.linalg.norm(query_features) > 0 and np.linalg.norm(feat) > 0:
                        similarity = np.dot(query_features, feat) / \
                                   (np.linalg.norm(query_features) * np.linalg.norm(feat))
                        distances[i] = 1 - similarity
                    else:
                        distances[i] = 1.0  # 最大距离
            elif metric == 'manhattan':
                for i, feat in enumerate(gallery_features):
                    distances[i] = np.sum(np.abs(query_features - feat))
            else:
                raise ValueError(f"不支持的距离度量: {metric}")
            
            return distances
            
        except Exception as e:
            logger.error(f"计算特征距离失败: {e}")
            return np.zeros(len(gallery_features), dtype=np.float32)
    
    @staticmethod
    def find_nearest_neighbors(query_features: np.ndarray,
                             gallery_features: List[np.ndarray],
                             k: int = 10,
                             metric: str = 'euclidean') -> Tuple[List[int], List[float]]:
        """
        查找最近邻
        
        Args:
            query_features: 查询特征
            gallery_features: 图库特征
            k: 返回的邻居数量
            metric: 距离度量
            
        Returns:
            (索引列表, 距离列表)
        """
        # 计算所有距离
        distances = FeatureUtils.compute_feature_distances(
            query_features, gallery_features, metric
        )
        
        # 排序并返回前k个
        k = min(k, len(gallery_features))
        indices = np.argsort(distances)[:k]
        sorted_distances = distances[indices]
        
        return indices.tolist(), sorted_distances.tolist()
    
    @staticmethod
    def batch_find_nearest_neighbors(query_features: List[np.ndarray],
                                  gallery_features: List[np.ndarray],
                                  k: int = 10,
                                  metric: str = 'euclidean') -> List[Tuple[List[int], List[float]]]:
        """
        批量查找最近邻
        
        Args:
            query_features: 查询特征列表
            gallery_features: 图库特征
            k: 返回的邻居数量
            metric: 距离度量
            
        Returns:
            结果列表
        """
        results = []
        
        for i, q_feat in enumerate(query_features):
            indices, distances = FeatureUtils.find_nearest_neighbors(
                q_feat, gallery_features, k, metric
            )
            results.append((indices, distances))
            
            if (i + 1) % 100 == 0:
                logger.info(f"已处理 {i + 1}/{len(query_features)} 个查询")
        
        return results
    
    @staticmethod
    def save_index(index: Any,
                  output_path: str,
                  metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        保存索引
        
        Args:
            index: FAISS索引
            output_path: 输出路径
            metadata: 元数据
            
        Returns:
            是否保存成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 保存索引
            faiss.write_index(index, output_path)
            
            # 保存元数据
            if metadata:
                meta_path = output_path + '.json'
                with open(meta_path, 'w', encoding='utf-8') as f:
                    json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"索引已保存到: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存索引失败: {e}")
            return False
    
    @staticmethod
    def load_index(input_path: str) -> Tuple[Any, Optional[Dict[str, Any]]]:
        """
        加载索引
        
        Args:
            input_path: 输入路径
            
        Returns:
            (FAISS索引, 元数据)
        """
        try:
            # 加载索引
            index = faiss.read_index(input_path)
            
            # 加载元数据
            meta_path = input_path + '.json'
            metadata = None
            if os.path.exists(meta_path):
                with open(meta_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
            
            logger.info(f"从 {input_path} 加载了索引，包含 {index.ntotal} 个特征")
            return index, metadata
            
        except Exception as e:
            logger.error(f"加载索引失败: {e}")
            return None, None

# 工具函数
def create_codebook(features_list: List[List[np.ndarray]],
                   codebook_size: int = 1024,
                   num_samples: int = 100000) -> np.ndarray:
    """
    创建码本
    
    Args:
        features_list: 特征列表的列表
        codebook_size: 码本大小
        num_samples: 用于训练的样本数量
        
    Returns:
        码本
    """
    try:
        from sklearn.cluster import KMeans
        import random
        
        # 收集所有特征
        all_features = []
        for features in features_list:
            for feat in features:
                if len(feat) > 0:
                    all_features.extend(feat)
        
        logger.info(f"收集了 {len(all_features)} 个特征点用于训练码本")
        
        # 采样特征点
        if len(all_features) > num_samples:
            all_features = random.sample(all_features, num_samples)
        
        # 检查特征数量
        if len(all_features) < codebook_size:
            logger.error(f"特征数量不足: {len(all_features)} < {codebook_size}")
            # 使用随机码本
            dim = all_features[0].shape[0] if all_features else 128
            return np.random.random((codebook_size, dim)).astype(np.float32)
        
        # 转换为numpy数组
        features_array = np.array(all_features).astype(np.float32)
        
        # 训练KMeans
        logger.info(f"开始训练KMeans，特征点数量: {len(features_array)}")
        
        kmeans = KMeans(
            n_clusters=codebook_size,
            random_state=42,
            n_init=10,
            max_iter=300
        )
        kmeans.fit(features_array)
        
        logger.info(f"码本训练完成，大小: {codebook_size}")
        return kmeans.cluster_centers_
        
    except Exception as e:
        logger.error(f"创建码本失败: {e}")
        # 返回随机码本
        return np.random.random((codebook_size, 128)).astype(np.float32)

# 示例用法
if __name__ == '__main__':
    # 示例：提取SIFT特征
    # images = [...]  # 图像列表
    # sift_features = FeatureUtils.extract_sift_features(images)
    # 
    # # 构建词袋直方图
    # codebook = np.random.random((1024, 128)).astype(np.float32)  # 示例码本
    # bow_histograms = FeatureUtils.build_bow_histogram(sift_features, codebook)
    # 
    # # 保存特征
    # FeatureUtils.save_features(bow_histograms, './output/bow_features.pkl')
    # 
    # # 构建索引
    # index = FeatureUtils.build_faiss_index(bow_histograms, index_type='ivf')
    # 
    # # 搜索
    # query_features = bow_histograms[:10]
    # distances, indices = FeatureUtils.search_faiss_index(index, query_features, k=5)
    # 
    logger.info("特征工具类已加载")