import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

from feature_extraction.color_extractor import ColorFeatureExtractor
from feature_extraction.sift_extractor import SIFTFeatureExtractor
from feature_extraction.texture_extractor import TextureFeatureExtractor
from feature_extraction.text_extractor import TextFeatureExtractor

logger = logging.getLogger(__name__)

class FixedWeightMultimodalMatcherComplete:
    """
    四模态匹配算法（固定权重）
    
    使用颜色、SIFT、纹理和文字特征，所有类别共享固定权重
    """
    
    def __init__(self, 
                 color_config: Optional[Dict] = None,
                 sift_config: Optional[Dict] = None,
                 texture_config: Optional[Dict] = None,
                 text_config: Optional[Dict] = None,
                 weights: Optional[Dict] = None,
                 fusion_method: str = 'weighted_sum'):
        """
        Args:
            color_config: 颜色特征提取器配置
            sift_config: SIFT特征提取器配置
            texture_config: 纹理特征提取器配置
            text_config: 文字特征提取器配置
            weights: 各模态权重配置
            fusion_method: 融合方法 ('weighted_sum', 'max', 'min', 'mean')
        """
        # 初始化特征提取器
        self.color_extractor = ColorFeatureExtractor(**(color_config or {}))
        self.sift_extractor = SIFTFeatureExtractor(**(sift_config or {}))
        self.texture_extractor = TextureFeatureExtractor(**(texture_config or {}))
        self.text_extractor = TextFeatureExtractor(**(text_config or {}))
        
        # 设置默认权重
        default_weights = {
            'color': 0.25,
            'sift': 0.25,
            'texture': 0.25,
            'text': 0.25
        }
        
        # 更新权重
        self.weights = default_weights.copy()
        if weights:
            self.weights.update(weights)
        
        # 归一化权重
        total_weight = sum(self.weights.values())
        if total_weight <= 0:
            raise ValueError("权重和必须大于0")
        
        for key in self.weights:
            self.weights[key] = self.weights[key] / total_weight
        
        self.fusion_method = fusion_method
        
        # 特征缓存
        self.feature_cache = {}
        
        logger.info(f"初始化固定权重多模态匹配器: weights={self.weights}, "
                   f"fusion={fusion_method}")
    
    def extract_features(self, 
                        image: np.ndarray,
                        image_id: Optional[str] = None,
                        metadata: Optional[Dict] = None) -> Dict:
        """
        提取图像的多模态特征
        
        Args:
            image: BGR格式图像
            image_id: 图像ID，用于缓存
            metadata: 图像元数据（暂不使用）
            
        Returns:
            包含所有模态特征的字典
        """
        # 检查缓存
        if image_id and image_id in self.feature_cache:
            logger.debug(f"从缓存加载多模态特征: {image_id}")
            return self.feature_cache[image_id]
        
        # 提取各模态特征
        features = {
            'color': self.color_extractor.extract(image),
            'sift': self.sift_extractor.extract(image),
            'texture': self.texture_extractor.extract(image),
            'text': self.text_extractor.extract(image),
            'metadata': metadata or {}
        }
        
        # 缓存特征
        if image_id:
            self.feature_cache[image_id] = features
        
        return features
    
    def build_index(self, gallery_dataset):
        """
        构建图库索引
        
        Args:
            gallery_dataset: 图库数据集
        """
        logger.info(f"构建图库索引，数据集大小: {len(gallery_dataset)}")
        
        self.gallery_features_list = []
        
        for i, item in enumerate(gallery_dataset):
            if (i + 1) % 10 == 0:
                logger.info(f"处理图库图像 {i+1}/{len(gallery_dataset)}")
            
            # 尝试多种方式获取图像数据
            image = None
            try:
                if isinstance(item, dict):
                    # 尝试常见的图像键名
                    for key in ['image', 'img', 'data', 'image_data']:
                        if key in item and isinstance(item[key], np.ndarray):
                            image = item[key]
                            break
                    # 如果找不到图像，尝试第一个numpy数组值
                    if image is None:
                        for value in item.values():
                            if isinstance(value, np.ndarray):
                                image = value
                                break
                elif isinstance(item, tuple) and len(item) > 0:
                    # 如果是元组，假设第一个元素是图像
                    image = item[0]
                elif isinstance(item, np.ndarray):
                    # 直接是numpy数组
                    image = item
            except Exception as e:
                logger.warning(f"处理项 {i} 时出错: {str(e)}")
            
            if image is None or not isinstance(image, np.ndarray):
                logger.warning(f"无法从项 {i} 中提取图像数据，跳过")
                continue
            
            # 提取特征
            features = self.extract_features(
                image=image,
                image_id=str(i)
            )
            self.gallery_features_list.append(features)
        
        logger.info(f"图库索引构建完成，成功处理 {len(self.gallery_features_list)} 个图像")
    
    def search(self, query_image_path=None, category=None, k=10):
        """
        搜索相似图像
        
        Args:
            query_image_path: 查询图像的路径
            category: 查询图像的类别（可选）
            k: 返回前k个结果
            
        Returns:
            排序后的搜索结果列表
        """
        logger.info(f"执行搜索，k={k}")
        
        # 处理查询图像
        query_feature = None
        try:
            # 使用image_path加载图像
            if query_image_path:
                # 读取图像
                query_image = cv2.imread(query_image_path)
                if query_image is None:
                    logger.warning(f"无法读取图像: {query_image_path}")
                    return []
                
                # 提取特征
                query_feature = self.extract_features(query_image)
            
            if query_feature is None:
                logger.warning("无法从查询图像中提取特征")
                return []
                
        except Exception as e:
            logger.error(f"提取查询图像特征时出错: {str(e)}")
            return []
        
        # 批量匹配
        matches = self.batch_match([query_feature], self.gallery_features_list, k)
        
        # 构建搜索结果
        results = []
        if matches and len(matches) > 0:
            for idx, score in matches[0]:
                results.append({
                    'index': int(idx),
                    'score': float(score)
                })
        
        # 确保结果按分数降序排序
        results.sort(key=lambda x: x['score'], reverse=True)
        
        return results[:k]
    
    def match(self, 
             query_features: Dict,
             gallery_features_list: List[Dict],
             top_k: int = 10) -> List[Tuple[int, float]]:
        """
        执行多模态特征匹配
        
        Args:
            query_features: 查询特征
            gallery_features_list: 图库特征列表
            top_k: 返回前k个结果
            
        Returns:
            排序后的匹配结果列表 [(索引, 相似度)]
        """
        # 计算所有图库图像的相似度
        scores = []
        
        for i, gallery_features in enumerate(gallery_features_list):
            # 计算各模态相似度
            similarities = self._compute_modality_similarities(
                query_features, gallery_features
            )
            
            # 融合相似度
            fused_sim = self._fuse_similarities(similarities)
            
            scores.append((i, fused_sim))
        
        # 按相似度降序排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前top_k个结果
        return scores[:top_k]
    
    def _compute_modality_similarities(self, 
                                     query_features: Dict,
                                     gallery_features: Dict) -> Dict[str, float]:
        """
        计算各模态的相似度
        """
        similarities = {}
        
        # 颜色相似度
        similarities['color'] = self.color_extractor.compute_similarity(
            query_features['color'],
            gallery_features['color'],
            method='cosine'
        )
        
        # SIFT相似度
        similarities['sift'] = self.sift_extractor.compute_similarity(
            query_features['sift'],
            gallery_features['sift'],
            method='cosine'
        )
        
        # 纹理相似度
        similarities['texture'] = self.texture_extractor.compute_similarity(
            query_features['texture'],
            gallery_features['texture'],
            method='cosine'
        )
        
        # 文字相似度
        similarities['text'] = self.text_extractor.compute_similarity(
            query_features['text'],
            gallery_features['text'],
            method='combined'
        )
        
        return similarities
    
    def _fuse_similarities(self, similarities: Dict[str, float]) -> float:
        """
        融合多种模态的相似度
        """
        if self.fusion_method == 'weighted_sum':
            # 加权求和
            fused = 0.0
            for key, weight in self.weights.items():
                fused += weight * similarities.get(key, 0.0)
            return fused
        
        elif self.fusion_method == 'max':
            # 取最大值
            return max(similarities.values())
        
        elif self.fusion_method == 'min':
            # 取最小值
            return min(similarities.values())
        
        elif self.fusion_method == 'mean':
            # 取平均值
            return sum(similarities.values()) / len(similarities)
        
        else:
            raise ValueError(f"不支持的融合方法: {self.fusion_method}")
    
    def batch_match(self, 
                   query_features_list: List[Dict],
                   gallery_features_list: List[Dict],
                   top_k: int = 10) -> List[List[Tuple[int, float]]]:
        """
        批量执行匹配
        
        Args:
            query_features_list: 查询特征列表
            gallery_features_list: 图库特征列表
            top_k: 返回前k个结果
            
        Returns:
            每个查询的匹配结果列表
        """
        results = []
        
        for i, query_features in enumerate(query_features_list):
            if (i + 1) % 10 == 0:
                logger.info(f"处理查询 {i+1}/{len(query_features_list)}")
            
            result = self.match(query_features, gallery_features_list, top_k)
            results.append(result)
        
        return results
    
    def clear_cache(self):
        """
        清空特征缓存
        """
        self.feature_cache.clear()
        logger.info("特征缓存已清空")
    
    def get_config(self) -> Dict:
        """
        获取配置信息
        
        Returns:
            配置字典
        """
        return {
            'color_config': {
                'color_space': self.color_extractor.color_space,
                'bins': self.color_extractor.bins,
                'spatial_pyramid_levels': self.color_extractor.spatial_pyramid_levels,
                'normalize': self.color_extractor.normalize
            },
            'sift_config': {
                'n_features': self.sift_extractor.n_features,
                'encoding_method': self.sift_extractor.encoding_method,
                'codebook_size': self.sift_extractor.codebook_size
            },
            'texture_config': {
                'feature_types': self.texture_extractor.feature_types,
                'spatial_pyramid_levels': self.texture_extractor.spatial_pyramid_levels
            },
            'text_config': {
                'lang': self.text_extractor.lang
            },
            'weights': self.weights,
            'fusion_method': self.fusion_method
        }
    
    def set_config(self, config: Dict):
        """
        设置配置信息
        
        Args:
            config: 配置字典
        """
        # 更新各特征提取器配置
        if 'color_config' in config:
            color_config = config['color_config']
            for key, value in color_config.items():
                if hasattr(self.color_extractor, key):
                    setattr(self.color_extractor, key, value)
        
        if 'sift_config' in config:
            sift_config = config['sift_config']
            for key, value in sift_config.items():
                if hasattr(self.sift_extractor, key):
                    setattr(self.sift_extractor, key, value)
        
        if 'texture_config' in config:
            texture_config = config['texture_config']
            for key, value in texture_config.items():
                if hasattr(self.texture_extractor, key):
                    setattr(self.texture_extractor, key, value)
        
        if 'text_config' in config:
            text_config = config['text_config']
            for key, value in text_config.items():
                if hasattr(self.text_extractor, key):
                    setattr(self.text_extractor, key, value)
        
        # 更新权重
        if 'weights' in config:
            self.weights.update(config['weights'])
            # 重新归一化权重
            total_weight = sum(self.weights.values())
            if total_weight > 0:
                for key in self.weights:
                    self.weights[key] = self.weights[key] / total_weight
        
        # 更新融合方法
        if 'fusion_method' in config:
            self.fusion_method = config['fusion_method']
        
        # 清空缓存，因为配置已更改
        self.clear_cache()
        
        logger.info(f"更新配置: {config}")
    
    def optimize_weights(self,
                        query_features_list: List[Dict],
                        gallery_features_list: List[Dict],
                        ground_truth_indices_list: List[List[int]],
                        n_trials: int = 50) -> Dict:
        """
        优化多模态权重
        
        Args:
            query_features_list: 查询特征列表
            gallery_features_list: 图库特征列表
            ground_truth_indices_list: 每个查询的真实正样本索引列表
            n_trials: 随机搜索的试验次数
            
        Returns:
            最佳权重配置
        """
        best_weights = None
        best_map = 0.0
        
        logger.info(f"开始权重优化，共 {n_trials} 次试验")
        
        for trial in range(n_trials):
            # 随机生成权重
            weights = {
                'color': np.random.uniform(0.1, 0.4),
                'sift': np.random.uniform(0.1, 0.4),
                'texture': np.random.uniform(0.1, 0.4),
                'text': np.random.uniform(0.1, 0.4)
            }
            
            # 归一化
            total = sum(weights.values())
            for key in weights:
                weights[key] = weights[key] / total
            
            # 保存当前权重
            current_weights = self.weights.copy()
            
            # 设置新权重
            self.weights = weights
            
            logger.info(f"试验 {trial+1}/{n_trials}: weights={weights}")
            
            # 评估性能
            aps = []
            
            for query_features, ground_truth_indices in zip(
                query_features_list, ground_truth_indices_list
            ):
                # 执行匹配
                matches = self.match(query_features, gallery_features_list, top_k=len(gallery_features_list))
                
                # 计算平均精度
                indices = [idx for idx, _ in matches]
                precision = 0.0
                relevant_count = 0
                
                for i, idx in enumerate(indices):
                    if idx in ground_truth_indices:
                        relevant_count += 1
                        precision += relevant_count / (i + 1)
                
                if relevant_count > 0:
                    ap = precision / relevant_count
                    aps.append(ap)
            
            # 计算MAP
            if aps:
                map_score = np.mean(aps)
            else:
                map_score = 0.0
            
            logger.info(f"  MAP: {map_score:.4f}")
            
            # 更新最佳权重
            if map_score > best_map:
                best_map = map_score
                best_weights = weights.copy()
                logger.info(f"  找到更好的权重! MAP = {best_map:.4f}")
            
            # 恢复当前权重
            self.weights = current_weights
        
        if best_weights:
            logger.info(f"权重优化完成，最佳权重: {best_weights}, 最佳MAP: {best_map:.4f}")
            # 更新为最佳权重
            self.weights = best_weights
        else:
            logger.warning("权重优化失败，未找到有效权重")
        
        return best_weights
    
    def build_sift_codebook(self, 
                           images: List[np.ndarray], 
                           save_path: Optional[str] = None) -> np.ndarray:
        """
        构建SIFT码本
        
        Args:
            images: 训练图像列表
            save_path: 保存路径
            
        Returns:
            学习到的码本
        """
        try:
            codebook = self.sift_extractor.build_codebook(images, save_path=save_path)
            logger.info("SIFT码本构建完成")
            return codebook
        except Exception as e:
            logger.error(f"构建SIFT码本失败: {e}")
            raise