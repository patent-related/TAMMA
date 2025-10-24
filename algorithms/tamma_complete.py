import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
from pathlib import Path

from feature_extraction.color_extractor import ColorFeatureExtractor
from feature_extraction.sift_extractor import SIFTFeatureExtractor
from feature_extraction.texture_extractor import TextureFeatureExtractor
from feature_extraction.text_extractor import TextFeatureExtractor
from utils.feature_utils import FeatureUtils

logger = logging.getLogger(__name__)

class TAMMAComplete:
    """
    TAMMA (Text-Aware Multi-Modal Aggregation) 完整算法实现
    
    实现了三级分层匹配策略：
    1. 颜色粗筛选
    2. 时空约束过滤
    3. 多模态精确匹配
    """
    
    def __init__(self, 
                 color_config: Optional[Dict] = None,
                 sift_config: Optional[Dict] = None,
                 texture_config: Optional[Dict] = None,
                 text_config: Optional[Dict] = None,
                 category_weights: Optional[Dict] = None,
                 color_threshold: float = 0.5,
                 top_k_coarse: int = 50,
                 spatial_weight: float = 0.3,
                 temporal_weight: float = 0.2,
                 fusion_method: str = 'weighted_sum',
                 use_gpu: bool = False,
                 gpu_device: int = 0):
        """
        Args:
            color_config: 颜色特征提取器配置
            sift_config: SIFT特征提取器配置
            texture_config: 纹理特征提取器配置
            text_config: 文字特征提取器配置
            category_weights: 类别特定权重配置
            color_threshold: 颜色粗筛选阈值
            top_k_coarse: 粗筛选保留的候选数量
            spatial_weight: 空间约束权重
            temporal_weight: 时间约束权重
            fusion_method: 融合方法 ('weighted_sum', 'max', 'min', 'mean')
            use_gpu: 是否使用GPU加速
            gpu_device: GPU设备ID
        """
        # 初始化特征提取器
        color_config_dict = color_config or {}
        color_config_dict['use_gpu'] = use_gpu
        if use_gpu:
            color_config_dict['gpu_device'] = gpu_device
        self.color_extractor = ColorFeatureExtractor(**color_config_dict)
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        # 初始化SIFT特征提取器
        sift_config_dict = sift_config or {}
        sift_config_dict['use_gpu'] = self.use_gpu
        if self.use_gpu:
            sift_config_dict['gpu_device'] = self.gpu_device
        self.sift_extractor = SIFTFeatureExtractor(**sift_config_dict)
        # 初始化纹理特征提取器
        texture_config_dict = texture_config or {}
        texture_config_dict['use_gpu'] = self.use_gpu
        if self.use_gpu:
            texture_config_dict['gpu_device'] = self.gpu_device
        self.texture_extractor = TextureFeatureExtractor(**texture_config_dict)
        # 初始化文本特征提取器
        text_config_dict = text_config or {}
        text_config_dict['use_gpu'] = self.use_gpu
        if self.use_gpu:
            text_config_dict['gpu_device'] = self.gpu_device
        self.text_extractor = TextFeatureExtractor(**text_config_dict)
        
        # 配置参数
        self.color_threshold = color_threshold
        self.top_k_coarse = top_k_coarse
        self.spatial_weight = spatial_weight
        self.temporal_weight = temporal_weight
        self.fusion_method = fusion_method
        
        # 类别特定权重
        self.category_weights = category_weights or self._get_default_category_weights()
        
        # 缓存
        self.feature_cache = {}
        
        logger.info(f"初始化TAMMA算法: fusion_method={fusion_method}")
    
    def _get_default_category_weights(self) -> Dict:
        """
        获取默认的类别特定权重配置
        """
        return {
            'book': {
                'color': 0.1,
                'sift': 0.2,
                'texture': 0.2,
                'text': 0.4,
                'spatio_temporal': 0.1
            },
            'wallet': {
                'color': 0.3,
                'sift': 0.3,
                'texture': 0.2,
                'text': 0.1,
                'spatio_temporal': 0.1
            },
            'cup': {
                'color': 0.4,
                'sift': 0.2,
                'texture': 0.2,
                'text': 0.1,
                'spatio_temporal': 0.1
            },
            'phone': {
                'color': 0.2,
                'sift': 0.3,
                'texture': 0.2,
                'text': 0.2,
                'spatio_temporal': 0.1
            },
            
            'key': {
                'color': 0.2,
                'sift': 0.4,
                'texture': 0.3,
                'text': 0.0,
                'spatio_temporal': 0.1
            },
            'bag': {
                'color': 0.3,
                'sift': 0.2,
                'texture': 0.3,
                'text': 0.1,
                'spatio_temporal': 0.1
            },
            'laptop': {
                'color': 0.3,
                'sift': 0.3,
                'texture': 0.2,
                'text': 0.1,
                'spatio_temporal': 0.1
            },
            'clothes': {
                'color': 0.4,
                'sift': 0.1,
                'texture': 0.3,
                'text': 0.1,
                'spatio_temporal': 0.1
            },
            'other': {
                'color': 0.25,
                'sift': 0.25,
                'texture': 0.25,
                'text': 0.15,
                'spatio_temporal': 0.1
            }
        }
    
    def extract_features(self, 
                        image: np.ndarray,
                        image_id: Optional[str] = None,
                        category: Optional[str] = None,
                        metadata: Optional[Dict] = None) -> Dict:
        """
        提取图像的多模态特征
        
        Args:
            image: BGR格式图像
            image_id: 图像ID，用于缓存
            category: 图像类别
            metadata: 图像元数据（包含时空信息）
            
        Returns:
            包含所有模态特征的字典
        """
        # 检查缓存
        if image_id and image_id in self.feature_cache:
            logger.debug(f"从缓存加载特征: {image_id}")
            return self.feature_cache[image_id]
        
        # 提取各模态特征
        features = {
            'category': category,
            'metadata': metadata or {}
        }
        
        # 提取颜色特征，确保返回numpy数组
        color_feat = self.color_extractor.extract(image)
        if not isinstance(color_feat, np.ndarray):
            logger.warning(f"颜色特征类型错误: {type(color_feat)}")
            # 创建默认的空特征
            color_feat = np.zeros(128, dtype=np.float32)
        features['color'] = color_feat
        
        # 提取SIFT特征，确保返回numpy数组
        try:
            # 更严格地检查codebook
            if not hasattr(self.sift_extractor, 'codebook') or self.sift_extractor.codebook is None:
                logger.warning(f"SIFT码本不存在，使用空特征")
                sift_feat = np.zeros(self.sift_extractor.codebook_size, dtype=np.float32)
            elif not isinstance(self.sift_extractor.codebook, np.ndarray):
                logger.error(f"SIFT码本类型错误: {type(self.sift_extractor.codebook)}，使用空特征")
                # 重置codebook为None，避免后续错误
                self.sift_extractor.codebook = None
                sift_feat = np.zeros(self.sift_extractor.codebook_size, dtype=np.float32)
            else:
                # 尝试提取特征
                try:
                    sift_feat = self.sift_extractor.extract(image)
                    if not isinstance(sift_feat, np.ndarray):
                        logger.warning(f"SIFT特征类型错误: {type(sift_feat)}")
                        sift_feat = np.zeros(self.sift_extractor.codebook_size, dtype=np.float32)
                except Exception as e:
                    logger.error(f"执行SIFT特征提取时出错: {e}")
                    sift_feat = np.zeros(self.sift_extractor.codebook_size, dtype=np.float32)
        except Exception as e:
            logger.error(f"处理SIFT特征提取时出错: {e}")
            sift_feat = np.zeros(self.sift_extractor.codebook_size, dtype=np.float32)
        features['sift'] = sift_feat
        
        # 提取纹理特征，确保返回numpy数组
        texture_feat = self.texture_extractor.extract(image)
        if not isinstance(texture_feat, np.ndarray):
            logger.warning(f"纹理特征类型错误: {type(texture_feat)}")
            # 创建默认的空特征
            texture_feat = np.zeros(64, dtype=np.float32)
        features['texture'] = texture_feat
        
        # 提取文本特征，确保返回numpy数组
        try:
            text_result = self.text_extractor.extract(image)
            # 从返回的字典中获取tfidf_vector
            if isinstance(text_result, dict) and 'tfidf_vector' in text_result:
                text_feat = text_result['tfidf_vector']
                # 检查tfidf_vector是否为numpy数组
                if not isinstance(text_feat, np.ndarray) or text_feat is None:
                    logger.warning(f"文本特征tfidf_vector类型错误或为None: {type(text_feat)}")
                    # 创建默认的空特征
                    text_feat = np.zeros(1000, dtype=np.float32)
            else:
                logger.warning(f"文本特征结果格式错误")
                text_feat = np.zeros(1000, dtype=np.float32)
        except Exception as e:
            logger.error(f"提取文本特征时出错: {e}")
            text_feat = np.zeros(1000, dtype=np.float32)
        features['text'] = text_feat
        
        # 缓存特征
        if image_id:
            self.feature_cache[image_id] = features
        
        return features
    
    def match(self, 
             query_features: Dict,
             gallery_features_list: List[Dict],
             top_k: int = 10) -> List[Tuple[int, float]]:
        """
        执行三级分层匹配
        
        Args:
            query_features: 查询特征
            gallery_features_list: 图库特征列表
            top_k: 返回前k个结果
            
        Returns:
            排序后的匹配结果列表 [(索引, 相似度)]
        """
        # 第一级：颜色粗筛选
        logger.info("执行第一级：颜色粗筛选")
        color_scores = []
        
        # 检查是否有FAISS索引可用
        if hasattr(self, 'color_index') and self.color_index is not None:
            logger.info("使用FAISS索引进行颜色粗筛选加速")
            try:
                # 使用FeatureUtils.search_faiss_index进行GPU加速检索
                # 我们需要检索更多候选，因为后面还要根据阈值过滤
                search_k = min(self.top_k_coarse * 5, len(gallery_features_list))
                distances, indices = FeatureUtils.search_faiss_index(
                    self.color_index,
                    [query_features['color']],
                    k=search_k,
                    batch_size=1000
                )
                
                # 转换距离为相似度
                # 对于L2距离，相似度可以用1/(1+distance)或其他适当的转换方法
                # 注意：这里假设FAISS使用的是L2距离，需要根据实际情况调整
                color_scores = []
                for idx_list, dist_list in zip(indices, distances):
                    for idx, dist in zip(idx_list, dist_list):
                        # 确保索引有效
                        if idx >= 0 and idx < len(gallery_features_list):
                            # 转换L2距离为相似度分数（0-1范围）
                            # 对于cosine相似度，需要特殊处理
                            # 这里我们使用直接计算的方式确保正确性
                            sim = self.color_extractor.compute_similarity(
                                query_features['color'],
                                gallery_features_list[idx]['color'],
                                method='cosine'
                            )
                            color_scores.append((int(idx), sim))
            except Exception as e:
                logger.error(f"使用FAISS索引检索失败: {e}")
                # 回退到原始方法
                color_scores = []
                for i, gallery_features in enumerate(gallery_features_list):
                    color_sim = self.color_extractor.compute_similarity(
                        query_features['color'], 
                        gallery_features['color'],
                        method='cosine'
                    )
                    color_scores.append((i, color_sim))
        else:
            # 原始方法：遍历所有图库特征
            for i, gallery_features in enumerate(gallery_features_list):
                color_sim = self.color_extractor.compute_similarity(
                    query_features['color'], 
                    gallery_features['color'],
                    method='cosine'
                )
                color_scores.append((i, color_sim))
        
        # 过滤并保留top_k_coarse个候选
        color_scores.sort(key=lambda x: x[1], reverse=True)
        
        # 首先尝试使用原始阈值
        coarse_candidates = [(i, score) for i, score in color_scores 
                            if score >= self.color_threshold][:self.top_k_coarse]
        
        # 如果没有候选通过阈值，降低阈值重试
        if not coarse_candidates:
            logger.warning(f"没有通过颜色粗筛选的候选（阈值：{self.color_threshold}），尝试降低阈值")
            # 降低阈值到0.3，确保至少有一些候选通过
            reduced_threshold = min(self.color_threshold, 0.3)
            coarse_candidates = [(i, score) for i, score in color_scores[:self.top_k_coarse]]
            
            if coarse_candidates:
                logger.info(f"使用降低的标准，保留前{len(coarse_candidates)}个候选")
            else:
                logger.warning("即使降低标准也没有找到候选")
                return []
        
        logger.info(f"颜色粗筛选后保留 {len(coarse_candidates)} 个候选")
        
        # 第二级：时空约束过滤
        logger.info("执行第二级：时空约束过滤")
        spatial_temporal_scores = []
        
        query_metadata = query_features['metadata']
        
        for idx, _ in coarse_candidates:
            gallery_metadata = gallery_features_list[idx]['metadata']
            
            # 计算空间相似度
            spatial_sim = self._compute_spatial_similarity(
                query_metadata, gallery_metadata
            )
            
            # 计算时间相似度
            temporal_sim = self._compute_temporal_similarity(
                query_metadata, gallery_metadata
            )
            
            # 综合时空相似度
            spatio_temporal_sim = (
                self.spatial_weight * spatial_sim + 
                self.temporal_weight * temporal_sim
            )
            
            spatial_temporal_scores.append((idx, spatio_temporal_sim))
        
        # 第三级：多模态精确匹配
        logger.info("执行第三级：多模态精确匹配")
        final_scores = []
        
        query_category = query_features.get('category', 'other')
        
        for idx, st_score in spatial_temporal_scores:
            gallery_features = gallery_features_list[idx]
            gallery_category = gallery_features.get('category', 'other')
            
            # 使用查询和图库中出现次数较多的类别
            if query_category in self.category_weights and gallery_category in self.category_weights:
                # 计算两个类别的权重平均
                weights = self._average_category_weights(query_category, gallery_category)
            elif query_category in self.category_weights:
                weights = self.category_weights[query_category]
            elif gallery_category in self.category_weights:
                weights = self.category_weights[gallery_category]
            else:
                # 使用安全的默认权重，确保即使'other'键不存在也能正常工作
                weights = self.category_weights.get('other', {
                    'color': 0.25,
                    'sift': 0.25,
                    'texture': 0.25,
                    'text': 0.15,
                    'spatio_temporal': 0.1
                })
            
            # 计算各模态相似度
            color_sim = self.color_extractor.compute_similarity(
                query_features['color'], 
                gallery_features['color'],
                method='cosine'
            )
            
            sift_sim = self.sift_extractor.compute_similarity(
                query_features['sift'],
                gallery_features['sift'],
                method='cosine'
            )
            
            texture_sim = self.texture_extractor.compute_similarity(
                query_features['texture'],
                gallery_features['texture'],
                method='cosine'
            )
            
            text_sim = self.text_extractor.compute_similarity(
                query_features['text'],
                gallery_features['text'],
                method='combined'
            )
            
            # 多模态融合
            final_sim = self._fuse_features(
                color_sim, sift_sim, texture_sim, text_sim, st_score,
                weights, self.fusion_method
            )
            
            final_scores.append((idx, final_sim))
        
        # 排序并返回结果
        final_scores.sort(key=lambda x: x[1], reverse=True)
        
        return final_scores[:top_k]
    
    def _average_category_weights(self, cat1: str, cat2: str) -> Dict:
        """
        计算两个类别的权重平均值
        """
        weights1 = self.category_weights[cat1]
        weights2 = self.category_weights[cat2]
        
        avg_weights = {}
        for key in weights1:
            avg_weights[key] = (weights1[key] + weights2[key]) / 2
        
        return avg_weights
    
    def _compute_spatial_similarity(self, 
                                   meta1: Dict,
                                   meta2: Dict) -> float:
        """
        计算空间相似度
        """
        # 获取位置信息
        loc1 = meta1.get('location', {})
        loc2 = meta2.get('location', {})
        
        # 如果没有位置信息，返回默认相似度
        if not loc1 or not loc2:
            return 0.5
        
        # 计算坐标距离
        try:
            # GPS坐标
            if 'latitude' in loc1 and 'longitude' in loc1 and \
               'latitude' in loc2 and 'longitude' in loc2:
                # 使用Haversine公式计算距离
                lat1, lon1 = loc1['latitude'], loc1['longitude']
                lat2, lon2 = loc2['latitude'], loc2['longitude']
                
                # 地球半径（km）
                R = 6371.0
                
                # 转换为弧度
                lat1_rad = np.radians(lat1)
                lon1_rad = np.radians(lon1)
                lat2_rad = np.radians(lat2)
                lon2_rad = np.radians(lon2)
                
                # Haversine公式
                dlon = lon2_rad - lon1_rad
                dlat = lat2_rad - lat1_rad
                a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
                c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
                distance = R * c
                
                # 转换为相似度（距离越近相似度越高）
                # 1km以内相似度为1，10km以外相似度为0
                similarity = max(0.0, 1.0 - (distance / 10.0))
                return similarity
            
            # 相对位置
            elif 'x' in loc1 and 'y' in loc1 and 'z' in loc1 and \
                 'x' in loc2 and 'y' in loc2 and 'z' in loc2:
                # 计算欧几里得距离
                pos1 = np.array([loc1['x'], loc1['y'], loc1['z']])
                pos2 = np.array([loc2['x'], loc2['y'], loc2['z']])
                distance = np.linalg.norm(pos1 - pos2)
                
                # 转换为相似度
                similarity = 1.0 / (1.0 + distance)
                return similarity
            
            # 区域信息
            elif 'area' in loc1 and 'area' in loc2:
                # 相同区域相似度为1
                return 1.0 if loc1['area'] == loc2['area'] else 0.3
                
        except Exception as e:
            logger.warning(f"计算空间相似度失败: {e}")
        
        return 0.5
    
    def _compute_temporal_similarity(self, 
                                    meta1: Dict,
                                    meta2: Dict) -> float:
        """
        计算时间相似度
        """
        # 获取时间信息
        time1 = meta1.get('time', None)
        time2 = meta2.get('time', None)
        
        # 如果没有时间信息，返回默认相似度
        if time1 is None or time2 is None:
            return 0.5
        
        try:
            # 转换为时间戳
            from datetime import datetime
            
            # 处理不同格式的时间
            if isinstance(time1, str):
                time1 = datetime.fromisoformat(time1.replace('Z', '+00:00'))
            if isinstance(time2, str):
                time2 = datetime.fromisoformat(time2.replace('Z', '+00:00'))
            
            # 计算时间差（小时）
            time_diff = abs((time2 - time1).total_seconds() / 3600)
            
            # 转换为相似度
            # 1小时内相似度为1，24小时以外相似度为0
            similarity = max(0.0, 1.0 - (time_diff / 24.0))
            
            return similarity
            
        except Exception as e:
            logger.warning(f"计算时间相似度失败: {e}")
        
        return 0.5
    
    def _fuse_features(self, 
                      color_sim: float,
                      sift_sim: float,
                      texture_sim: float,
                      text_sim: float,
                      spatio_temporal_sim: float,
                      weights: Dict,
                      method: str = 'weighted_sum') -> float:
        """
        多模态特征融合
        """
        if method == 'weighted_sum':
            # 加权求和
            fused = (
                weights['color'] * color_sim +
                weights['sift'] * sift_sim +
                weights['texture'] * texture_sim +
                weights['text'] * text_sim +
                weights['spatio_temporal'] * spatio_temporal_sim
            )
        
        elif method == 'max':
            # 取最大值
            fused = max(color_sim, sift_sim, texture_sim, text_sim, spatio_temporal_sim)
        
        elif method == 'min':
            # 取最小值
            fused = min(color_sim, sift_sim, texture_sim, text_sim, spatio_temporal_sim)
        
        elif method == 'mean':
            # 取平均值
            fused = (color_sim + sift_sim + texture_sim + text_sim + spatio_temporal_sim) / 5
        
        else:
            raise ValueError(f"不支持的融合方法: {method}")
        
        # 确保在0-1范围内
        fused = max(0.0, min(1.0, fused))
        
        return fused
    
    def clear_cache(self):
        """
        清空特征缓存
        """
        self.feature_cache.clear()
        logger.info("特征缓存已清空")
    
    def save(self, path: str):
        """
        保存模型配置
        
        Args:
            path: 保存路径
        """
        import pickle
        
        # 保存配置信息
        config = {
            'color_threshold': self.color_threshold,
            'top_k_coarse': self.top_k_coarse,
            'spatial_weight': self.spatial_weight,
            'temporal_weight': self.temporal_weight,
            'fusion_method': self.fusion_method,
            'category_weights': self.category_weights
        }
        
        save_dir = Path(path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        with open(path, 'wb') as f:
            pickle.dump(config, f)
        
        logger.info(f"模型配置已保存到: {path}")
    
    def load(self, path: str):
        """
        加载模型配置
        
        Args:
            path: 加载路径
        """
        import pickle
        
        with open(path, 'rb') as f:
            config = pickle.load(f)
        
        # 恢复配置
        self.color_threshold = config.get('color_threshold', self.color_threshold)
        self.top_k_coarse = config.get('top_k_coarse', self.top_k_coarse)
        self.spatial_weight = config.get('spatial_weight', self.spatial_weight)
        self.temporal_weight = config.get('temporal_weight', self.temporal_weight)
        self.fusion_method = config.get('fusion_method', self.fusion_method)
        self.category_weights = config.get('category_weights', self.category_weights)
        
        logger.info(f"模型配置已从 {path} 加载")
    
    def build_index(self, gallery_dataset: List[Dict]):
        """
        构建图库索引
        
        Args:
            gallery_dataset: 图库数据集，每个元素是包含'image_path'或'path'和可选'category'的字典
        """
        import time
        import os
        
        self.gallery_dataset = gallery_dataset
        self.gallery_features_list = []
        
        logger.info(f"开始构建图库索引，共 {len(gallery_dataset)} 张图像")
        start_time = time.time()
        
        # 首先收集所有图像，用于构建SIFT码本
        all_images = []
        valid_items = []
        
        logger.info("收集图像数据...")
        for i, item in enumerate(gallery_dataset):
            try:
                # 支持'path'或'image_path'键
                image_path = item.get('path', '') or item.get('image_path', '')
                if not os.path.exists(image_path):
                    logger.warning(f"图像不存在: {image_path}")
                    continue
                
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"无法读取图像: {image_path}")
                    continue
                
                all_images.append(image)
                valid_items.append((i, item, image, image_path))
                
                # 进度日志
                if (i + 1) % 50 == 0:
                    logger.info(f"已收集 {i+1}/{len(gallery_dataset)} 张图像")
            except Exception as e:
                logger.error(f"收集图像时出错: {e}")
        
        # 为SIFT特征提取器构建码本（如果需要）
        if hasattr(self, 'sift_extractor') and hasattr(self.sift_extractor, 'codebook') and self.sift_extractor.codebook is None and all_images:
            logger.info("为SIFT特征提取器构建码本...")
            try:
                # 使用收集的图像构建码本
                codebook_path = os.path.join('./cache', 'sift_codebook.pkl')
                os.makedirs('./cache', exist_ok=True)
                self.sift_extractor.build_codebook(all_images, save_path=codebook_path)
                logger.info("SIFT码本构建完成")
            except Exception as e:
                logger.error(f"构建SIFT码本失败: {e}")
        
        # 处理每张图像，提取特征
        logger.info("提取图像特征...")
        for i, item, image, image_path in valid_items:
            try:
                # 提取特征
                features = self.extract_features(
                    image=image,
                    image_id=str(i),
                    category=item.get('category', 'other'),
                    metadata=item.get('metadata', {})
                )
                
                self.gallery_features_list.append(features)
                
                # 进度日志
                if (len(self.gallery_features_list) % 10 == 0):
                    logger.info(f"已处理 {len(self.gallery_features_list)}/{len(valid_items)} 张图像")
            except Exception as e:
                logger.error(f"处理图像 {image_path} 时出错: {e}")
        
        # 构建FAISS索引用于加速检索
        try:
            # 为各模态特征构建FAISS索引
            if hasattr(self, 'use_gpu') and self.use_gpu:
                logger.info(f"为各模态特征构建GPU加速的FAISS索引（设备：{self.gpu_device if hasattr(self, 'gpu_device') else 0}）")
            else:
                logger.info("为各模态特征构建CPU FAISS索引")
            
            # 检查是否有任何特征
            if not self.gallery_features_list:
                logger.warning("没有提取到任何特征，跳过索引构建")
            else:
                # 使用FeatureUtils.build_faiss_index方法构建索引
                # 为颜色特征构建索引，添加额外的类型检查
                color_features = []
                valid_indices = []
                for i, f in enumerate(self.gallery_features_list):
                    color_feat = f.get('color')
                    if isinstance(color_feat, np.ndarray) and color_feat.ndim == 1:
                        color_features.append(color_feat)
                        valid_indices.append(i)
                    else:
                        logger.warning(f"索引 {i} 的颜色特征无效，类型: {type(color_feat)}")
                
                logger.info(f"构建颜色索引，有效特征数: {len(color_features)}/{len(self.gallery_features_list)}")
                
                # 只有当有有效特征时才构建索引
                if color_features:
                    try:
                        self.color_index = FeatureUtils.build_faiss_index(
                            color_features, 
                            index_type='flat',  # 可以根据数据规模选择'ivf'或'hnsw'
                            use_gpu=hasattr(self, 'use_gpu') and self.use_gpu,
                            gpu_device=hasattr(self, 'gpu_device') and self.gpu_device
                        )
                        # 保存有效的索引映射
                        self.valid_indices = valid_indices
                        
                        # 为SIFT特征构建索引，使用相同的有效索引
                        try:
                            sift_features = [self.gallery_features_list[i]['sift'] for i in valid_indices]
                            logger.info(f"构建SIFT索引，有效特征数: {len(sift_features)}")
                            self.sift_index = FeatureUtils.build_faiss_index(
                                sift_features,
                                index_type='flat',
                                use_gpu=hasattr(self, 'use_gpu') and self.use_gpu,
                                gpu_device=hasattr(self, 'gpu_device') and self.gpu_device
                            )
                        except Exception as e:
                            logger.error(f"构建SIFT索引时出错: {e}")
                            self.sift_index = None
                        
                        # 为纹理特征构建索引，使用相同的有效索引
                        try:
                            texture_features = [self.gallery_features_list[i]['texture'] for i in valid_indices]
                            logger.info(f"构建纹理索引，有效特征数: {len(texture_features)}")
                            self.texture_index = FeatureUtils.build_faiss_index(
                                texture_features,
                                index_type='flat',
                                use_gpu=hasattr(self, 'use_gpu') and self.use_gpu,
                                gpu_device=hasattr(self, 'gpu_device') and self.gpu_device
                            )
                        except Exception as e:
                            logger.error(f"构建纹理索引时出错: {e}")
                            self.texture_index = None
                        
                        # 更新gallery_features_list，只保留有效的特征
                        if valid_indices:
                            self.gallery_features_list = [self.gallery_features_list[i] for i in valid_indices]
                            logger.info(f"更新图库特征列表，保留 {len(self.gallery_features_list)} 个有效特征")
                        
                        logger.info("FAISS索引构建完成")
                    except Exception as e:
                        logger.error(f"构建索引过程中出错: {e}")
                else:
                    logger.warning("没有有效的颜色特征，跳过索引构建")
        except ImportError as e:
            logger.warning(f"缺少必要的库，跳过索引构建: {e}")
        except Exception as e:
            logger.error(f"构建FAISS索引时出错: {e}")
        
        logger.info(f"图库索引构建完成，成功处理 {len(self.gallery_features_list)} 张图像")
        logger.info(f"耗时: {time.time() - start_time:.2f}秒")
        
        # 确保即使索引构建失败也能继续运行
        return True
    
    def search(self, query_image_path: str, category: str = 'other', k: int = 10) -> List[Dict]:
        """
        执行检索
        
        Args:
            query_image_path: 查询图像路径
            category: 查询图像类别
            k: 返回前k个结果
            
        Returns:
            检索结果列表，每个元素包含'index'和'score'字段
        """
        # 检查索引是否已构建
        if not hasattr(self, 'gallery_features_list') or not self.gallery_features_list:
            raise ValueError("请先调用build_index方法构建索引")
        
        # 读取查询图像
        query_image = cv2.imread(query_image_path)
        if query_image is None:
            raise ValueError(f"无法读取查询图像: {query_image_path}")
        
        # 提取查询特征
        query_features = self.extract_features(
            image=query_image,
            category=category
        )
        
        # 执行匹配
        match_results = self.match(
            query_features=query_features,
            gallery_features_list=self.gallery_features_list,
            top_k=k
        )
        
        # 格式化结果
        results = []
        for idx, score in match_results:
            results.append({
                'index': idx,
                'score': float(score)
            })
        
        return results
    
    def save_index(self, path: str):
        """
        保存索引
        
        Args:
            path: 保存路径
        """
        import pickle
        
        index_data = {
            'gallery_dataset': self.gallery_dataset,
            'gallery_features_list': self.gallery_features_list
        }
        
        # 创建目录
        save_dir = Path(path).parent
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存索引
        with open(path, 'wb') as f:
            pickle.dump(index_data, f)
        
        logger.info(f"索引已保存到: {path}")
    
    def load_index(self, path: str):
        """
        加载索引
        
        Args:
            path: 加载路径
        """
        import pickle
        
        with open(path, 'rb') as f:
            index_data = pickle.load(f)
        
        self.gallery_dataset = index_data['gallery_dataset']
        self.gallery_features_list = index_data['gallery_features_list']
        
        logger.info(f"索引已从 {path} 加载，包含 {len(self.gallery_features_list)} 个图库项")