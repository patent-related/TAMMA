import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging

from feature_extraction.color_extractor import ColorFeatureExtractor

logger = logging.getLogger(__name__)

class ColorOnlyMatcherComplete:
    """
    仅使用颜色特征的基线匹配算法
    
    支持空间金字塔特征提取和多种距离度量方法
    """
    
    def __init__(self, 
                 color_space: str = 'hsv',
                 bins: Tuple[int, int, int] = (8, 8, 8),
                 spatial_pyramid_levels: int = 2,
                 similarity_method: str = 'cosine',
                 normalize: bool = True):
        """
        Args:
            color_space: 颜色空间 ('hsv', 'rgb', 'lab', 'ycrcb')
            bins: 直方图分箱数量
            spatial_pyramid_levels: 空间金字塔层级
            similarity_method: 相似度计算方法 ('cosine', 'l1', 'l2', 'chi2', 'bhattacharyya')
            normalize: 是否归一化特征
        """
        # 初始化颜色特征提取器
        # 确保bins是一个三元组
        if isinstance(bins, int):
            h_bins = s_bins = v_bins = bins
        elif isinstance(bins, tuple) and len(bins) == 3:
            h_bins, s_bins, v_bins = bins
        else:
            raise ValueError(f"bins参数必须是整数或三元组，得到: {type(bins)}")
        
        self.color_extractor = ColorFeatureExtractor(
            color_space=color_space,
            h_bins=h_bins,
            s_bins=s_bins,
            v_bins=v_bins,
            pyramid_levels=spatial_pyramid_levels,
            use_spatial_pyramid=spatial_pyramid_levels > 0
        )
        
        self.similarity_method = similarity_method
        
        # 特征缓存
        self.feature_cache = {}
        
        logger.info(f"初始化仅颜色特征匹配器: {color_space}, "
                   f"bins={bins}, "
                   f"pyramid={spatial_pyramid_levels}, "
                   f"method={similarity_method}")
    
    def extract_features(self, 
                        image: np.ndarray,
                        image_id: Optional[str] = None,
                        metadata: Optional[Dict] = None) -> Dict:
        """
        提取图像的颜色特征
        
        Args:
            image: BGR格式图像
            image_id: 图像ID，用于缓存
            metadata: 图像元数据（暂不使用）
            
        Returns:
            包含颜色特征的字典
        """
        # 检查缓存
        if image_id and image_id in self.feature_cache:
            logger.debug(f"从缓存加载颜色特征: {image_id}")
            return self.feature_cache[image_id]
        
        # 提取颜色特征
        color_feature = self.color_extractor.extract(image)
        
        features = {
            'color': color_feature,
            'metadata': metadata or {}
        }
        
        # 缓存特征
        if image_id:
            self.feature_cache[image_id] = features
        
        return features
    
    def match(self, 
             query_features: Dict,
             gallery_features_list: List[Dict],
             top_k: int = 10) -> List[Tuple[int, float]]:
        """
        执行颜色特征匹配
        
        Args:
            query_features: 查询特征
            gallery_features_list: 图库特征列表
            top_k: 返回前k个结果
            
        Returns:
            排序后的匹配结果列表 [(索引, 相似度)]
        """
        # 计算所有图库图像的颜色相似度
        scores = []
        
        for i, gallery_features in enumerate(gallery_features_list):
            # 计算颜色相似度
            similarity = self.color_extractor.compute_similarity(
                query_features['color'],
                gallery_features['color'],
                method=self.similarity_method
            )
            
            scores.append((i, similarity))
        
        # 按相似度降序排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前top_k个结果
        return scores[:top_k]
    
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
            'color_space': self.color_extractor.color_space,
            'bins': self.color_extractor.bins,
            'spatial_pyramid_levels': self.color_extractor.spatial_pyramid_levels,
            'similarity_method': self.similarity_method,
            'normalize': self.color_extractor.normalize
        }
    
    def set_config(self, config: Dict):
        """
        设置配置信息
        
        Args:
            config: 配置字典
        """
        # 更新颜色提取器配置
        if 'color_space' in config:
            self.color_extractor.color_space = config['color_space']
        if 'bins' in config:
            self.color_extractor.bins = config['bins']
        if 'spatial_pyramid_levels' in config:
            self.color_extractor.spatial_pyramid_levels = config['spatial_pyramid_levels']
        if 'normalize' in config:
            self.color_extractor.normalize = config['normalize']
        
        # 更新相似度方法
        if 'similarity_method' in config:
            self.similarity_method = config['similarity_method']
        
        # 清空缓存，因为配置已更改
        self.clear_cache()
        
        logger.info(f"更新配置: {config}")
    
    def evaluate_similarity_method(self,
                                 query_features: Dict,
                                 gallery_features_list: List[Dict],
                                 ground_truth_indices: List[int]) -> Dict:
        """
        评估不同相似度方法的性能
        
        Args:
            query_features: 查询特征
            gallery_features_list: 图库特征列表
            ground_truth_indices: 真实正样本索引列表
            
        Returns:
            不同方法的性能评估结果
        """
        methods = ['cosine', 'l1', 'l2', 'chi2', 'bhattacharyya']
        results = {}
        
        for method in methods:
            # 保存当前方法
            current_method = self.similarity_method
            
            # 设置测试方法
            self.similarity_method = method
            
            # 执行匹配
            matches = self.match(query_features, gallery_features_list, top_k=len(gallery_features_list))
            
            # 计算评估指标
            indices = [idx for idx, _ in matches]
            scores = [score for _, score in matches]
            
            # 计算平均精度
            precision = 0.0
            relevant_count = 0
            for i, idx in enumerate(indices):
                if idx in ground_truth_indices:
                    relevant_count += 1
                    precision += relevant_count / (i + 1)
            
            if relevant_count > 0:
                average_precision = precision / relevant_count
            else:
                average_precision = 0.0
            
            # 计算召回率
            recall = relevant_count / len(ground_truth_indices) if ground_truth_indices else 0.0
            
            # 计算F1分数
            if precision + recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
            else:
                f1 = 0.0
            
            results[method] = {
                'average_precision': average_precision,
                'recall': recall,
                'f1': f1,
                'matches': matches[:10]  # 保存前10个匹配结果
            }
            
            # 恢复当前方法
            self.similarity_method = current_method
        
        return results
    
    def build_index(self, gallery_dataset):
        """
        为评估流程构建索引

        Args:
            gallery_dataset: 图库数据集
        """
        logger.info(f"为 {len(gallery_dataset)} 个图库图像构建索引")
        
        # 提取所有图库图像的特征
        self.gallery_features_list = []
        for i, item in enumerate(gallery_dataset):
            if (i + 1) % 50 == 0:
                logger.info(f"处理图库图像 {i+1}/{len(gallery_dataset)}")
            
            # 尝试多种方式获取图像数据
            image = None
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
    
    def optimize_parameters(self,
                          query_features_list: List[Dict],
                          gallery_features_list: List[Dict],
                          ground_truth_indices_list: List[List[int]]) -> Dict:
        """
        优化参数配置

        Args:
            query_features_list: 查询特征列表
            gallery_features_list: 图库特征列表
            ground_truth_indices_list: 每个查询的真实正样本索引列表

        Returns:
            最佳参数配置
        """
        # 参数搜索空间
        param_grid = {
            'color_space': ['hsv', 'lab', 'ycrcb', 'rgb'],
            'bins': [(8, 8, 8), (16, 16, 16), (8, 12, 3)],  # HSV空间常用的分箱策略
            'spatial_pyramid_levels': [0, 1, 2, 3],
            'similarity_method': ['cosine', 'chi2', 'bhattacharyya']
        }
        
        best_config = None
        best_map = 0.0
        
        # 简单网格搜索（实际应用中可以使用更高效的方法）
        import itertools
        
        total_combinations = (
            len(param_grid['color_space']) *
            len(param_grid['bins']) *
            len(param_grid['spatial_pyramid_levels']) *
            len(param_grid['similarity_method'])
        )
        
        logger.info(f"开始参数优化，共 {total_combinations} 种组合")
        
        combination_count = 0
        
        for color_space, bins, pyramid_levels, similarity_method in itertools.product(
            param_grid['color_space'],
            param_grid['bins'],
            param_grid['spatial_pyramid_levels'],
            param_grid['similarity_method']
        ):
            combination_count += 1
            
            # 跳过无效的颜色空间和bins组合
            if color_space in ['rgb', 'lab', 'ycrcb'] and len(bins) != 3:
                continue
            
            logger.info(f"测试组合 {combination_count}/{total_combinations}: "
                       f"{color_space}, bins={bins}, pyramid={pyramid_levels}, method={similarity_method}")
            
            # 设置临时配置
            config = {
                'color_space': color_space,
                'bins': bins,
                'spatial_pyramid_levels': pyramid_levels,
                'similarity_method': similarity_method
            }
            self.set_config(config)
            
            # 清空缓存
            self.clear_cache()
            
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
            
            # 更新最佳配置
            if map_score > best_map:
                best_map = map_score
                best_config = config.copy()
                logger.info(f"  找到更好的配置! MAP = {best_map:.4f}")
        
        if best_config:
            logger.info(f"参数优化完成，最佳配置: {best_config}, 最佳MAP: {best_map:.4f}")
            # 恢复最佳配置
            self.set_config(best_config)
        else:
            logger.warning("参数优化失败，未找到有效配置")
        
        return best_config