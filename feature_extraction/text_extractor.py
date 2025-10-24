import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
import logging
import re
from collections import Counter
import jieba
import jieba.analyse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path

# 尝试导入cupy，用于GPU加速
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    logger.warning("未找到cupy库，GPU加速功能不可用")
    CUPY_AVAILABLE = False

logger = logging.getLogger(__name__)

class TextFeatureExtractor:
    """
    文字特征提取器
    
    基于PaddleOCR实现，支持：
    - 多语言文字识别
    - 图像预处理
    - 结构化文字提取
    - 多种文字相似度计算方法
    """
    
    def __init__(self, 
                 lang: str = 'ch',
                 use_gpu: bool = False,
                 gpu_device: int = 0,
                 det_db_thresh: float = 0.3,
                 det_db_box_thresh: float = 0.6,
                 det_db_unclip_ratio: float = 1.5,
                 rec_batch_num: int = 6,
                 use_angle_cls: bool = True,
                 use_text_filter: bool = True,
                 max_text_length: int = 1024,
                 vectorizer_path: Optional[str] = None):
        """
        Args:
            lang: 语言类型 ('ch', 'en', 'french', 'german', 'japan', 'korean', 'chinese_cht', 'ta', 'te', 'ka', 'latin', 'arabic', 'cyrillic', 'devanagari', 'tamil')
            use_gpu: 是否使用GPU
            det_db_thresh: 文本检测阈值
            det_db_box_thresh: 文本框阈值
            det_db_unclip_ratio: 文本框膨胀系数
            rec_batch_num: 识别批次大小
            use_angle_cls: 是否使用方向分类器
            use_text_filter: 是否过滤无效文本
            max_text_length: 最大文本长度
            vectorizer_path: 预训练TF-IDF向量器路径
        """
        self.lang = lang
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        self.det_db_thresh = det_db_thresh
        self.det_db_box_thresh = det_db_box_thresh
        self.det_db_unclip_ratio = det_db_unclip_ratio
        self.rec_batch_num = rec_batch_num
        self.use_angle_cls = use_angle_cls
        self.use_text_filter = use_text_filter
        self.max_text_length = max_text_length
        self.vectorizer_path = vectorizer_path
        
        # 检查GPU加速可用性
        self.gpu_available = self.use_gpu and CUPY_AVAILABLE
        
        # 初始化OCR
        self.ocr = self._init_ocr()
        
        # 停用词列表
        self.stopwords = self._load_stopwords()
        
        # 初始化TF-IDF向量器
        self.vectorizer = None
        if vectorizer_path:
            self._load_vectorizer(vectorizer_path)
        else:
            self._init_vectorizer()
        
        logger.info(f"初始化文字特征提取器: lang={lang}, use_gpu={use_gpu}")
    
    def _init_ocr(self):
        """
        初始化PaddleOCR
        """
        try:
            from paddleocr import PaddleOCR
            
            ocr = PaddleOCR(
                lang=self.lang,
                use_angle_cls=self.use_angle_cls,
                use_gpu=self.use_gpu,
                det_db_thresh=self.det_db_thresh,
                det_db_box_thresh=self.det_db_box_thresh,
                det_db_unclip_ratio=self.det_db_unclip_ratio,
                rec_batch_num=self.rec_batch_num
            )
            return ocr
        except ImportError:
            logger.warning("未安装PaddleOCR，将使用备用的文本处理功能")
            return None
        except Exception as e:
            logger.error(f"初始化OCR失败: {e}")
            return None
    
    def _init_vectorizer(self):
        """
        初始化TF-IDF向量器
        """
        self.vectorizer = TfidfVectorizer(
            tokenizer=self._tokenize,
            stop_words=self.stopwords,
            max_features=1000,
            ngram_range=(1, 2)
        )
    
    def _load_vectorizer(self, path: str):
        """
        加载预训练的TF-IDF向量器
        """
        try:
            with open(path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            logger.info(f"已加载TF-IDF向量器: {path}")
        except Exception as e:
            logger.error(f"加载TF-IDF向量器失败: {e}")
            self._init_vectorizer()
    
    def _load_stopwords(self) -> List[str]:
        """
        加载停用词列表
        """
        # 基础停用词
        stopwords = [
            '的', '了', '和', '是', '在', '有', '我', '他', '她', '它', '们',
            '这', '那', '个', '一', '不', '就', '都', '而', '及', '与', '或',
            '但', '如果', '因为', '所以', '虽然', '但是', '对于', '关于', 'with',
            'the', 'a', 'an', 'and', 'or', 'but', 'if', 'because', 'so', 'for',
            'of', 'in', 'on', 'to', 'at', 'by'
        ]
        
        # 尝试加载更多停用词
        try:
            import pandas as pd
            # 可以从文件或其他来源加载更多停用词
        except ImportError:
            pass
        
        return stopwords
    
    def _tokenize(self, text: str) -> List[str]:
        """
        分词函数
        """
        if self.lang in ['ch', 'chinese_cht']:
            # 中文分词
            return [word for word in jieba.cut(text) if word.strip()]
        else:
            # 英文等按空格分词
            return re.findall(r'\b\w+\b', text.lower())
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        图像预处理
        """
        # 转换为灰度图
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # 自适应阈值化
        thresh = cv2.adaptiveThreshold(
            gray, 255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            11, 2
        )
        
        # 降噪
        denoised = cv2.medianBlur(thresh, 3)
        
        # 形态学操作，增强文本
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        processed = cv2.morphologyEx(denoised, cv2.MORPH_CLOSE, kernel)
        
        return processed
    
    def _filter_text(self, text: str) -> bool:
        """
        过滤无效文本
        """
        # 检查文本长度
        if len(text) < 2:
            return False
        
        # 检查是否只包含数字或特殊字符
        if re.match(r'^[\d\s`~!@#$%^&*()_\-+=\[\]{}|;:\'",.<>/?]+$', text):
            return False
        
        # 检查是否包含足够的有效字符
        valid_chars = re.findall(r'[\w\u4e00-\u9fa5]', text)
        if len(valid_chars) / len(text) < 0.5:
            return False
        
        return True
    
    def extract(self, image: np.ndarray) -> Dict:
        """
        提取图像中的文字特征
        
        Args:
            image: BGR格式图像
            
        Returns:
            包含文字特征的字典
        """
        # 初始化结果，将tfidf_vector设为默认零向量
        result = {
            'raw_text': '',
            'texts': [],
            'confidences': [],
            'positions': [],
            'keywords': [],
            'tfidf_vector': np.zeros(100, dtype=np.float32)  # 默认零向量
        }
        
        try:
            # 图像预处理
            processed_image = self._preprocess_image(image)
            
            # 使用OCR识别文字
            if self.ocr is not None:
                ocr_result = self.ocr.ocr(processed_image, cls=self.use_angle_cls)
                
                if ocr_result and len(ocr_result) > 0 and ocr_result[0] is not None:
                    # 提取文字信息
                    for line in ocr_result[0]:
                        position = line[0]  # 坐标
                        text_info = line[1]  # 文字和置信度
                        
                        if isinstance(text_info, (list, tuple)) and len(text_info) >= 2:
                            text = text_info[0]
                            confidence = text_info[1]
                            
                            # 过滤无效文本
                            if self.use_text_filter and not self._filter_text(text):
                                continue
                            
                            result['texts'].append(text)
                            result['confidences'].append(confidence)
                            result['positions'].append(position)
            
            # 合并所有文字
            result['raw_text'] = ' '.join(result['texts'][:self.max_text_length])
            
            # 提取关键词
            if result['raw_text']:
                result['keywords'] = self._extract_keywords(result['raw_text'])
            
            # 计算TF-IDF向量，确保始终返回有效的向量
            if self.vectorizer:
                try:
                    if result['raw_text']:
                        # 如果向量器未拟合，先拟合
                        if not hasattr(self.vectorizer, 'vocabulary_'):
                            self.vectorizer.fit([result['raw_text']])
                        
                        tfidf_vector = self.vectorizer.transform([result['raw_text']]).toarray()[0]
                        result['tfidf_vector'] = tfidf_vector
                    else:
                        # 如果没有文本，根据向量器维度返回零向量
                        if hasattr(self.vectorizer, 'vocabulary_'):
                            vector_dim = len(self.vectorizer.vocabulary_)
                            result['tfidf_vector'] = np.zeros(vector_dim, dtype=np.float32)
                            logger.debug(f"未检测到有效文本，返回全零向量，维度: {vector_dim}")
                except Exception as e:
                    logger.warning(f"计算TF-IDF向量失败: {e}")
                    # 出错时确保仍返回有效向量
                    if hasattr(self.vectorizer, 'vocabulary_'):
                        vector_dim = len(self.vectorizer.vocabulary_)
                        result['tfidf_vector'] = np.zeros(vector_dim, dtype=np.float32)
                    else:
                        result['tfidf_vector'] = np.zeros(100, dtype=np.float32)
        
        except Exception as e:
            logger.error(f"文字特征提取失败: {e}")
        
        return result
    
    def _extract_keywords(self, text: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """
        提取关键词
        """
        try:
            if self.lang in ['ch', 'chinese_cht']:
                # 中文关键词提取
                keywords = jieba.analyse.extract_tags(
                    text, 
                    topK=top_k, 
                    withWeight=True, 
                    allowPOS=()
                )
            else:
                # 英文等关键词提取
                words = self._tokenize(text)
                # 过滤停用词
                words = [word for word in words if word not in self.stopwords]
                # 计算词频
                word_counts = Counter(words)
                # 取前top_k个
                keywords = word_counts.most_common(top_k)
                # 归一化权重
                total = sum(count for _, count in keywords)
                if total > 0:
                    keywords = [(word, count/total) for word, count in keywords]
            
            return keywords
        except Exception as e:
            logger.warning(f"关键词提取失败: {e}")
            return []
    
    def train_vectorizer(self, texts: List[str], save_path: Optional[str] = None):
        """
        训练TF-IDF向量器
        
        Args:
            texts: 训练文本列表
            save_path: 保存路径
        """
        try:
            self.vectorizer.fit(texts)
            
            if save_path:
                save_dir = Path(save_path).parent
                save_dir.mkdir(parents=True, exist_ok=True)
                
                with open(save_path, 'wb') as f:
                    pickle.dump(self.vectorizer, f)
                logger.info(f"TF-IDF向量器已保存到: {save_path}")
            
            logger.info(f"TF-IDF向量器训练完成，词汇表大小: {len(self.vectorizer.vocabulary_)}")
        except Exception as e:
            logger.error(f"训练TF-IDF向量器失败: {e}")
    
    def compute_similarity(self, 
                          text_feature1: Union[Dict, np.ndarray], 
                          text_feature2: Union[Dict, np.ndarray], 
                          method: str = 'combined') -> float:
        """
        计算两个文字特征之间的相似度
        
        Args:
            text_feature1: 文字特征1 (字典或numpy数组)
            text_feature2: 文字特征2 (字典或numpy数组)
            method: 相似度计算方法 ('tfidf', 'keyword', 'edit_distance', 'combined')
            
        Returns:
            similarity: 相似度得分 (0-1)
        """
        method = method.lower()
        
        # 确保text_feature1和text_feature2是字典类型
        if isinstance(text_feature1, np.ndarray):
            # 如果是numpy数组，转换为字典格式
            text_feature1 = {'tfidf_vector': text_feature1, 'keywords': [], 'raw_text': ''}
        
        if isinstance(text_feature2, np.ndarray):
            # 如果是numpy数组，转换为字典格式
            text_feature2 = {'tfidf_vector': text_feature2, 'keywords': [], 'raw_text': ''}
        
        # 确保两个参数都是字典
        if not isinstance(text_feature1, dict) or not isinstance(text_feature2, dict):
            logger.warning(f"计算文字相似度时输入类型错误: {type(text_feature1)}, {type(text_feature2)}")
            return 0.0
        
        try:
            if method == 'tfidf':
                # TF-IDF余弦相似度
                tfidf1 = text_feature1.get('tfidf_vector', None)
                tfidf2 = text_feature2.get('tfidf_vector', None)
                
                if tfidf1 is not None and tfidf2 is not None:
                    if self.gpu_available:
                        similarity = self._compute_tfidf_similarity_gpu(tfidf1, tfidf2)
                    else:
                        similarity = self._compute_tfidf_similarity_cpu(tfidf1, tfidf2)
                else:
                    similarity = 0.0
            
            elif method == 'keyword':
                # 关键词匹配相似度
                keywords1 = dict(text_feature1.get('keywords', []))
                keywords2 = dict(text_feature2.get('keywords', []))
                
                # 计算关键词交集
                common_words = set(keywords1.keys()) & set(keywords2.keys())
                if not common_words:
                    similarity = 0.0
                else:
                    # 加权求和
                    similarity_sum = 0.0
                    weight_sum = 0.0
                    
                    for word in common_words:
                        weight1 = keywords1[word]
                        weight2 = keywords2[word]
                        similarity_sum += (weight1 * weight2)
                        weight_sum += (weight1 + weight2)
                    
                    if weight_sum > 0:
                        similarity = 2 * similarity_sum / weight_sum
                    else:
                        similarity = 0.0
            
            elif method == 'edit_distance':
                # 编辑距离相似度
                text1 = text_feature1.get('raw_text', '')
                text2 = text_feature2.get('raw_text', '')
                
                if not text1 or not text2:
                    similarity = 0.0 if text1 != text2 else 1.0
                else:
                    # 计算编辑距离
                    try:
                        import Levenshtein
                        distance = Levenshtein.distance(text1, text2)
                        max_len = max(len(text1), len(text2))
                        similarity = 1 - (distance / max_len)
                    except ImportError:
                        similarity = 0.0
            
            elif method == 'combined':
                # 组合多种相似度
                similarities = []
                weights = []
                
                # TF-IDF相似度
                tfidf1 = text_feature1.get('tfidf_vector', None)
                tfidf2 = text_feature2.get('tfidf_vector', None)
                
                if tfidf1 is not None and tfidf2 is not None:
                    try:
                        vec1 = tfidf1.reshape(1, -1)
                        vec2 = tfidf2.reshape(1, -1)
                        
                        # 处理维度不匹配
                        if vec1.shape[1] != vec2.shape[1]:
                            max_dim = max(vec1.shape[1], vec2.shape[1])
                            padded1 = np.zeros((1, max_dim))
                            padded2 = np.zeros((1, max_dim))
                            padded1[:, :vec1.shape[1]] = vec1
                            padded2[:, :vec2.shape[1]] = vec2
                            vec1, vec2 = padded1, padded2
                        
                        tfidf_sim = cosine_similarity(vec1, vec2)[0, 0]
                        similarities.append(tfidf_sim)
                        weights.append(0.5)
                    except Exception as e:
                        logger.warning(f"计算TF-IDF相似度时出错: {e}")
                
                # 关键词相似度
                keywords1 = dict(text_feature1.get('keywords', []))
                keywords2 = dict(text_feature2.get('keywords', []))
                common_words = set(keywords1.keys()) & set(keywords2.keys())
                
                if common_words:
                    try:
                        similarity_sum = 0.0
                        weight_sum = 0.0
                        
                        for word in common_words:
                            weight1 = keywords1[word]
                            weight2 = keywords2[word]
                            similarity_sum += (weight1 * weight2)
                            weight_sum += (weight1 + weight2)
                        
                        if weight_sum > 0:
                            keyword_sim = 2 * similarity_sum / weight_sum
                            similarities.append(keyword_sim)
                            weights.append(0.3)
                    except Exception as e:
                        logger.warning(f"计算关键词相似度时出错: {e}")
                
                # 编辑距离相似度
                text1 = text_feature1.get('raw_text', '')
                text2 = text_feature2.get('raw_text', '')
                
                if text1 and text2:
                    try:
                        import Levenshtein
                        distance = Levenshtein.distance(text1, text2)
                        max_len = max(len(text1), len(text2))
                        edit_sim = 1 - (distance / max_len)
                        similarities.append(edit_sim)
                        weights.append(0.2)
                    except (ImportError, Exception) as e:
                        logger.warning(f"计算编辑距离相似度时出错: {e}")
                
                # 计算加权平均
                if similarities and weights:
                    try:
                        weights = np.array(weights)
                        weights = weights / np.sum(weights)  # 归一化权重
                        similarity = np.sum(np.array(similarities) * weights)
                    except Exception as e:
                        logger.warning(f"计算加权平均时出错: {e}")
                        similarity = 0.0
                else:
                    # 如果没有文字，返回一个默认值
                    similarity = 0.0
            
            else:
                raise ValueError(f"不支持的相似度方法: {method}")
        
        except Exception as e:
            logger.error(f"计算文字相似度时出错: {e}")
            similarity = 0.0
        
        # 确保在0-1范围内
        similarity = max(0.0, min(1.0, similarity))
        return similarity
    
    def _compute_tfidf_similarity_cpu(self, vector1, vector2):
        """
        CPU版本的TF-IDF余弦相似度计算
        """
        vec1 = vector1.reshape(1, -1)
        vec2 = vector2.reshape(1, -1)
        
        # 处理维度不匹配
        if vec1.shape[1] != vec2.shape[1]:
            max_dim = max(vec1.shape[1], vec2.shape[1])
            padded1 = np.zeros((1, max_dim))
            padded2 = np.zeros((1, max_dim))
            padded1[:, :vec1.shape[1]] = vec1
            padded2[:, :vec2.shape[1]] = vec2
            vec1, vec2 = padded1, padded2
        
        return cosine_similarity(vec1, vec2)[0, 0]
    
    def _compute_tfidf_similarity_gpu(self, vector1, vector2):
        """
        GPU版本的TF-IDF余弦相似度计算，使用cupy加速
        """
        try:
            # 转移数据到GPU
            vec1_gpu = cp.array(vector1).reshape(1, -1)
            vec2_gpu = cp.array(vector2).reshape(1, -1)
            
            # 处理维度不匹配
            if vec1_gpu.shape[1] != vec2_gpu.shape[1]:
                max_dim = max(vec1_gpu.shape[1], vec2_gpu.shape[1])
                padded1_gpu = cp.zeros((1, max_dim))
                padded2_gpu = cp.zeros((1, max_dim))
                padded1_gpu[:, :vec1_gpu.shape[1]] = vec1_gpu
                padded2_gpu[:, :vec2_gpu.shape[1]] = vec2_gpu
                vec1_gpu, vec2_gpu = padded1_gpu, padded2_gpu
            
            # 计算余弦相似度
            # 归一化向量
            norm1 = cp.linalg.norm(vec1_gpu)
            norm2 = cp.linalg.norm(vec2_gpu)
            
            if norm1 == 0 or norm2 == 0:
                return 0.0
            
            vec1_normalized = vec1_gpu / norm1
            vec2_normalized = vec2_gpu / norm2
            
            # 点积计算相似度
            similarity = cp.sum(vec1_normalized * vec2_normalized)
            
            # 将结果转回CPU
            return float(similarity)
        except Exception as e:
            logger.error(f"GPU TF-IDF相似度计算失败: {e}，回退到CPU计算")
            # 回退到CPU计算
            return self._compute_tfidf_similarity_cpu(vector1, vector2)
        # 确保在0-1范围内
        similarity = max(0.0, min(1.0, similarity))
        
        return similarity