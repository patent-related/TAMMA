import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from torchvision import transforms, models
from typing import List, Dict, Tuple, Optional, Union
import logging
from tqdm import tqdm

logger = logging.getLogger(__name__)

class DeepLearningMatcherComplete:
    """
    深度学习特征匹配算法
    
    支持ResNet50、VGG16、EfficientNet、ViT等预训练模型
    """
    
    def __init__(self, 
                 model_name: str = 'resnet50',
                 feature_dim: int = 2048,
                 pretrained: bool = True,
                 batch_size: int = 32,
                 use_gpu: bool = True,
                 normalize_features: bool = True,
                 embedding_layer: Optional[str] = None,
                 resize_size: Tuple[int, int] = (224, 224)):
        """
        Args:
            model_name: 预训练模型名称
            feature_dim: 特征维度
            pretrained: 是否使用预训练权重
            batch_size: 批处理大小
            use_gpu: 是否使用GPU
            normalize_features: 是否归一化特征
            embedding_layer: 用于提取特征的层名称
            resize_size: 图像调整大小
        """
        self.model_name = model_name.lower()
        self.feature_dim = feature_dim
        self.pretrained = pretrained
        self.batch_size = batch_size
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.normalize_features = normalize_features
        self.embedding_layer = embedding_layer
        self.resize_size = resize_size
        
        # 设备
        self.device = torch.device('cuda' if self.use_gpu else 'cpu')
        logger.info(f"使用设备: {self.device}")
        
        # 初始化模型
        self.model = self._init_model()
        
        # 图像预处理变换
        self.transform = self._get_transform()
        
        # 特征缓存
        self.feature_cache = {}
        
        logger.info(f"初始化深度学习匹配器: {model_name}, pretrained={pretrained}, "
                   f"feature_dim={feature_dim}, batch_size={batch_size}")
    
    def _init_model(self) -> nn.Module:
        """
        初始化预训练模型
        """
        try:
            if self.model_name == 'resnet50':
                model = models.resnet50(pretrained=self.pretrained)
                # 移除最后的全连接层
                model = nn.Sequential(*list(model.children())[:-1])
                self.feature_dim = 2048
                
            elif self.model_name == 'resnet101':
                model = models.resnet101(pretrained=self.pretrained)
                model = nn.Sequential(*list(model.children())[:-1])
                self.feature_dim = 2048
                
            elif self.model_name == 'vgg16':
                model = models.vgg16(pretrained=self.pretrained)
                # 移除最后的分类层
                model = nn.Sequential(*list(model.children())[:-1])
                self.feature_dim = 512 * 7 * 7
                
            elif self.model_name == 'vgg19':
                model = models.vgg19(pretrained=self.pretrained)
                model = nn.Sequential(*list(model.children())[:-1])
                self.feature_dim = 512 * 7 * 7
                
            elif self.model_name == 'efficientnet_b0':
                model = models.efficientnet_b0(pretrained=self.pretrained)
                # 移除最后的分类层
                model.classifier = nn.Identity()
                self.feature_dim = 1280
                
            elif self.model_name == 'efficientnet_b4':
                model = models.efficientnet_b4(pretrained=self.pretrained)
                model.classifier = nn.Identity()
                self.feature_dim = 1792
                
            elif self.model_name == 'vit_b_16':
                model = models.vit_b_16(pretrained=self.pretrained)
                # 提取cls token的输出
                model = nn.Sequential(*list(model.children())[:-1])
                self.feature_dim = 768
                
            else:
                raise ValueError(f"不支持的模型: {self.model_name}")
            
            # 移动到设备
            model = model.to(self.device)
            
            # 设置为评估模式
            model.eval()
            
            logger.info(f"成功加载模型: {self.model_name}")
            return model
            
        except Exception as e:
            logger.error(f"加载模型失败: {e}")
            raise
    
    def _get_transform(self) -> transforms.Compose:
        """
        获取图像预处理变换
        """
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(self.resize_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet均值
                std=[0.229, 0.224, 0.225]    # ImageNet标准差
            )
        ])
    
    def preprocess_image(self, image: np.ndarray) -> torch.Tensor:
        """
        预处理单个图像
        
        Args:
            image: BGR格式图像
            
        Returns:
            预处理后的张量
        """
        # 转换为RGB格式
        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 应用变换
        image_tensor = self.transform(image)
        
        return image_tensor
    
    def extract_features(self, 
                        image: np.ndarray,
                        image_id: Optional[str] = None,
                        batch_process: bool = False) -> np.ndarray:
        """
        提取单张图像的特征
        
        Args:
            image: BGR格式图像
            image_id: 图像ID，用于缓存
            batch_process: 是否批量处理
            
        Returns:
            提取的特征向量
        """
        # 检查缓存
        if image_id and image_id in self.feature_cache:
            logger.debug(f"从缓存加载特征: {image_id}")
            return self.feature_cache[image_id]
        
        # 预处理
        image_tensor = self.preprocess_image(image)
        image_tensor = image_tensor.unsqueeze(0).to(self.device)
        
        # 提取特征
        with torch.no_grad():
            features = self.model(image_tensor)
        
        # 调整特征形状
        features = features.squeeze().cpu().numpy()
        
        # 展平为向量
        if features.ndim > 1:
            features = features.flatten()
        
        # 归一化
        if self.normalize_features:
            norm = np.linalg.norm(features)
            if norm > 1e-10:
                features = features / norm
        
        # 缓存特征
        if image_id:
            self.feature_cache[image_id] = features
        
        return features
    
    def extract_features_batch(self, 
                             images: List[np.ndarray],
                             image_ids: Optional[List[str]] = None) -> List[np.ndarray]:
        """
        批量提取特征
        
        Args:
            images: 图像列表
            image_ids: 图像ID列表，用于缓存
            
        Returns:
            特征向量列表
        """
        # 预处理图像
        image_tensors = []
        cache_indices = {}
        cached_features = {}
        uncached_indices = []
        
        for i, image in enumerate(images):
            # 检查缓存
            if image_ids and image_ids[i] and image_ids[i] in self.feature_cache:
                cached_features[i] = self.feature_cache[image_ids[i]]
                cache_indices[i] = True
            else:
                image_tensor = self.preprocess_image(image)
                image_tensors.append(image_tensor)
                uncached_indices.append(i)
                cache_indices[i] = False
        
        # 创建数据集
        if image_tensors:
            dataset = TensorDataset(torch.stack(image_tensors))
            dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
            
            # 提取未缓存图像的特征
            features_list = []
            with torch.no_grad():
                for batch in tqdm(dataloader, desc="提取特征"):
                    batch = batch[0].to(self.device)
                    batch_features = self.model(batch)
                    
                    # 调整形状并归一化
                    for feat in batch_features:
                        feat = feat.squeeze().cpu().numpy()
                        if feat.ndim > 1:
                            feat = feat.flatten()
                        
                        if self.normalize_features:
                            norm = np.linalg.norm(feat)
                            if norm > 1e-10:
                                feat = feat / norm
                        
                        features_list.append(feat)
            
            # 填充结果
            results = [None] * len(images)
            
            # 填充缓存特征
            for idx, feature in cached_features.items():
                results[idx] = feature
            
            # 填充新提取的特征
            for i, idx in enumerate(uncached_indices):
                results[idx] = features_list[i]
                # 更新缓存
                if image_ids and image_ids[idx]:
                    self.feature_cache[image_ids[idx]] = features_list[i]
            
            return results
        else:
            # 全部都是缓存特征
            results = [None] * len(images)
            for idx, feature in cached_features.items():
                results[idx] = feature
            return results
    
    def compute_similarity(self, 
                          feature1: np.ndarray,
                          feature2: np.ndarray,
                          method: str = 'cosine') -> float:
        """
        计算两个特征向量的相似度
        
        Args:
            feature1: 特征向量1
            feature2: 特征向量2
            method: 相似度计算方法 ('cosine', 'euclidean', 'dot')
            
        Returns:
            相似度值
        """
        if method == 'cosine':
            # 余弦相似度
            return np.dot(feature1, feature2)
            
        elif method == 'euclidean':
            # 欧氏距离（转换为相似度）
            dist = np.linalg.norm(feature1 - feature2)
            # 归一化到[0, 1]区间
            return 1.0 / (1.0 + dist)
            
        elif method == 'dot':
            # 点积
            return np.dot(feature1, feature2)
            
        else:
            raise ValueError(f"不支持的相似度方法: {method}")
    
    def match(self, 
             query_feature: np.ndarray,
             gallery_features: List[np.ndarray],
             top_k: int = 10,
             similarity_method: str = 'cosine') -> List[Tuple[int, float]]:
        """
        执行特征匹配
        
        Args:
            query_feature: 查询特征
            gallery_features: 图库特征列表
            top_k: 返回前k个结果
            similarity_method: 相似度计算方法
            
        Returns:
            排序后的匹配结果列表 [(索引, 相似度)]
        """
        # 计算所有图库图像的相似度
        scores = []
        
        for i, gallery_feature in enumerate(gallery_features):
            similarity = self.compute_similarity(
                query_feature, gallery_feature, method=similarity_method
            )
            scores.append((i, similarity))
        
        # 按相似度降序排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 返回前top_k个结果
        return scores[:top_k]
    
    def batch_match(self, 
                   query_features: List[np.ndarray],
                   gallery_features: List[np.ndarray],
                   top_k: int = 10,
                   similarity_method: str = 'cosine') -> List[List[Tuple[int, float]]]:
        """
        批量执行匹配
        
        Args:
            query_features: 查询特征列表
            gallery_features: 图库特征列表
            top_k: 返回前k个结果
            similarity_method: 相似度计算方法
            
        Returns:
            每个查询的匹配结果列表
        """
        results = []
        
        # 将特征列表转换为numpy数组以进行批量计算
        gallery_array = np.array(gallery_features)
        
        for i, query_feature in enumerate(query_features):
            if (i + 1) % 10 == 0:
                logger.info(f"处理查询 {i+1}/{len(query_features)}")
            
            # 批量计算相似度
            if similarity_method == 'cosine':
                # 余弦相似度 = 点积（因为特征已归一化）
                similarities = np.dot(gallery_array, query_feature)
            
            elif similarity_method == 'euclidean':
                # 欧氏距离
                differences = gallery_array - query_feature
                dists = np.linalg.norm(differences, axis=1)
                similarities = 1.0 / (1.0 + dists)
            
            elif similarity_method == 'dot':
                # 点积
                similarities = np.dot(gallery_array, query_feature)
            
            else:
                raise ValueError(f"不支持的相似度方法: {similarity_method}")
            
            # 获取排序索引
            sorted_indices = np.argsort(similarities)[::-1][:top_k]
            
            # 构建结果
            result = [(int(idx), float(similarities[idx])) for idx in sorted_indices]
            results.append(result)
        
        return results
    
    def fine_tune(self, 
                 images: List[np.ndarray],
                 labels: List[int],
                 epochs: int = 10,
                 learning_rate: float = 1e-4,
                 validation_split: float = 0.2,
                 freeze_base: bool = True) -> Dict:
        """
        微调预训练模型
        
        Args:
            images: 训练图像列表
            labels: 标签列表
            epochs: 训练轮数
            learning_rate: 学习率
            validation_split: 验证集比例
            freeze_base: 是否冻结基础模型
            
        Returns:
            训练历史记录
        """
        logger.info(f"开始微调模型: {self.model_name}, epochs={epochs}, "
                   f"lr={learning_rate}, freeze_base={freeze_base}")
        
        # 准备数据集
        all_indices = np.arange(len(images))
        np.random.shuffle(all_indices)
        
        val_size = int(len(images) * validation_split)
        train_indices = all_indices[val_size:]
        val_indices = all_indices[:val_size]
        
        # 预处理训练和验证数据
        train_tensors = []
        train_labels = []
        
        for idx in train_indices:
            img_tensor = self.preprocess_image(images[idx])
            train_tensors.append(img_tensor)
            train_labels.append(labels[idx])
        
        val_tensors = []
        val_labels = []
        
        for idx in val_indices:
            img_tensor = self.preprocess_image(images[idx])
            val_tensors.append(img_tensor)
            val_labels.append(labels[idx])
        
        # 创建数据集和数据加载器
        train_dataset = TensorDataset(
            torch.stack(train_tensors),
            torch.tensor(train_labels, dtype=torch.long)
        )
        
        val_dataset = TensorDataset(
            torch.stack(val_tensors),
            torch.tensor(val_labels, dtype=torch.long)
        )
        
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)
        
        # 冻结/解冻模型参数
        for param in self.model.parameters():
            param.requires_grad = not freeze_base
        
        # 添加分类头
        num_classes = len(set(labels))
        
        # 根据模型类型添加不同的分类头
        if self.model_name in ['resnet50', 'resnet101']:
            classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.feature_dim, 512),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            
        elif self.model_name in ['vgg16', 'vgg19']:
            classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.feature_dim, 4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, 4096),
                nn.ReLU(),
                nn.Dropout(0.5),
                nn.Linear(4096, num_classes)
            )
            
        elif self.model_name.startswith('efficientnet'):
            classifier = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(self.feature_dim, num_classes)
            )
            
        elif self.model_name == 'vit_b_16':
            classifier = nn.Sequential(
                nn.Linear(self.feature_dim, num_classes)
            )
            
        else:
            classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(self.feature_dim, num_classes)
            )
        
        classifier = classifier.to(self.device)
        
        # 组合完整模型
        full_model = nn.Sequential(
            self.model,
            classifier
        )
        full_model = full_model.to(self.device)
        
        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(full_model.parameters(), lr=learning_rate)
        
        # 学习率调度器
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=3, factor=0.5
        )
        
        # 训练历史
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        # 开始训练
        for epoch in range(epochs):
            full_model.train()
            train_loss = 0.0
            correct = 0
            total = 0
            
            for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
                images, targets = batch
                images, targets = images.to(self.device), targets.to(self.device)
                
                # 前向传播
                outputs = full_model(images)
                loss = criterion(outputs, targets)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # 统计
                train_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            # 计算训练指标
            train_epoch_loss = train_loss / len(train_loader.dataset)
            train_epoch_acc = correct / total
            
            # 验证
            full_model.eval()
            val_loss = 0.0
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    images, targets = batch
                    images, targets = images.to(self.device), targets.to(self.device)
                    
                    outputs = full_model(images)
                    loss = criterion(outputs, targets)
                    
                    val_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    total += targets.size(0)
                    correct += predicted.eq(targets).sum().item()
            
            # 计算验证指标
            val_epoch_loss = val_loss / len(val_loader.dataset)
            val_epoch_acc = correct / total
            
            # 更新学习率
            scheduler.step(val_epoch_loss)
            
            # 保存历史
            history['train_loss'].append(train_epoch_loss)
            history['train_acc'].append(train_epoch_acc)
            history['val_loss'].append(val_epoch_loss)
            history['val_acc'].append(val_epoch_acc)
            
            logger.info(f"Epoch {epoch+1}/{epochs}: "
                       f"Train Loss: {train_epoch_loss:.4f}, Train Acc: {train_epoch_acc:.4f}, "
                       f"Val Loss: {val_epoch_loss:.4f}, Val Acc: {val_epoch_acc:.4f}")
        
        logger.info("模型微调完成")
        return history
    
    def clear_cache(self):
        """
        清空特征缓存
        """
        self.feature_cache.clear()
        logger.info("特征缓存已清空")
    
    def save_model(self, path: str):
        """
        保存模型
        
        Args:
            path: 保存路径
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.get_config()
        }, path)
        logger.info(f"模型已保存到: {path}")
    
    def load_model(self, path: str):
        """
        加载模型
        
        Args:
            path: 模型路径
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # 更新配置
        if 'config' in checkpoint:
            config = checkpoint['config']
            self.model_name = config.get('model_name', self.model_name)
            self.feature_dim = config.get('feature_dim', self.feature_dim)
            self.resize_size = config.get('resize_size', self.resize_size)
            self.normalize_features = config.get('normalize_features', self.normalize_features)
        
        # 设置为评估模式
        self.model.eval()
        logger.info(f"模型已从: {path} 加载")
    
    def get_config(self) -> Dict:
        """
        获取配置信息
        
        Returns:
            配置字典
        """
        return {
            'model_name': self.model_name,
            'feature_dim': self.feature_dim,
            'pretrained': self.pretrained,
            'batch_size': self.batch_size,
            'use_gpu': self.use_gpu,
            'normalize_features': self.normalize_features,
            'embedding_layer': self.embedding_layer,
            'resize_size': self.resize_size
        }