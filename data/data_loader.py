import os
import json
import numpy as np
import pandas as pd
import cv2
import random
import torch
from typing import List, Dict, Tuple, Optional, Any, Callable
import logging
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import albumentations as A

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class MultiModalDatasetLoader:
    """
    多模态数据集加载器
    
    支持图像、文本、元数据等多模态数据的加载和预处理
    """
    
    def __init__(self, 
                 dataset_dir: str,
                 image_size: Tuple[int, int] = (224, 224),
                 batch_size: int = 32,
                 shuffle: bool = True,
                 num_workers: int = 4,
                 pin_memory: bool = True,
                 random_seed: int = 42):
        """
        Args:
            dataset_dir: 数据集目录
            image_size: 图像大小
            batch_size: 批次大小
            shuffle: 是否打乱数据
            num_workers: 工作线程数
            pin_memory: 是否使用内存锁定
            random_seed: 随机种子
        """
        self.dataset_dir = dataset_dir
        self.image_size = image_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        
        # 设置随机种子
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # 初始化图像转换
        self.transform = self._get_default_transforms()
        self.augmentation = self._get_default_augmentation()
        
        # 数据集信息
        self.dataset_info = {}
        self.loaded_data = {
            'images': {},  # 图像数据
            'texts': {},   # 文本数据
            'metadata': {}, # 元数据
            'annotations': []  # 标注信息
        }
        
        logger.info(f"初始化数据加载器: dataset_dir={dataset_dir}, image_size={image_size}, batch_size={batch_size}")
    
    def _get_default_transforms(self):
        """
        获取默认的图像转换
        """
        return transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def _get_default_augmentation(self):
        """
        获取默认的数据增强配置
        """
        return A.Compose([
            A.RandomRotate90(),
            A.HorizontalFlip(),
            A.OneOf([
                A.GaussianBlur(),
                A.MotionBlur(),
            ], p=0.3),
            A.OneOf([
                A.HueSaturationValue(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.Resize(*self.image_size)
        ])
    
    def load_dataset(self, 
                    split: str = 'all',
                    load_images: bool = True,
                    load_texts: bool = True,
                    load_metadata: bool = True,
                    max_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        加载数据集
        
        Args:
            split: 数据集分割 (train/val/test/all)
            load_images: 是否加载图像
            load_texts: 是否加载文本
            load_metadata: 是否加载元数据
            max_samples: 最大样本数
            
        Returns:
            加载的数据集信息
        """
        logger.info(f"加载数据集: split={split}, load_images={load_images}, load_texts={load_texts}, load_metadata={load_metadata}")
        
        # 重置加载的数据
        self.loaded_data = {
            'images': {},
            'texts': {},
            'metadata': {},
            'annotations': []
        }
        
        # 检查数据集目录
        if not os.path.exists(self.dataset_dir):
            logger.error(f"数据集目录不存在: {self.dataset_dir}")
            return self.loaded_data
        
        # 加载数据集信息
        info_file = os.path.join(self.dataset_dir, 'dataset_info.json')
        if os.path.exists(info_file):
            try:
                with open(info_file, 'r', encoding='utf-8') as f:
                    self.dataset_info = json.load(f)
                logger.info(f"数据集信息: {self.dataset_info}")
            except Exception as e:
                logger.error(f"加载数据集信息失败: {e}")
        
        # 加载标注文件
        annotations = self._load_annotations(split)
        if not annotations:
            logger.warning("没有找到标注文件，尝试从图像目录加载")
            annotations = self._load_from_images()
        
        # 限制样本数
        if max_samples and len(annotations) > max_samples:
            annotations = annotations[:max_samples]
            logger.info(f"限制样本数为: {max_samples}")
        
        self.loaded_data['annotations'] = annotations
        
        # 加载多模态数据
        for idx, item in enumerate(annotations):
            item_id = item.get('id', str(idx))
            
            # 加载图像
            if load_images and 'image_path' in item:
                image = self._load_image(item['image_path'])
                if image is not None:
                    self.loaded_data['images'][item_id] = image
            
            # 加载文本
            if load_texts and 'text' in item:
                self.loaded_data['texts'][item_id] = item['text']
            
            # 加载元数据
            if load_metadata:
                metadata = {k: v for k, v in item.items() if k not in ['id', 'image_path', 'text']}
                if metadata:
                    self.loaded_data['metadata'][item_id] = metadata
            
            # 进度显示
            if (idx + 1) % 100 == 0 or (idx + 1) == len(annotations):
                logger.info(f"已加载 {idx + 1}/{len(annotations)} 个样本")
        
        logger.info(f"数据集加载完成: 总样本数={len(annotations)}, 图像数={len(self.loaded_data['images'])}, 文本数={len(self.loaded_data['texts'])}")
        return self.loaded_data
    
    def _load_annotations(self, split: str) -> List[Dict[str, Any]]:
        """
        加载标注文件
        
        Args:
            split: 数据集分割
            
        Returns:
            标注列表
        """
        # 尝试不同的标注文件格式
        annotation_files = []
        
        # 检查特定分割的标注文件
        if split != 'all':
            annotation_files.append(os.path.join(self.dataset_dir, f'{split}.json'))
        
        # 检查通用标注文件
        annotation_files.extend([
            os.path.join(self.dataset_dir, 'annotations.json'),
            os.path.join(self.dataset_dir, 'metadata.json'),
            os.path.join(self.dataset_dir, 'dataset.json')
        ])
        
        # 尝试加载标注文件
        for ann_file in annotation_files:
            if os.path.exists(ann_file):
                try:
                    with open(ann_file, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    
                    # 处理不同格式的标注
                    if isinstance(data, list):
                        annotations = data
                    elif isinstance(data, dict):
                        if split in data:
                            annotations = data[split]
                        elif 'annotations' in data:
                            annotations = data['annotations']
                        else:
                            annotations = [data]
                    else:
                        continue
                    
                    # 确保每个标注都有必要的字段
                    processed_annotations = []
                    for i, ann in enumerate(annotations):
                        if not isinstance(ann, dict):
                            continue
                        
                        # 确保有ID
                        if 'id' not in ann:
                            ann['id'] = str(i)
                        
                        # 确保图像路径是绝对路径
                        if 'image_path' in ann:
                            if not os.path.isabs(ann['image_path']):
                                ann['image_path'] = os.path.join(self.dataset_dir, ann['image_path'])
                        
                        processed_annotations.append(ann)
                    
                    logger.info(f"从 {ann_file} 加载了 {len(processed_annotations)} 个标注")
                    return processed_annotations
                    
                except Exception as e:
                    logger.error(f"加载标注文件 {ann_file} 失败: {e}")
                    continue
        
        return []
    
    def _load_from_images(self) -> List[Dict[str, Any]]:
        """
        直接从图像目录加载
        
        Returns:
            标注列表
        """
        annotations = []
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
        
        # 遍历图像目录
        for root, _, files in os.walk(self.dataset_dir):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in image_extensions:
                    image_path = os.path.join(root, file)
                    item = {
                        'id': os.path.splitext(file)[0],
                        'image_path': image_path,
                        'filename': file,
                        'category': os.path.basename(root) if root != self.dataset_dir else 'unknown'
                    }
                    annotations.append(item)
        
        logger.info(f"从图像目录加载了 {len(annotations)} 个样本")
        return annotations
    
    def _load_image(self, image_path: str) -> Optional[Image.Image]:
        """
        加载图像
        
        Args:
            image_path: 图像路径
            
        Returns:
            图像对象
        """
        try:
            # 检查路径是否存在
            if not os.path.exists(image_path):
                # 尝试在数据集目录中查找
                alt_path = os.path.join(self.dataset_dir, image_path)
                if not os.path.exists(alt_path):
                    logger.warning(f"图像文件不存在: {image_path}")
                    return None
                image_path = alt_path
            
            # 使用PIL加载图像
            image = Image.open(image_path).convert('RGB')
            return image
            
        except Exception as e:
            logger.error(f"加载图像 {image_path} 失败: {e}")
            return None
    
    def get_data_loader(self, 
                       mode: str = 'standard',
                       augmentation: bool = False,
                       **kwargs) -> DataLoader:
        """
        获取数据加载器
        
        Args:
            mode: 数据加载模式 (standard/pytorch/transformer)
            augmentation: 是否使用数据增强
            **kwargs: 额外参数
            
        Returns:
            DataLoader对象
        """
        # 创建数据集实例
        if mode == 'pytorch':
            dataset = PyTorchMultiModalDataset(
                self.loaded_data,
                transform=self.augmentation if augmentation else self.transform,
                **kwargs
            )
        else:
            dataset = StandardMultiModalDataset(
                self.loaded_data,
                image_size=self.image_size,
                augmentation=augmentation,
                **kwargs
            )
        
        # 创建DataLoader
        data_loader = DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=dataset.collate_fn if hasattr(dataset, 'collate_fn') else None
        )
        
        logger.info(f"创建数据加载器: mode={mode}, dataset_size={len(dataset)}, batch_size={self.batch_size}")
        return data_loader
    
    def create_query_gallery_split(self, 
                                 query_ratio: float = 0.2,
                                 category_balanced: bool = True) -> Tuple[List[Dict], List[Dict]]:
        """
        创建查询集和图库集的分割
        
        Args:
            query_ratio: 查询集比例
            category_balanced: 是否按类别平衡分割
            
        Returns:
            (查询集, 图库集)
        """
        annotations = self.loaded_data['annotations']
        if not annotations:
            logger.error("没有加载的标注数据")
            return [], []
        
        if category_balanced:
            # 按类别分组
            category_groups = {}
            for item in annotations:
                category = item.get('category', 'unknown')
                if category not in category_groups:
                    category_groups[category] = []
                category_groups[category].append(item)
            
            # 从每个类别中采样
            query_set = []
            gallery_set = []
            
            for category, items in category_groups.items():
                # 打乱列表
                random.shuffle(items)
                
                # 计算查询集大小
                query_size = max(1, int(len(items) * query_ratio))
                
                # 分割
                query_set.extend(items[:query_size])
                gallery_set.extend(items[query_size:])
            
        else:
            # 随机分割
            random.shuffle(annotations)
            query_size = max(1, int(len(annotations) * query_ratio))
            
            query_set = annotations[:query_size]
            gallery_set = annotations[query_size:]
        
        logger.info(f"创建查询集和图库集: 查询集大小={len(query_set)}, 图库集大小={len(gallery_set)}")
        return query_set, gallery_set
    
    def preprocess_images(self, 
                         images: List[np.ndarray],
                         normalize: bool = True) -> List[np.ndarray]:
        """
        预处理图像列表
        
        Args:
            images: 图像列表
            normalize: 是否归一化
            
        Returns:
            处理后的图像列表
        """
        processed = []
        
        for img in images:
            # 调整大小
            if img.shape[:2] != self.image_size:
                img = cv2.resize(img, self.image_size)
            
            # 归一化
            if normalize:
                img = img.astype(np.float32) / 255.0
                
                # 应用ImageNet均值和标准差
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = (img - mean) / std
            
            processed.append(img)
        
        return processed
    
    def save_processed_dataset(self, 
                              output_dir: str,
                              format: str = 'json') -> bool:
        """
        保存处理后的数据集
        
        Args:
            output_dir: 输出目录
            format: 保存格式 (json/npy/pickle)
            
        Returns:
            是否保存成功
        """
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            # 保存标注
            annotations_path = os.path.join(output_dir, 'annotations.json')
            with open(annotations_path, 'w', encoding='utf-8') as f:
                json.dump(self.loaded_data['annotations'], f, ensure_ascii=False, indent=2)
            
            # 保存数据集信息
            info_path = os.path.join(output_dir, 'dataset_info.json')
            with open(info_path, 'w', encoding='utf-8') as f:
                json.dump({
                    **self.dataset_info,
                    'processed_time': pd.Timestamp.now().isoformat(),
                    'num_samples': len(self.loaded_data['annotations'])
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"处理后的数据集已保存到: {output_dir}")
            return True
            
        except Exception as e:
            logger.error(f"保存处理后的数据集失败: {e}")
            return False
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取数据集统计信息
        
        Returns:
            统计信息
        """
        stats = {
            'total_samples': len(self.loaded_data['annotations']),
            'images_loaded': len(self.loaded_data['images']),
            'texts_loaded': len(self.loaded_data['texts']),
            'metadata_loaded': len(self.loaded_data['metadata'])
        }
        
        # 类别统计
        categories = {}
        for item in self.loaded_data['annotations']:
            category = item.get('category', 'unknown')
            categories[category] = categories.get(category, 0) + 1
        stats['categories'] = categories
        
        # 其他统计
        if self.loaded_data['annotations']:
            # 检查是否有文本长度统计
            text_lengths = []
            for item in self.loaded_data['annotations']:
                if 'text' in item:
                    text_lengths.append(len(item['text']))
            
            if text_lengths:
                stats['text_statistics'] = {
                    'min_length': min(text_lengths),
                    'max_length': max(text_lengths),
                    'mean_length': np.mean(text_lengths),
                    'median_length': np.median(text_lengths)
                }
        
        return stats
    
    def show_sample(self, index: int = 0) -> Dict[str, Any]:
        """
        显示样本信息
        
        Args:
            index: 样本索引
            
        Returns:
            样本信息
        """
        if index >= len(self.loaded_data['annotations']):
            logger.error(f"索引超出范围: {index} >= {len(self.loaded_data['annotations'])}")
            return {}
        
        item = self.loaded_data['annotations'][index]
        item_id = item.get('id', str(index))
        
        sample_info = {
            'annotation': item
        }
        
        # 添加图像信息
        if item_id in self.loaded_data['images']:
            image = self.loaded_data['images'][item_id]
            sample_info['image_shape'] = image.size if hasattr(image, 'size') else str(type(image))
        
        # 添加文本信息
        if item_id in self.loaded_data['texts']:
            text = self.loaded_data['texts'][item_id]
            sample_info['text_preview'] = text[:100] + '...' if len(text) > 100 else text
        
        # 添加元数据信息
        if item_id in self.loaded_data['metadata']:
            sample_info['metadata_keys'] = list(self.loaded_data['metadata'][item_id].keys())
        
        return sample_info

class StandardMultiModalDataset(Dataset):
    """
    标准多模态数据集
    """
    
    def __init__(self, 
                 data: Dict[str, Any],
                 image_size: Tuple[int, int] = (224, 224),
                 augmentation: bool = False,
                 use_cache: bool = True):
        """
        Args:
            data: 数据集数据
            image_size: 图像大小
            augmentation: 是否使用数据增强
            use_cache: 是否使用缓存
        """
        self.annotations = data.get('annotations', [])
        self.images = data.get('images', {})
        self.texts = data.get('texts', {})
        self.metadata = data.get('metadata', {})
        
        self.image_size = image_size
        self.augmentation = augmentation
        self.use_cache = use_cache
        
        # 图像预处理
        self.image_transform = self._get_image_transform()
    
    def _get_image_transform(self):
        """
        获取图像转换函数
        """
        def transform(image):
            # 调整大小
            if isinstance(image, Image.Image):
                image = image.resize(self.image_size)
                image = np.array(image)
            elif isinstance(image, np.ndarray):
                image = cv2.resize(image, self.image_size)
            
            # 数据增强
            if self.augmentation:
                import albumentations as A
                aug = A.Compose([
                    A.RandomRotate90(),
                    A.Flip(),
                    A.OneOf([
                        A.GaussianBlur(),
                        A.MotionBlur(),
                    ], p=0.3),
                    A.OneOf([
                        A.HueSaturationValue(),
                        A.RandomBrightnessContrast(),
                    ], p=0.3),
                ])
                image = aug(image=image)['image']
            
            # 归一化
            image = image.astype(np.float32) / 255.0
            
            return image
        
        return transform
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations[idx]
        item_id = item.get('id', str(idx))
        
        # 构建样本
        sample = {
            'id': item_id,
            'annotation': item
        }
        
        # 加载图像
        if item_id in self.images:
            image = self.images[item_id]
            image = self.image_transform(image)
            sample['image'] = image
        
        # 加载文本
        if item_id in self.texts:
            sample['text'] = self.texts[item_id]
        
        # 加载元数据
        if item_id in self.metadata:
            sample['metadata'] = self.metadata[item_id]
        
        return sample
    
    def collate_fn(self, batch):
        """
        批次处理函数
        """
        collated = {
            'ids': [],
            'annotations': [],
        }
        
        # 检查批次中的键
        has_image = any('image' in sample for sample in batch)
        has_text = any('text' in sample for sample in batch)
        has_metadata = any('metadata' in sample for sample in batch)
        
        # 收集数据
        for sample in batch:
            collated['ids'].append(sample['id'])
            collated['annotations'].append(sample['annotation'])
            
            if has_image and 'image' in sample:
                if 'images' not in collated:
                    collated['images'] = []
                collated['images'].append(sample['image'])
            
            if has_text and 'text' in sample:
                if 'texts' not in collated:
                    collated['texts'] = []
                collated['texts'].append(sample['text'])
            
            if has_metadata and 'metadata' in sample:
                if 'metadata' not in collated:
                    collated['metadata'] = []
                collated['metadata'].append(sample['metadata'])
        
        # 转换为numpy数组
        if has_image:
            collated['images'] = np.stack(collated['images'])
        
        return collated

class PyTorchMultiModalDataset(Dataset):
    """
    PyTorch格式的多模态数据集
    """
    
    def __init__(self, 
                 data: Dict[str, Any],
                 transform: Optional[Any] = None,
                 text_processor: Optional[Callable] = None):
        """
        Args:
            data: 数据集数据
            transform: 图像转换函数
            text_processor: 文本处理函数
        """
        self.annotations = data.get('annotations', [])
        self.images = data.get('images', {})
        self.texts = data.get('texts', {})
        self.metadata = data.get('metadata', {})
        
        self.transform = transform
        self.text_processor = text_processor or (lambda x: x)
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        item = self.annotations[idx]
        item_id = item.get('id', str(idx))
        
        # 构建样本
        sample = {
            'id': item_id
        }
        
        # 加载图像
        if item_id in self.images:
            image = self.images[item_id]
            if self.transform:
                image = self.transform(image)
            sample['image'] = image
        
        # 加载文本
        if item_id in self.texts:
            text = self.text_processor(self.texts[item_id])
            sample['text'] = text
        
        # 加载标签
        if 'category' in item:
            sample['category'] = item['category']
        
        # 加载边界框
        if 'bbox' in item:
            sample['bbox'] = torch.tensor(item['bbox'], dtype=torch.float32)
        
        return sample
    
    def collate_fn(self, batch):
        """
        批次处理函数
        """
        collated = {
            'ids': [sample['id'] for sample in batch]
        }
        
        # 检查批次中的键
        keys = set()
        for sample in batch:
            keys.update(sample.keys())
        
        # 收集数据
        for key in keys - {'id'}:
            if all(key in sample for sample in batch):
                if key == 'image':
                    collated[key] = torch.stack([sample[key] for sample in batch])
                elif key == 'category':
                    # 处理类别标签
                    categories = [sample[key] for sample in batch]
                    # 如果是字符串，需要映射到索引
                    if all(isinstance(cat, str) for cat in categories):
                        # 这里应该使用预定义的类别映射
                        # 简化版本：使用临时映射
                        cat_to_idx = {cat: i for i, cat in enumerate(set(categories))}
                        collated[key] = torch.tensor([cat_to_idx[cat] for cat in categories])
                    else:
                        collated[key] = torch.tensor(categories)
                elif isinstance(batch[0][key], torch.Tensor):
                    collated[key] = torch.stack([sample[key] for sample in batch])
                else:
                    collated[key] = [sample[key] for sample in batch]
        
        return collated

# 工具函数
def create_synthetic_dataset(output_dir: str, 
                            num_samples: int = 1000,
                            num_categories: int = 10,
                            image_size: Tuple[int, int] = (224, 224)) -> bool:
    """
    创建合成数据集
    
    Args:
        output_dir: 输出目录
        num_samples: 样本数量
        num_categories: 类别数量
        image_size: 图像大小
        
    Returns:
        是否创建成功
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        images_dir = os.path.join(output_dir, 'images')
        os.makedirs(images_dir, exist_ok=True)
        
        # 创建类别
        categories = [f'category_{i}' for i in range(num_categories)]
        
        # 创建样本
        annotations = []
        for i in range(num_samples):
            # 随机类别
            category = random.choice(categories)
            
            # 生成随机图像
            image_filename = f'sample_{i}.png'
            image_path = os.path.join(images_dir, image_filename)
            
            # 创建随机图像
            image = np.random.randint(0, 256, (*image_size, 3), dtype=np.uint8)
            cv2.imwrite(image_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
            
            # 创建标注
            annotation = {
                'id': f'sample_{i}',
                'image_path': os.path.join('images', image_filename),
                'category': category,
                'text': f'This is a {category} object.',
                'metadata': {
                    'timestamp': pd.Timestamp.now().isoformat(),
                    'random_value': np.random.random(),
                    'source': 'synthetic'
                }
            }
            annotations.append(annotation)
            
            if (i + 1) % 100 == 0:
                logger.info(f"已生成 {i + 1}/{num_samples} 个样本")
        
        # 保存标注
        with open(os.path.join(output_dir, 'annotations.json'), 'w', encoding='utf-8') as f:
            json.dump(annotations, f, ensure_ascii=False, indent=2)
        
        # 保存数据集信息
        dataset_info = {
            'name': 'Synthetic MultiModal Dataset',
            'description': 'A synthetic multimodal dataset for testing',
            'num_samples': num_samples,
            'num_categories': num_categories,
            'image_size': image_size,
            'created_time': pd.Timestamp.now().isoformat()
        }
        
        with open(os.path.join(output_dir, 'dataset_info.json'), 'w', encoding='utf-8') as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"合成数据集已创建到: {output_dir}, 样本数: {num_samples}")
        return True
        
    except Exception as e:
        logger.error(f"创建合成数据集失败: {e}")
        return False

# 示例用法
if __name__ == '__main__':
    # 创建合成数据集
    # create_synthetic_dataset('./data/synthetic', num_samples=100)
    
    # 加载数据集
    loader = MultiModalDatasetLoader(
        dataset_dir='./data/synthetic',
        image_size=(224, 224),
        batch_size=4
    )
    
    # 加载数据
    data = loader.load_dataset()
    
    # 查看统计信息
    stats = loader.get_statistics()
    print(f"数据集统计: {stats}")
    
    # 查看样本
    sample = loader.show_sample(0)
    print(f"第一个样本: {sample}")
    
    # 获取数据加载器
    # data_loader = loader.get_data_loader()
    # for batch in data_loader:
    #     print(f"批次大小: {len(batch['ids'])}")
    #     break