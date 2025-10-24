#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TAMMA多模态检索算法演示脚本

该脚本演示了如何使用TAMMA算法及其组件进行多模态检索任务，
包括配置加载、特征提取、索引构建、相似度搜索和结果评估。
"""

import os
import sys
import time
import json
import yaml
import argparse
import logging
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 导入必要的模块
from utils.config_manager import ConfigManager
from utils.feature_utils import FeatureUtils
from utils.image_preprocessor import ImagePreprocessor
from feature_extraction.color_extractor import ColorFeatureExtractor
from feature_extraction.texture_extractor import TextureFeatureExtractor
from feature_extraction.text_extractor import TextFeatureExtractor
from algorithms.tamma_complete import TAMMAComplete
from evaluation.advanced_evaluator import AdvancedEvaluator
from data.dataset_generator import LostFoundDatasetGeneratorComplete

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("./tamma_demo.log")
    ]
)
logger = logging.getLogger("tamma_demo")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='TAMMA多模态检索算法演示')
    parser.add_argument('--config', type=str, 
                       default='../configs/experiment_config.yaml',
                       help='配置文件路径')
    parser.add_argument('--output-dir', type=str, 
                       default='./demo_results',
                       help='结果输出目录')
    parser.add_argument('--num-samples', type=int, 
                       default=100,
                       help='生成的样本数量')
    parser.add_argument('--category', type=str, 
                       default='book',
                       help='测试的物品类别')
    return parser.parse_args()


def setup_directories(output_dir):
    """创建必要的目录"""
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'features'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
    return output_dir


def generate_sample_dataset(config, num_samples, category, output_dir):
    """生成示例数据集"""
    logger.info(f"生成示例数据集: {category}类别，{num_samples}个样本")
    
    # 初始化数据集生成器
    generator = LostFoundDatasetGeneratorComplete(
        output_dir=os.path.join(output_dir, 'images'),
        image_size=tuple(config['dataset']['synthetic_params']['image_size'])
    )
    
    # 生成合成数据集
    dataset = generator.generate_synthetic_dataset(
        num_samples=num_samples
    )
    
    # 保存数据集信息
    dataset_info = {
        'category': category,
        'num_samples': len(dataset),
        'samples': dataset
    }
    
    with open(os.path.join(output_dir, 'dataset_info.json'), 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    logger.info(f"数据集生成完成，保存在: {output_dir}/images/")
    return dataset


def make_serializable(obj):
    """递归地将numpy对象转换为可JSON序列化的类型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: make_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [make_serializable(item) for item in obj]
    elif obj is None:
        return []
    else:
        return obj

def demonstrate_feature_extraction(dataset, output_dir):
    """演示特征提取"""
    logger.info("演示特征提取...")
    
    # 初始化各种特征提取器
    color_extractor = ColorFeatureExtractor(
        color_space='hsv',
        h_bins=8,
        s_bins=8,
        v_bins=8,
        pyramid_levels=2
    )
    
    texture_extractor = TextureFeatureExtractor(
        feature_types=['lbp'],
        lbp_radius=1,
        lbp_n_points=8
    )
    
    text_extractor = TextFeatureExtractor(
        lang='ch',
        use_gpu=False
    )
    
    # 提取少量样本的特征进行演示
    features_dict = {}
    
    # dataset 是数据集目录路径，我们需要读取其中的图像
    import glob
    image_dir = os.path.join(dataset, 'images')
    if not os.path.exists(image_dir):
        # 尝试其他可能的目录结构
        image_dir = os.path.join(dataset, 'synthetic', 'images')
    
    if os.path.exists(image_dir):
            image_paths = glob.glob(os.path.join(image_dir, '*.jpg')) + glob.glob(os.path.join(image_dir, '*.png'))
            num_demo_samples = min(5, len(image_paths))
            
            for i in range(num_demo_samples):
                image_path = image_paths[i]
                
                try:
                    # 先读取图像
                    image = cv2.imread(image_path)
                    if image is None:
                        logger.error(f"无法读取图像: {image_path}")
                        continue
                    
                    # 提取颜色特征
                    color_features = color_extractor.extract(image)
                    
                    # 提取纹理特征
                    texture_features = texture_extractor.extract(image)
                    
                    # 提取文字特征
                    text_features = text_extractor.extract(image)
                    
                    # 转换所有特征为可序列化格式
                    serializable_color = make_serializable(color_features)
                    serializable_texture = make_serializable(texture_features)
                    serializable_text = make_serializable(text_features)
                    
                    features_dict[os.path.basename(image_path)] = {
                        'color': serializable_color,
                        'texture': serializable_texture,
                        'text': serializable_text
                    }
                    
                    logger.info(f"样本 {i+1}/{num_demo_samples} 特征提取完成")
                    
                except Exception as e:
                    logger.error(f"样本 {image_path} 特征提取失败: {str(e)}")
    else:
        logger.error(f"找不到图像目录: {image_dir}")
        return features_dict
    
    # 保存特征
    try:
        with open(os.path.join(output_dir, 'features/demo_features.json'), 'w', encoding='utf-8') as f:
            json.dump(features_dict, f, ensure_ascii=False, indent=2)
        logger.info("特征提取演示完成")
    except Exception as e:
        logger.error(f"保存特征时出错: {str(e)}")
        # 如果JSON保存失败，尝试打印特征类型以帮助调试
        for img_name, features in features_dict.items():
            logger.error(f"图像 {img_name} 特征类型:")
            logger.error(f"  color: {type(features['color'])}")
            logger.error(f"  texture: {type(features['texture'])}")
            logger.error(f"  text: {type(features['text'])}")
    
    return features_dict


def demonstrate_tamma_algorithm(dataset, config, output_dir):
    """演示TAMMA算法"""
    logger.info("演示TAMMA算法...")
    
    # 初始化TAMMA算法参数
    tamma_params = config['algorithms']['tamma']['params']
    
    # 准备各模态配置
    color_config = {
        'color_space': tamma_params.get('color_space', 'HSV'),
        'h_bins': tamma_params.get('h_bins', 8),
        's_bins': tamma_params.get('s_bins', 8),
        'v_bins': tamma_params.get('v_bins', 8)
    }
    
    texture_config = {
        'feature_types': ['lbp']
    }
    
    text_config = {
        'lang': 'ch',
        'use_gpu': False
    }
    
    # 进一步降低颜色粗筛选阈值以避免无候选问题
    color_threshold = tamma_params.get('color_threshold', 0.2)  # 默认值进一步降低到0.2
    logger.info(f"使用颜色粗筛选阈值: {color_threshold}")
    
    tamma = TAMMAComplete(
        color_config=color_config,
        texture_config=texture_config,
        text_config=text_config,
        color_threshold=color_threshold,
        top_k_coarse=tamma_params.get('top_k_coarse', 50),
        fusion_method=tamma_params.get('fusion_method', 'weighted_sum')
    )
    
    # 准备查询集和图库集
    import glob
    
    # 检查dataset类型，处理字符串和列表两种情况
    if isinstance(dataset, str):
        logger.info(f"处理字符串类型的数据集路径: {dataset}")
        base_dir = dataset
    elif isinstance(dataset, list):
        logger.info(f"处理列表类型的数据集，包含 {len(dataset)} 个样本")
        # 尝试从列表中提取图像路径信息
        if dataset and isinstance(dataset[0], dict) and 'image_path' in dataset[0]:
            first_path = dataset[0]['image_path']
            base_dir = os.path.dirname(first_path)
        else:
            base_dir = output_dir
            logger.warning(f"无法从列表中提取目录路径，使用输出目录: {base_dir}")
    else:
        logger.warning(f"dataset类型错误，预期列表或字符串，实际为{type(dataset)}")
        base_dir = output_dir
    
    # 尝试多种可能的图像目录结构
    possible_paths = [
        os.path.join(base_dir, 'images'),
        os.path.join(base_dir, 'synthetic', 'images'),
        os.path.join(base_dir, 'images', 'synthetic'),
        os.path.join(output_dir, 'images'),
        os.path.join(output_dir, 'images', 'synthetic')
    ]
    
    image_dir = None
    for path in possible_paths:
        if os.path.exists(path):
            image_dir = path
            logger.info(f"找到图像目录: {image_dir}")
            break
    
    # 如果没有找到有效目录，使用默认路径
    if image_dir is None:
        image_dir = os.path.join(base_dir, 'images')
        logger.warning(f"未找到标准图像目录，使用默认路径: {image_dir}")
        # 确保目录存在
        os.makedirs(image_dir, exist_ok=True)
    if os.path.exists(image_dir):
        all_images = glob.glob(os.path.join(image_dir, '*.jpg')) + glob.glob(os.path.join(image_dir, '*.png'))
        all_images = all_images[:50]  # 限制数量以加快演示
        
        gallery_size = min(40, len(all_images))
        query_size = min(5, len(all_images) - gallery_size)
        
        gallery_images = all_images[:gallery_size]
        query_images = all_images[gallery_size:gallery_size+query_size]
        
        logger.info(f"准备图库集 ({gallery_size}张图片) 和查询集 ({query_size}张图片)")
        
        # 构建图库索引（使用简化的格式）
        gallery_dataset = []
        for img_path in gallery_images:
            # 为演示创建简单的图库项
            gallery_dataset.append({
                'image_path': img_path,
                'category': 'book'  # 假设所有图像都是book类别
            })
        
        start_time = time.time()
        tamma.build_index(gallery_dataset)
        build_time = time.time() - start_time
        logger.info(f"图库索引构建完成，耗时: {build_time:.2f}秒")
        
        # 执行查询
        results = []
        for query_image in query_images:
            query_start = time.time()
            query_results = tamma.search(
                query_image_path=query_image,
                category='book',
                k=5
            )
        query_time = time.time() - query_start
        
        results.append({
            'query': os.path.basename(query_image),
            'category': 'book',
            'results': query_results,
            'query_time': query_time
        })
        
        logger.info(f"查询 {os.path.basename(query_image)} 完成，耗时: {query_time:.4f}秒")
        
        # 保存查询结果
        os.makedirs(os.path.join(output_dir, 'results'), exist_ok=True)
        with open(os.path.join(output_dir, 'results/tamma_results.json'), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        logger.info("TAMMA算法演示完成")
        return results
    else:
        logger.error(f"找不到图像目录: {image_dir}")
        return []


def visualize_results(results, dataset, output_dir):
    """可视化检索结果"""
    if not results:
        logger.warning("没有结果可供可视化")
        return
    
    logger.info("可视化检索结果...")
    
    # 确定图像基础目录
    base_image_dir = None
    if isinstance(dataset, str):
        base_image_dir = dataset
    elif isinstance(dataset, list):
        # 尝试从列表中提取目录
        if dataset and isinstance(dataset[0], dict) and 'image_path' in dataset[0]:
            first_path = dataset[0]['image_path']
            base_image_dir = os.path.dirname(first_path)
        else:
            base_image_dir = output_dir
    else:
        logger.warning(f"dataset类型错误，预期列表或字符串，实际为{type(dataset)}")
        base_image_dir = output_dir
    
    # 创建一个简单的结果可视化
    num_visualizations = min(3, len(results))
    
    for i in range(num_visualizations):
        result = results[i]
        query_image = result['query']
        
        # 创建可视化图表
        plt.figure(figsize=(15, 8))
        
        # 显示查询图像
        plt.subplot(1, 6, 1)
        # 尝试多种可能的路径查找查询图像
        possible_paths = [
            os.path.join(base_image_dir, 'synthetic', 'images', query_image),
            os.path.join(base_image_dir, 'images', 'synthetic', query_image),
            os.path.join(base_image_dir, 'images', query_image),
            os.path.join(output_dir, 'images', 'synthetic', query_image),
            os.path.join(output_dir, 'images', query_image)
        ]
        
        query_path = None
        for path in possible_paths:
            if os.path.exists(path):
                query_path = path
                break
                
        if query_path and os.path.exists(query_path):
            img = plt.imread(query_path)
            plt.imshow(img)
            plt.title('Query')
            plt.axis('off')
        
        # 显示检索结果
        for j, item in enumerate(result['results'][:5]):
            # 假设item包含image_path或我们需要使用索引在图库中查找
            if 'image_path' in item:
                result_image = item['image_path']
            else:
                # 如果只有索引，我们可能需要使用一个单独的图库列表
                # 这里简化处理，假设item中的信息足够
                continue
                
            # 尝试在dataset路径中查找结果图像
            result_path = os.path.join(dataset, 'synthetic', 'images', os.path.basename(result_image))
            
            plt.subplot(1, 6, j+2)
            if os.path.exists(result_path):
                img = plt.imread(result_path)
                plt.imshow(img)
                plt.title(f"Rank {j+1}\nScore: {item['score']:.2f}")
                plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'results/visualization_{i+1}.png'), dpi=300)
        plt.close()
    
    logger.info("结果可视化完成")


def evaluate_results(results, dataset, output_dir):
    """评估检索结果"""
    logger.info("评估检索结果...")
    
    # 准备评估数据
    all_query_results = []
    all_ground_truth = []
    
    gallery_size = 50  # 与前面的图库集大小保持一致
    
    # 检查dataset类型并进行适当处理
    if isinstance(dataset, str):
        logger.info(f"处理字符串类型的数据集路径: {dataset}")
        # 对于字符串类型的数据集，我们将创建一个简单的模拟数据集用于评估
        # 或者直接提供基于结果的评估而不依赖于真实数据集
        # 这里我们提供一个基本的评估框架
        metrics = {
            'precision_at_k': {1: 0.0, 3: 0.0, 5: 0.0},
            'recall_at_k': {1: 0.0, 3: 0.0, 5: 0.0},
            'mrr': 0.0,
            'map': 0.0
        }
    elif not isinstance(dataset, list):
        logger.warning(f"dataset类型错误，预期列表或字符串，实际为{type(dataset).__name__}")
        metrics = {
            'precision_at_k': {1: 0.0, 3: 0.0, 5: 0.0},
            'recall_at_k': {1: 0.0, 3: 0.0, 5: 0.0},
            'mrr': 0.0,
            'map': 0.0
        }
        
        # 保存评估结果
        with open(os.path.join(output_dir, 'results/evaluation_metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        logger.info("评估结果: 无数据")
        return metrics
    
    for result in results:
        query_idx = gallery_size + results.index(result)
        # 简单地将与查询图像同一类别的图库图像作为正样本
        try:
            # 安全访问category字段
            query_category = dataset[query_idx].get('category', '') if query_idx < len(dataset) and isinstance(dataset[query_idx], dict) else ''
            gt = []
            for i in range(min(gallery_size, len(dataset))):
                if isinstance(dataset[i], dict) and dataset[i].get('category', '') == query_category:
                    gt.append(i)
            
            # 转换结果格式为 (索引, 相似度) 元组列表
            pred_with_scores = []
            if isinstance(result.get('results'), list):
                for item in result['results']:
                    if isinstance(item, dict) and 'index' in item and 'similarity' in item:
                        pred_with_scores.append((item['index'], item['similarity']))
            
            all_ground_truth.append(gt)
            all_query_results.append(pred_with_scores)
        except Exception as e:
            logger.error(f"处理结果时出错: {e}")
            # 添加空结果以保持对齐
            all_ground_truth.append([])
            all_query_results.append([])
    
    # 检查是否有查询结果
    if len(all_query_results) == 0:
        logger.warning("没有查询结果可评估")
        metrics = {
            'precision_at_k': {1: 0.0, 3: 0.0, 5: 0.0},
            'recall_at_k': {1: 0.0, 3: 0.0, 5: 0.0},
            'mrr': 0.0,
            'map': 0.0
        }
        
        # 保存评估结果
        with open(os.path.join(output_dir, 'results/evaluation_metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
        
        logger.info("评估结果: 无数据")
        return metrics
    
    # 初始化评估器
    evaluator = AdvancedEvaluator(metrics=['precision_at_k', 'recall_at_k', 'mrr', 'map', 'ndcg'], max_k=10)
    
    # 批量评估
    evaluation_results = evaluator.evaluate_batch(
        all_query_results=all_query_results,
        all_ground_truth_indices=all_ground_truth,
        algorithm_name='TAMMAComplete',
        dataset_name='Synthetic Dataset'
    )
    
    # 从评估结果中提取所需指标，使用get方法安全地访问可能不存在的键
    metrics = {
        'precision_at_k': {},
        'recall_at_k': {},
        'mrr': evaluation_results.get('average_metrics', {}).get('mrr', {}).get('mean', 0.0),
        'map': evaluation_results.get('average_metrics', {}).get('map', {}).get('mean', 0.0)
    }
    
    # 提取不同K值的精确率作为top-k准确率
    for k in [1, 3, 5]:
        if f'precision@{k}' in evaluation_results.get('average_metrics', {}):
            metrics['precision_at_k'][k] = evaluation_results['average_metrics'][f'precision@{k}']['mean']
            metrics['recall_at_k'][k] = evaluation_results['average_metrics'][f'recall@{k}']['mean']
    
    # 保存评估结果
    with open(os.path.join(output_dir, 'results/evaluation_metrics.json'), 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    
    # 打印评估结果
    logger.info("评估结果:")
    logger.info(f"Top-1精确率: {metrics['precision_at_k'].get(1, 0.0):.4f}")
    logger.info(f"Top-3精确率: {metrics['precision_at_k'].get(3, 0.0):.4f}")
    logger.info(f"Top-5精确率: {metrics['precision_at_k'].get(5, 0.0):.4f}")
    logger.info(f"MRR: {metrics['mrr']:.4f}")
    logger.info(f"MAP: {metrics['map']:.4f}")
    
    return metrics


def main():
    """主函数"""
    try:
        # 解析参数
        args = parse_arguments()
        
        # 加载配置
        config_manager = ConfigManager()
        config = config_manager.load_config(args.config)
        
        # 设置输出目录
        output_dir = setup_directories(args.output_dir)
        
        logger.info("开始TAMMA算法演示...")
        
        # 生成示例数据集
        dataset = generate_sample_dataset(
            config=config,
            num_samples=args.num_samples,
            category=args.category,
            output_dir=output_dir
        )
        
        # 演示特征提取
        features = demonstrate_feature_extraction(
            dataset=dataset,
            output_dir=output_dir
        )
        
        # 演示TAMMA算法
        results = demonstrate_tamma_algorithm(
            dataset=dataset,
            config=config,
            output_dir=output_dir
        )
        
        # 可视化结果
        visualize_results(
            results=results,
            dataset=dataset,
            output_dir=output_dir
        )
        
        # 评估结果
        metrics = evaluate_results(
            results=results,
            dataset=dataset,
            output_dir=output_dir
        )
        
        logger.info("TAMMA算法演示完成！")
        logger.info(f"所有结果保存在: {output_dir}")
        
    except Exception as e:
        logger.error(f"演示过程中发生错误: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()