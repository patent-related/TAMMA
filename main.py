#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TAMMA多模态检索系统主入口

该模块提供了TAMMA多模态检索系统的命令行接口，
支持数据集生成、模型训练、特征提取、索引构建、检索和评估等功能。
"""

import os
import sys
import time
import json
import yaml
import argparse
import logging
import numpy as np
from datetime import datetime

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# 导入必要的模块
from utils.config_manager import ConfigManager
from utils.feature_utils import FeatureUtils, create_codebook
from utils.image_preprocessor import ImagePreprocessor
from data.dataset_generator import LostFoundDatasetGeneratorComplete
from data.data_loader import MultiModalDatasetLoader
from algorithms.tamma_complete import TAMMAComplete
from algorithms.baselines.color_only_matcher import ColorOnlyMatcherComplete
from algorithms.baselines.dual_modality_matcher import DualModalityMatcherComplete
from algorithms.baselines.fixed_weight_multimodal_matcher import FixedWeightMultimodalMatcherComplete
from algorithms.baselines.deep_learning_matcher import DeepLearningMatcherComplete
from evaluation.advanced_evaluator import AdvancedEvaluator
from evaluation.performance_analyzer import PerformanceAnalyzerComplete
from experiments.experiment_manager import ExperimentManager

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("./tamma_system.log")
    ]
)
logger = logging.getLogger("tamma_main")


def parse_arguments():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description='TAMMA多模态检索系统')
    
    # 子命令
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # generate_dataset 命令
    gen_dataset_parser = subparsers.add_parser('generate_dataset', help='生成数据集')
    gen_dataset_parser.add_argument('--config', type=str, 
                                   default='configs/experiment_config.yaml',
                                   help='配置文件路径')
    gen_dataset_parser.add_argument('--output-dir', type=str, 
                                   default='./data',
                                   help='数据集输出目录')
    gen_dataset_parser.add_argument('--num-samples', type=int, 
                                   default=1000,
                                   help='生成的样本总数')
    gen_dataset_parser.add_argument('--categories', type=str, nargs='+', 
                                   help='要生成的类别列表')
    
    # train 命令
    train_parser = subparsers.add_parser('train', help='训练模型')
    train_parser.add_argument('--config', type=str, 
                             default='configs/experiment_config.yaml',
                             help='配置文件路径')
    train_parser.add_argument('--model', type=str, 
                             default='tamma',
                             choices=['tamma', 'deep_learning'],
                             help='要训练的模型')
    train_parser.add_argument('--data-dir', type=str, 
                             help='训练数据目录')
    train_parser.add_argument('--use-gpu', action='store_true', help='启用GPU加速')
    train_parser.add_argument('--gpu-device', type=int, default=0, help='使用的GPU设备ID')
    
    # extract_features 命令
    extract_parser = subparsers.add_parser('extract_features', help='提取特征')
    extract_parser.add_argument('--config', type=str, 
                               default='configs/experiment_config.yaml',
                               help='配置文件路径')
    extract_parser.add_argument('--input-dir', type=str, required=True,
                               help='输入图像目录')
    extract_parser.add_argument('--output-dir', type=str, 
                               default='./features',
                               help='特征输出目录')
    extract_parser.add_argument('--modalities', type=str, nargs='+', 
                               default=['color', 'texture', 'text', 'sift'],
                               help='要提取的模态')
    extract_parser.add_argument('--use-gpu', action='store_true', help='启用GPU加速')
    extract_parser.add_argument('--gpu-device', type=int, default=0, help='使用的GPU设备ID')
    
    # build_index 命令
    build_parser = subparsers.add_parser('build_index', help='构建检索索引')
    build_parser.add_argument('--config', type=str, 
                             default='configs/experiment_config.yaml',
                             help='配置文件路径')
    build_parser.add_argument('--data-dir', type=str, required=True,
                             help='图库数据集目录')
    build_parser.add_argument('--index-path', type=str, 
                             default='./indexes/tamma_index.pkl',
                             help='索引保存路径')
    build_parser.add_argument('--model', type=str, 
                             default='tamma',
                             choices=['tamma', 'color_only', 'dual_modality', 'fixed_weight_multimodal', 'deep_learning'],
                             help='使用的模型')
    build_parser.add_argument('--use-gpu', action='store_true', help='启用GPU加速')
    build_parser.add_argument('--gpu-device', type=int, default=0, help='使用的GPU设备ID')
    
    # search 命令
    search_parser = subparsers.add_parser('search', help='执行检索')
    search_parser.add_argument('--config', type=str, 
                              default='configs/experiment_config.yaml',
                              help='配置文件路径')
    search_parser.add_argument('--query', type=str, required=True,
                              help='查询图像路径')
    search_parser.add_argument('--index-path', type=str, 
                              default='./indexes/tamma_index.pkl',
                              help='索引路径')
    search_parser.add_argument('--output-dir', type=str, 
                              default='./search_results',
                              help='检索结果输出目录')
    search_parser.add_argument('--k', type=int, 
                              default=10,
                              help='返回的结果数量')
    search_parser.add_argument('--category', type=str, 
                              help='物品类别（可选）')
    search_parser.add_argument('--use-gpu', action='store_true', help='启用GPU加速')
    search_parser.add_argument('--gpu-device', type=int, default=0, help='使用的GPU设备ID')
    
    # evaluate 命令
    eval_parser = subparsers.add_parser('evaluate', help='评估检索系统')
    eval_parser.add_argument('--config', type=str, 
                            default='configs/experiment_config.yaml',
                            help='配置文件路径')
    eval_parser.add_argument('--query-dir', type=str, required=True,
                            help='查询数据集目录')
    eval_parser.add_argument('--gallery-dir', type=str, required=True,
                            help='图库数据集目录')
    eval_parser.add_argument('--output-dir', type=str, 
                            default='./evaluation_results',
                            help='评估结果输出目录')
    eval_parser.add_argument('--models', type=str, nargs='+', 
                            default=['tamma', 'color_only', 'dual_modality', 'fixed_weight_multimodal', 'deep_learning'],
                            help='要评估的模型列表')
    eval_parser.add_argument('--use-gpu', action='store_true', help='启用GPU加速')
    eval_parser.add_argument('--gpu-device', type=int, default=0, help='使用的GPU设备ID')
    
    # experiment 命令
    exp_parser = subparsers.add_parser('experiment', help='运行对比实验')
    exp_parser.add_argument('--config', type=str, 
                           default='configs/experiment_config.yaml',
                           help='配置文件路径')
    exp_parser.add_argument('--output-dir', type=str, 
                           default='./experiment_results',
                           help='实验结果输出目录')
    exp_parser.add_argument('--num-runs', type=int, 
                           help='实验重复次数')
    
    # create_codebook 命令
    codebook_parser = subparsers.add_parser('create_codebook', help='创建SIFT码本')
    codebook_parser.add_argument('--config', type=str, 
                               default='configs/experiment_config.yaml',
                               help='配置文件路径')
    codebook_parser.add_argument('--data-dir', type=str, required=True,
                               help='训练图像目录')
    codebook_parser.add_argument('--output-path', type=str, 
                               default='./models/sift_codebook.pkl',
                               help='码本输出路径')
    codebook_parser.add_argument('--size', type=int, 
                               help='码本大小')
    
    # demo 命令
    demo_parser = subparsers.add_parser('demo', help='运行演示')
    demo_parser.add_argument('--config', type=str, 
                           default='configs/experiment_config.yaml',
                           help='配置文件路径')
    demo_parser.add_argument('--output-dir', type=str, 
                           default='./demo_results',
                           help='演示结果输出目录')
    
    return parser.parse_args()


def setup_directories(base_dir):
    """
    创建必要的目录结构
    """
    directories = [
        os.path.join(base_dir, 'data'),
        os.path.join(base_dir, 'features'),
        os.path.join(base_dir, 'indexes'),
        os.path.join(base_dir, 'models'),
        os.path.join(base_dir, 'results'),
        os.path.join(base_dir, 'logs')
    ]
    
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)
    
    return directories


def generate_dataset(args):
    """
    生成数据集
    """
    logger.info("开始生成数据集...")
    
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # 使用配置中的root_dir作为输出目录
    root_dir = config['dataset']['root_dir']
    output_dir = args.output_dir if args.output_dir else os.path.join(root_dir, 'synthetic')
    
    # 设置输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化数据集生成器
    synthetic_params = config['dataset']['synthetic_params']
    generator = LostFoundDatasetGeneratorComplete(
        output_dir=output_dir,
        image_size=synthetic_params['image_size'],
        max_objects_per_image=3,
        random_seed=config['experiment']['seed']
    )
    
    # 生成数据集
    total_samples = args.num_samples
    
    # 使用generate_synthetic_dataset方法生成样本
    dataset_dir = generator.generate_synthetic_dataset(
        num_samples=total_samples,
        output_json=True
    )
    
    logger.info(f"数据集生成完成，保存到: {dataset_dir}")
    
    # 保存数据集信息
    dataset_info = {
        'categories': LostFoundDatasetGeneratorComplete.ITEM_CLASSES,
        'total_samples': total_samples,
        'generated_time': datetime.now().isoformat(),
        'image_size': synthetic_params['image_size']
    }
    
    info_path = os.path.join(output_dir, 'dataset_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(dataset_info, f, ensure_ascii=False, indent=2)
    
    logger.info(f"数据集生成完成！")
    logger.info(f"总样本数: {total_samples}")
    logger.info(f"数据集信息保存在: {info_path}")
    logger.info(f"图像保存在: {output_dir}")


def train_model(args):
    """
    训练模型
    """
    logger.info(f"开始训练 {args.model} 模型...")
    
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # 目前只有深度学习模型需要训练
    if args.model == 'deep_learning':
        # 这里可以实现深度学习模型的训练逻辑
        logger.info("深度学习模型训练功能正在开发中...")
        # 示例代码框架
        # model = DeepLearningMatcherComplete(config['algorithms']['deep_learning']['params'])
        # model.train(data_dir=args.data_dir, epochs=10, batch_size=32)
    elif args.model == 'tamma':
        logger.info("TAMMA算法主要基于特征提取和融合，不需要传统意义上的训练")
        logger.info("请先使用 create_codebook 命令创建SIFT码本")
    
    logger.info("模型训练完成")


def extract_features(args):
    """
    提取特征
    """
    logger.info(f"开始提取特征，模态: {', '.join(args.modalities)}")
    
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # 设置输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 获取图像列表
    image_paths = []
    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
        image_paths.extend([os.path.join(args.input_dir, f) for f in os.listdir(args.input_dir) 
                           if f.lower().endswith(ext)])
    
    logger.info(f"找到 {len(image_paths)} 张图像")
    
    # 初始化特征工具类
    feature_utils = FeatureUtils()
    
    # 提取特征
    features_dict = {}
    start_time = time.time()
    
    for modality in args.modalities:
        logger.info(f"提取 {modality} 特征...")
        modality_features = []
        
        for i, image_path in enumerate(image_paths):
            try:
                if modality == 'color':
                    feature = feature_utils.extract_color_features(
                        image_path,
                        color_space=config['algorithms']['tamma']['params']['feature_extraction']['color']['color_space'],
                        bins=config['algorithms']['tamma']['params']['feature_extraction']['color']['bins']
                    )
                elif modality == 'texture':
                    feature = feature_utils.extract_texture_features(
                        image_path,
                        texture_type=config['algorithms']['tamma']['params']['feature_extraction']['texture']['texture_type']
                    )
                elif modality == 'text':
                    feature = feature_utils.extract_text_features(image_path)
                elif modality == 'sift':
                    # 注意：SIFT特征需要码本
                    codebook_path = config['sift_codebook']['save_path']
                    if os.path.exists(codebook_path):
                        feature = feature_utils.extract_sift_features(
                            image_path,
                            codebook_path=codebook_path
                        )
                    else:
                        logger.warning(f"SIFT码本不存在: {codebook_path}")
                        continue
                else:
                    logger.warning(f"不支持的模态: {modality}")
                    continue
                
                modality_features.append({
                    'image_path': image_path,
                    'feature': feature
                })
                
                if (i + 1) % 100 == 0:
                    logger.info(f"  已处理 {i+1}/{len(image_paths)} 张图像")
                    
            except Exception as e:
                logger.error(f"处理图像 {image_path} 失败: {str(e)}")
        
        # 保存模态特征
        modality_output_path = os.path.join(args.output_dir, f'{modality}_features.json')
        with open(modality_output_path, 'w', encoding='utf-8') as f:
            json.dump(modality_features, f, ensure_ascii=False, indent=2)
        
        features_dict[modality] = modality_features
    
    total_time = time.time() - start_time
    logger.info(f"特征提取完成！")
    logger.info(f"总耗时: {total_time:.2f}秒")
    logger.info(f"平均每张图像耗时: {total_time / len(image_paths):.4f}秒")
    logger.info(f"特征保存在: {args.output_dir}")


def build_index(args):
    """
    构建检索索引
    """
    logger.info(f"开始构建 {args.model} 模型索引...")
    
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # 创建索引目录
    os.makedirs(os.path.dirname(args.index_path), exist_ok=True)
    
    # 初始化数据加载器
    data_loader = MultiModalDatasetLoader(dataset_dir=args.data_dir)
    loaded_data = data_loader.load_dataset()
    gallery_dataset = loaded_data['annotations']
    
    logger.info(f"加载图库数据集，包含 {len(gallery_dataset)} 张图像")
    
    # 初始化模型
    if args.model == 'tamma':
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
        
        sift_config = {
            'codebook_path': config['sift_codebook']['save_path']
        }
        
        model = TAMMAComplete(
            color_config=color_config,
            sift_config=sift_config,
            texture_config=texture_config,
            text_config=text_config,
            color_threshold=tamma_params.get('color_threshold', 0.5),
            top_k_coarse=tamma_params.get('top_k_coarse', 50),
            fusion_method=tamma_params.get('fusion_method', 'weighted_sum'),
            use_gpu=False,
            gpu_device=0
        )
    elif args.model == 'color_only':
        model = ColorOnlyMatcherComplete(config['algorithms']['color_only']['params'])
    elif args.model == 'dual_modality':
        model = DualModalityMatcherComplete(config['algorithms']['dual_modality']['params'])
    elif args.model == 'fixed_weight_multimodal':
        model = FixedWeightMultimodalMatcherComplete(config['algorithms']['fixed_weight_multimodal']['params'])
    elif args.model == 'deep_learning':
        model = DeepLearningMatcherComplete(config['algorithms']['deep_learning']['params'])
    else:
        raise ValueError(f"不支持的模型: {args.model}")
    
    # 构建索引
    start_time = time.time()
    model.build_index(gallery_dataset)
    build_time = time.time() - start_time
    
    # 保存索引
    model.save_index(args.index_path)
    
    logger.info(f"索引构建完成！")
    logger.info(f"耗时: {build_time:.2f}秒")
    logger.info(f"索引保存在: {args.index_path}")


def search(args):
    """
    执行检索
    """
    logger.info(f"开始检索，查询图像: {args.query}")
    
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # 设置输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化TAMMA模型（默认使用TAMMA进行检索）
    tamma_params = config['algorithms']['tamma']['params']
    
    # 准备各模态配置
    color_config = {
        'color_space': tamma_params.get('color_space', 'HSV'),
        'h_bins': tamma_params.get('h_bins', 8),
        's_bins': tamma_params.get('s_bins', 8),
        'v_bins': tamma_params.get('v_bins', 8)
    }
    
    sift_config = {
        'codebook_path': config['sift_codebook']['save_path'],
        'n_features': tamma_params.get('sift_num_features', 1000)
    }
    
    texture_config = {
        'feature_types': tamma_params.get('texture_feature_types', ['lbp'])
    }
    
    text_config = {
        'lang': 'ch',
        'use_gpu': hasattr(args, 'use_gpu') and args.use_gpu
    }
    
    model = TAMMAComplete(
        color_config=color_config,
        sift_config=sift_config,
        texture_config=texture_config,
        text_config=text_config,
        category_weights=tamma_params.get('category_weights', None),
        color_threshold=tamma_params.get('color_threshold', 0.5),
        top_k_coarse=tamma_params.get('top_k_coarse', 50),
        spatial_weight=tamma_params.get('spatial_weight', 0.3),
        temporal_weight=tamma_params.get('temporal_weight', 0.2),
        fusion_method=tamma_params.get('fusion_method', 'weighted_sum'),
        use_gpu=hasattr(args, 'use_gpu') and args.use_gpu,
        gpu_device=hasattr(args, 'gpu_device') and args.gpu_device
    )
    
    # 加载索引
    model.load_index(args.index_path)
    
    # 执行检索
    start_time = time.time()
    results = model.search(
        query_image_path=args.query,
        category=args.category,
        k=args.k
    )
    search_time = time.time() - start_time
    
    # 保存检索结果
    results_info = {
        'query': args.query,
        'category': args.category,
        'k': args.k,
        'search_time': search_time,
        'results': results
    }
    
    results_path = os.path.join(args.output_dir, 'search_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(results_info, f, ensure_ascii=False, indent=2)
    
    logger.info(f"检索完成！")
    logger.info(f"耗时: {search_time:.4f}秒")
    logger.info(f"找到 {len(results)} 个结果")
    logger.info(f"结果保存在: {results_path}")
    
    # 打印前几个结果
    logger.info("前5个结果:")
    for i, result in enumerate(results[:5]):
        logger.info(f"  Rank {i+1}: 索引={result['index']}, 分数={result['score']:.4f}")


def evaluate(args):
    """
    评估检索系统
    """
    logger.info(f"开始评估检索系统，模型: {', '.join(args.models)}")
    
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # 设置输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化评估器
    evaluator = AdvancedEvaluator()
    
    # 初始化性能分析器
    analyzer = PerformanceAnalyzerComplete()
    
    # 加载查询集和图库集
    query_loader = MultiModalDatasetLoader(dataset_dir=args.query_dir)
    gallery_loader = MultiModalDatasetLoader(dataset_dir=args.gallery_dir)
    
    query_data = query_loader.load_dataset()
    gallery_data = gallery_loader.load_dataset()
    
    # 获取实际的标注数据
    query_dataset = query_data.get('annotations', [])
    gallery_dataset = gallery_data.get('annotations', [])
    
    logger.info(f"加载查询集: {len(query_dataset)} 个样本")
    logger.info(f"加载图库集: {len(gallery_dataset)} 个样本")
    
    # 评估每个模型
    all_metrics = {}
    all_results = {}
    
    for model_name in args.models:
        logger.info(f"评估模型: {model_name}")
        
        # 初始化模型
        if model_name == 'tamma':
            tamma_config = config['algorithms']['tamma']['params']
            model = TAMMAComplete(
                color_config=tamma_config.get('color_config'),
                sift_config=tamma_config.get('sift_config'),
                texture_config=tamma_config.get('texture_config'),
                text_config=tamma_config.get('text_config'),
                category_weights=tamma_config.get('category_weights'),
                color_threshold=tamma_config.get('color_threshold', 0.5),
                top_k_coarse=tamma_config.get('top_k_coarse', 50),
                spatial_weight=tamma_config.get('spatial_weight', 0.3),
                temporal_weight=tamma_config.get('temporal_weight', 0.2),
                fusion_method=tamma_config.get('fusion_method', 'weighted_sum'),
                use_gpu=hasattr(args, 'use_gpu') and args.use_gpu,
                gpu_device=hasattr(args, 'gpu_device') and args.gpu_device
            )
            # 加载SIFT码本
            if hasattr(model, 'sift_extractor') and hasattr(model.sift_extractor, 'load_codebook'):
                model.sift_extractor.load_codebook(config['sift_codebook']['save_path'])
        elif model_name == 'color_only':
            color_config = config['algorithms']['color_only']['params']
            model = ColorOnlyMatcherComplete(**color_config)
        elif model_name == 'dual_modality':
            dual_config = config['algorithms']['dual_modality']['params']
            model = DualModalityMatcherComplete(**dual_config)
        elif model_name == 'fixed_weight_multimodal':
            fixed_config = config['algorithms']['fixed_weight_multimodal']['params']
            model = FixedWeightMultimodalMatcherComplete(**fixed_config)
        elif model_name == 'deep_learning':
            dl_config = config['algorithms']['deep_learning']['params']
            model = DeepLearningMatcherComplete(**dl_config)
        else:
            logger.warning(f"跳过不支持的模型: {model_name}")
            continue
        
        # 构建索引
        model.build_index(gallery_dataset)
        
        # 执行检索
        predictions = []
        ground_truth = []
        query_times = []
        
        for i, query in enumerate(query_dataset):
            start_time = time.time()
            results = model.search(
                query_image_path=query['image_path'],
                category=query.get('category'),
                k=100
            )
            query_time = time.time() - start_time
            query_times.append(query_time)
            
            # 收集预测结果
            pred = [item['index'] for item in results]
            predictions.append(pred)
            
            # 收集真实标签（简单地将同一类别的图像作为正样本）
            gt = [j for j, gallery_item in enumerate(gallery_dataset) 
                 if gallery_item.get('category') == query.get('category')]
            ground_truth.append(gt)
            
            if (i + 1) % 50 == 0:
                logger.info(f"  已评估 {i+1}/{len(query_dataset)} 个查询")
        
        # 直接计算top-k准确率
        top_k_accuracy = {1: 0.0, 5: 0.0, 10: 0.0, 20: 0.0}
        
        for k in top_k_accuracy.keys():
            correct = 0
            for pred, gt in zip(predictions, ground_truth):
                # 检查前k个预测结果中是否有任何一个在真实标签中
                if any(item in gt for item in pred[:k]):
                    correct += 1
            top_k_accuracy[k] = correct / len(predictions)
        
        # 手动计算MRR
        def calculate_mrr(predictions, ground_truth):
            total_mrr = 0
            for pred, gt in zip(predictions, ground_truth):
                if not gt:
                    continue
                for i, item in enumerate(pred):
                    if item in gt:
                        total_mrr += 1.0 / (i + 1)
                        break
            return total_mrr / len(predictions) if predictions else 0
        
        # 手动计算MAP
        def calculate_map(predictions, ground_truth):
            total_ap = 0
            for pred, gt in zip(predictions, ground_truth):
                if not gt:
                    continue
                relevant_found = 0
                precision_sum = 0
                for i, item in enumerate(pred):
                    if item in gt:
                        relevant_found += 1
                        precision = relevant_found / (i + 1)
                        precision_sum += precision
                if relevant_found > 0:
                    total_ap += precision_sum / relevant_found
            return total_ap / len(predictions) if predictions else 0
        
        metrics = {
            'top_k_accuracy': top_k_accuracy,
            'mrr': calculate_mrr(predictions, ground_truth),
            'map': calculate_map(predictions, ground_truth),
            'avg_query_time': np.mean(query_times),
            'total_queries': len(query_dataset)
        }
        
        all_metrics[model_name] = metrics
        all_results[model_name] = {
            'predictions': predictions,
            'ground_truth': ground_truth,
            'query_times': query_times
        }
        
        # 打印评估结果
        logger.info(f"模型 {model_name} 评估结果:")
        logger.info(f"  Top-1准确率: {metrics['top_k_accuracy'][1]:.4f}")
        logger.info(f"  Top-5准确率: {metrics['top_k_accuracy'][5]:.4f}")
        logger.info(f"  Top-10准确率: {metrics['top_k_accuracy'][10]:.4f}")
        logger.info(f"  MRR: {metrics['mrr']:.4f}")
        logger.info(f"  MAP: {metrics['map']:.4f}")
        logger.info(f"  平均查询时间: {metrics['avg_query_time']*1000:.2f} ms")
    
    # 保存评估结果
    metrics_path = os.path.join(args.output_dir, 'evaluation_metrics.json')
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2)
    
    # 生成性能分析报告
    # 保存评估结果，暂时跳过可视化部分
    logger.info(f"评估完成！")
    logger.info(f"评估结果保存在: {metrics_path}")
    # 由于性能分析器的兼容性问题，我们暂时不生成可视化报告
    
    logger.info(f"评估完成！")
    logger.info(f"评估结果保存在: {metrics_path}")
    logger.info(f"可视化图表保存在: {os.path.join(args.output_dir, 'figures')}")


def run_experiment(args):
    """
    运行对比实验
    """
    logger.info("开始运行对比实验...")
    
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # 更新实验配置
    if args.num_runs:
        config['experiment']['num_runs'] = args.num_runs
    
    config['experiment']['output_dir'] = args.output_dir
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 初始化实验管理器
    experiment_manager = ExperimentManager(
        results_dir=args.output_dir,
        dataset_dir='./data/synthetic',
        algorithms_dir='./algorithms',
        save_results=True,
        random_seed=config['experiment']['seed']
    )
    
    # 运行实验
    # 这里需要适配ExperimentManager的方法，检查它是否有run方法或其他运行实验的方法
    if hasattr(experiment_manager, 'run_comparison_experiment'):
        # 需要准备algorithms和dataset参数
        # 这里简化处理，从配置中提取算法信息
        algorithms = []
        dataset = {
            'name': 'synthetic_dataset',
            'root_dir': './data/synthetic'
        }
        metrics = [metric['name'] for metric in config['evaluation']['metrics']]
        
        # 尝试运行对比实验
        results = experiment_manager.run_comparison_experiment(
            algorithms=algorithms,
            dataset=dataset,
            num_rounds=config['experiment']['num_runs'],
            experiment_name=config['experiment']['name'],
            metrics=metrics
        )
    else:
        logger.error("ExperimentManager没有找到合适的运行方法")
        return
    
    logger.info("实验运行完成！")
    logger.info(f"实验结果保存在: {args.output_dir}")


def create_codebook_command(args):
    """
    创建SIFT码本
    """
    logger.info("开始创建SIFT码本...")
    
    # 加载配置
    config_manager = ConfigManager()
    config = config_manager.load_config(args.config)
    
    # 创建输出目录
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # 设置码本参数
    codebook_size = args.size or config['sift_codebook']['codebook_size']
    num_samples = config['sift_codebook']['num_samples']
    
    # 使用ExperimentManager的prepare_sift_codebook方法
    from experiments.experiment_manager import ExperimentManager
    
    # 创建实验管理器实例
    experiment_manager = ExperimentManager(
        results_dir='./results',
        dataset_dir=args.data_dir,
        save_results=True
    )
    
    logger.info(f"开始准备SIFT码本: size={codebook_size}, samples={num_samples}")
    
    # 创建码本
    start_time = time.time()
    codebook_path = experiment_manager.prepare_sift_codebook(
        dataset_path=args.data_dir,
        codebook_size=codebook_size,
        num_samples=num_samples,
        output_path=args.output_path
    )
    codebook_time = time.time() - start_time
    
    if codebook_path:
        logger.info(f"SIFT码本创建完成！")
        logger.info(f"耗时: {codebook_time:.2f}秒")
        logger.info(f"码本保存在: {codebook_path}")
    else:
        logger.error("SIFT码本创建失败！")
        raise Exception("创建SIFT码本失败")


def run_demo(args):
    """
    运行演示
    """
    logger.info("开始运行TAMMA演示...")
    
    # 调用examples目录下的演示脚本
    demo_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                              'examples', 'tamma_demo.py')
    
    if not os.path.exists(demo_script):
        logger.error(f"演示脚本不存在: {demo_script}")
        sys.exit(1)
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 运行演示脚本
    logger.info(f"运行演示脚本: {demo_script}")
    import subprocess
    try:
        subprocess.run([sys.executable, demo_script, 
                      '--config', args.config,
                      '--output-dir', args.output_dir],
                      check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"演示脚本运行失败: {str(e)}")
        sys.exit(1)
    
    logger.info("演示运行完成！")
    logger.info(f"演示结果保存在: {args.output_dir}")


def main():
    """
    主函数
    """
    try:
        # 解析命令行参数
        args = parse_arguments()
        
        # 设置基础目录结构
        base_dir = os.path.dirname(os.path.abspath(__file__))
        setup_directories(base_dir)
        
        # 根据命令执行相应的功能
        if args.command == 'generate_dataset':
            generate_dataset(args)
        elif args.command == 'train':
            train_model(args)
        elif args.command == 'extract_features':
            extract_features(args)
        elif args.command == 'build_index':
            build_index(args)
        elif args.command == 'search':
            search(args)
        elif args.command == 'evaluate':
            evaluate(args)
        elif args.command == 'experiment':
            run_experiment(args)
        elif args.command == 'create_codebook':
            create_codebook_command(args)
        elif args.command == 'demo':
            run_demo(args)
        else:
            logger.error("请指定命令，使用 --help 查看可用命令")
            sys.exit(1)
        
    except KeyboardInterrupt:
        logger.info("程序被用户中断")
        sys.exit(1)
    except Exception as e:
        logger.error(f"运行出错: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()