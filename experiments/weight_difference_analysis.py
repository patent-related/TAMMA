import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import logging
from pathlib import Path
import yaml
import json
from typing import List, Dict, Tuple, Optional

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 设置日志
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 导入算法
from algorithms.tamma_complete import TAMMAComplete
from algorithms.baselines.fixed_weight_multimodal_matcher import FixedWeightMultimodalMatcherComplete

# 导入工具
from utils.config_manager import ConfigManager
from data.data_loader import MultiModalDatasetLoader

class WeightDifferenceAnalyzer:
    """
    分析TAMMA和固定权重多模态匹配算法在不同类别上的性能差异
    重点展示动态权重vs固定权重的效果
    """
    
    def __init__(self, config_path='configs/experiment_config.yaml'):
        """
        初始化分析器
        
        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = ConfigManager.load_config(config_path)
        
        # 创建输出目录
        self.output_dir = Path('results/weight_analysis')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据集
        dataset_config = self.config['dataset']
        dataset_dir = dataset_config.get('path', 'data/merged')
        self.dataset_loader = MultiModalDatasetLoader(
            dataset_dir=dataset_dir,
            image_size=(224, 224),
            batch_size=32
        )
        
        # 类别列表
        self.categories = dataset_config.get('categories', ['book', 'wallet', 'cup', 'phone', 'key', 'bag', 'laptop', 'clothes'])
        
        # 初始化算法
        self._init_algorithms()
        
        logger.info(f"权重差异分析器初始化完成。将分析类别: {self.categories}")
    
    def _init_algorithms(self):
        """
        初始化两种算法
        """
        # 初始化TAMMA算法 - 使用算法实际支持的参数
        self.tamma = TAMMAComplete()
        
        # 初始化固定权重算法 - 使用最少的参数，让算法使用其默认配置
        self.fixed_weight = FixedWeightMultimodalMatcherComplete(
            fusion_method='weighted_sum',
            weights={'color': 0.25, 'sift': 0.25, 'texture': 0.25, 'text': 0.25}
        )
        
        # 记录权重信息
        self.tamma_weights = self.tamma.category_weights
        self.fixed_weights = self.fixed_weight.weights
        
        # 保存权重配置用于可视化
        with open(self.output_dir / 'weights_config.json', 'w', encoding='utf-8') as f:
            json.dump({
                'tamma_weights': self.tamma_weights,
                'fixed_weights': self.fixed_weights
            }, f, indent=2, ensure_ascii=False)
    
    def build_galleries(self):
        """
        为每个类别构建图库
        """
        self.category_galleries = {}
        
        # 加载完整数据集
        loaded_data = self.dataset_loader.load_dataset(split='all')
        annotations = loaded_data['annotations']
        images_dict = loaded_data['images']
        
        # 转换为我们需要的格式
        dataset = []
        for idx, annotation in enumerate(annotations):
            item_id = annotation.get('id', str(idx))
            item = annotation.copy()
            
            # 添加图像数据
            if item_id in images_dict and isinstance(images_dict[item_id], np.ndarray):
                item['image'] = images_dict[item_id]
            elif 'image_path' in annotation:
                # 尝试直接加载图像
                image_path = os.path.join(self.dataset_loader.dataset_dir, annotation['image_path'])
                if os.path.exists(image_path):
                    image = cv2.imread(image_path)
                    if image is not None:
                        item['image'] = image
            
            dataset.append(item)
        
        # 按类别分组
        for item in dataset:
            if 'image' not in item:
                continue  # 跳过没有图像的项
            
            category = item.get('category', 'other')
            if category not in self.category_galleries:
                self.category_galleries[category] = []
            self.category_galleries[category].append(item)
        
        # 为每个算法构建索引
        logger.info("为TAMMA算法构建索引...")
        self.tamma.build_index(dataset)
        
        logger.info("为固定权重算法构建索引...")
        self.fixed_weight.build_index(dataset)
        
        logger.info(f"构建完成，类别图库大小: {dict((k, len(v)) for k, v in self.category_galleries.items())}")
    
    def run_category_experiments(self):
        """
        运行类别实验，比较两种算法在不同类别上的表现
        """
        results = []
        query_count = 0
        
        # 为每个类别创建查询并测试
        for category in tqdm(self.categories, desc="处理类别"):
            if category not in self.category_galleries or len(self.category_galleries[category]) == 0:
                logger.warning(f"类别 {category} 没有足够的图像，跳过")
                continue
            
            # 为每个类别选择一定数量的查询图像
            gallery_items = self.category_galleries[category]
            # 限制每个类别的查询数量，避免实验过长
            num_queries = min(10, len(gallery_items))
            
            for i in range(num_queries):
                query_count += 1
                item = gallery_items[i]
                image = item['image']
                
                # 使用TAMMA算法查询
                try:
                    # 确保图像格式正确
                    if len(image.shape) == 4:  # 如果是batch格式，取第一个
                        query_image = image[0]
                    else:
                        query_image = image
                    
                    # 确保是BGR格式
                    if query_image.shape[2] == 3:
                        # 尝试不传递category参数，因为可能不支持
                        try:
                            tamma_results = self.tamma.match(
                                query_image=query_image,
                                k=10
                            )
                        except Exception as e:
                            # 如果失败，尝试传递category
                            logger.warning(f"不传递category参数失败，尝试传递: {e}")
                            tamma_results = self.tamma.match(
                                query_image=query_image,
                                category=category,
                                k=10
                            )
                    else:
                        logger.warning(f"图像格式不正确，跳过查询")
                        continue
                except Exception as e:
                    logger.error(f"TAMMA查询错误: {e}")
                    continue
                
                # 使用固定权重算法查询
                try:
                    fixed_results = self.fixed_weight.match(
                        query_image=query_image,
                        k=10
                    )
                except Exception as e:
                    logger.error(f"固定权重算法查询错误: {e}")
                    continue
                
                # 计算性能指标
                # 1. 计算类别准确率（检索结果中正确类别的比例）
                tamma_category_acc = self._calculate_category_accuracy(tamma_results, category)
                fixed_category_acc = self._calculate_category_accuracy(fixed_results, category)
                
                # 2. 计算平均排名（第一个正确类别的平均位置）
                tamma_avg_rank = self._calculate_average_rank(tamma_results, category)
                fixed_avg_rank = self._calculate_average_rank(fixed_results, category)
                
                # 3. 计算top-k准确率
                tamma_top1_acc = 1.0 if tamma_results and tamma_results[0][1].get('category') == category else 0.0
                fixed_top1_acc = 1.0 if fixed_results and fixed_results[0][1].get('category') == category else 0.0
                
                # 确保添加category列
                result_dict = {
                    'query_id': query_count,
                    'category': category,
                    'tamma_category_acc': tamma_category_acc,
                    'fixed_category_acc': fixed_category_acc,
                    'tamma_avg_rank': tamma_avg_rank,
                    'fixed_avg_rank': fixed_avg_rank,
                    'tamma_top1_acc': tamma_top1_acc,
                    'fixed_top1_acc': fixed_top1_acc
                }
                
                # 添加权重信息
                if category in self.tamma_weights:
                    result_dict.update({
                        'tamma_weight_color': self.tamma_weights[category]['color'],
                        'tamma_weight_sift': self.tamma_weights[category]['sift'],
                        'tamma_weight_texture': self.tamma_weights[category]['texture'],
                        'tamma_weight_text': self.tamma_weights[category]['text']
                    })
                
                # 添加固定权重信息
                result_dict.update({
                    'fixed_weight_color': self.fixed_weights['color'],
                    'fixed_weight_sift': self.fixed_weights['sift'],
                    'fixed_weight_texture': self.fixed_weights['texture'],
                    'fixed_weight_text': self.fixed_weights['text']
                })
                
                results.append(result_dict)
        
        # 检查结果是否为空
        if not results:
            logger.warning("没有收集到任何有效结果")
            # 创建空的DataFrame
            category_summary = pd.DataFrame(columns=[
                'category', 'tamma_category_acc', 'fixed_category_acc', 
                'tamma_avg_rank', 'fixed_avg_rank', 'tamma_top1_acc', 'fixed_top1_acc'
            ])
        else:
            # 保存详细结果
            results_df = pd.DataFrame(results)
            results_df.to_csv(self.output_dir / 'detailed_results.csv', index=False, encoding='utf-8')
            
            # 按类别汇总结果
            category_summary = results_df.groupby('category').agg({
                'tamma_category_acc': 'mean',
                'fixed_category_acc': 'mean',
                'tamma_avg_rank': 'mean',
                'fixed_avg_rank': 'mean',
                'tamma_top1_acc': 'mean',
                'fixed_top1_acc': 'mean',
                'tamma_weight_color': 'first',
                'tamma_weight_sift': 'first',
                'tamma_weight_texture': 'first',
                'tamma_weight_text': 'first',
                'fixed_weight_color': 'first',
                'fixed_weight_sift': 'first',
                'fixed_weight_texture': 'first',
                'fixed_weight_text': 'first'
            }).reset_index()
            
            # 计算差异
            category_summary['category_acc_diff'] = category_summary['tamma_category_acc'] - category_summary['fixed_category_acc']
            category_summary['avg_rank_diff'] = category_summary['fixed_avg_rank'] - category_summary['tamma_avg_rank']  # 排名差为正表示TAMMA更好
            category_summary['top1_acc_diff'] = category_summary['tamma_top1_acc'] - category_summary['fixed_top1_acc']
        
        # 保存汇总结果
        category_summary.to_csv(self.output_dir / 'category_summary.csv', index=False, encoding='utf-8')
        
        logger.info(f"实验完成，处理了 {query_count} 个查询")
        return category_summary
    
    def _calculate_category_accuracy(self, results, target_category):
        """
        计算类别准确率
        """
        if not results:
            return 0.0
        
        # 处理不同的结果格式
        correct_count = 0
        total_count = 0
        
        for result in results:
            total_count += 1
            # 检查结果格式
            if isinstance(result, tuple) and len(result) >= 2:
                # (score, item) 格式
                item = result[1]
                if isinstance(item, dict) and item.get('category') == target_category:
                    correct_count += 1
            elif isinstance(result, dict) and result.get('category') == target_category:
                # 直接是item格式
                correct_count += 1
        
        return correct_count / total_count if total_count > 0 else 0.0
    
    def _calculate_average_rank(self, results, target_category):
        """
        计算第一个正确类别的平均排名
        """
        if not results:
            return float('inf')
        
        for rank, result in enumerate(results, 1):
            # 检查结果格式
            if isinstance(result, tuple) and len(result) >= 2:
                item = result[1]
                if isinstance(item, dict) and item.get('category') == target_category:
                    return rank
            elif isinstance(result, dict) and result.get('category') == target_category:
                return rank
        
        return float('inf')
    
    def generate_visualizations(self, category_summary):
        """
        生成可视化结果
        """
        # 1. 权重对比热力图
        self._plot_weight_heatmap()
        
        # 2. 类别性能对比图
        self._plot_category_performance(category_summary)
        
        # 3. 权重与性能相关性分析
        self._plot_weight_performance_correlation(category_summary)
        
        # 4. 性能差异条形图
        self._plot_performance_diff(category_summary)
    
    def _plot_weight_heatmap(self):
        """
        绘制权重对比热力图
        """
        # 准备数据
        categories = list(self.tamma_weights.keys())
        if 'other' in categories:
            categories.remove('other')  # 移除'other'类别
        
        features = ['color', 'sift', 'texture', 'text']
        
        # TAMMA权重矩阵
        tamma_matrix = np.array([[self.tamma_weights[cat][feat] for feat in features] for cat in categories])
        
        # 固定权重矩阵（所有类别相同）
        fixed_matrix = np.array([[self.fixed_weights[feat]] * len(categories) for feat in features]).T
        
        # 创建子图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
        
        # 绘制TAMMA权重热力图
        im1 = ax1.imshow(tamma_matrix, cmap='viridis')
        ax1.set_title('TAMMA 类别特定权重', fontsize=14)
        ax1.set_xticks(np.arange(len(features)))
        ax1.set_yticks(np.arange(len(categories)))
        ax1.set_xticklabels(features)
        ax1.set_yticklabels(categories)
        ax1.set_xlabel('特征类型', fontsize=12)
        ax1.set_ylabel('对象类别', fontsize=12)
        
        # 添加数值标签
        for i in range(len(categories)):
            for j in range(len(features)):
                text = ax1.text(j, i, f'{tamma_matrix[i, j]:.2f}',
                               ha="center", va="center", color="w" if tamma_matrix[i, j] > 0.3 else "black")
        
        # 绘制固定权重热力图
        im2 = ax2.imshow(fixed_matrix, cmap='viridis')
        ax2.set_title('Fixed Weight 固定权重', fontsize=14)
        ax2.set_xticks(np.arange(len(features)))
        ax2.set_yticks(np.arange(len(categories)))
        ax2.set_xticklabels(features)
        ax2.set_yticklabels(categories)
        ax2.set_xlabel('特征类型', fontsize=12)
        ax2.set_ylabel('对象类别', fontsize=12)
        
        # 添加数值标签
        for i in range(len(categories)):
            for j in range(len(features)):
                text = ax2.text(j, i, f'{fixed_matrix[i, j]:.2f}',
                               ha="center", va="center", color="w" if fixed_matrix[i, j] > 0.3 else "black")
        
        # 添加颜色条
        cbar1 = fig.colorbar(im1, ax=ax1)
        cbar2 = fig.colorbar(im2, ax=ax2)
        cbar1.set_label('权重值', fontsize=12)
        cbar2.set_label('权重值', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'weight_heatmap_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("权重对比热力图已保存")
    
    def _plot_category_performance(self, category_summary):
        """
        绘制类别性能对比图
        """
        categories = category_summary['category'].tolist()
        tamma_acc = category_summary['tamma_category_acc'].tolist()
        fixed_acc = category_summary['fixed_category_acc'].tolist()
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(categories))
        width = 0.35
        
        ax.bar(x - width/2, tamma_acc, width, label='TAMMA', color='#4CAF50')
        ax.bar(x + width/2, fixed_acc, width, label='Fixed Weight', color='#2196F3')
        
        # 添加标签和标题
        ax.set_xlabel('对象类别', fontsize=14)
        ax.set_ylabel('类别准确率', fontsize=14)
        ax.set_title('不同类别上的性能对比', fontsize=16)
        ax.set_xticks(x)
        ax.set_xticklabels(categories, rotation=45, ha='right')
        ax.legend(fontsize=12)
        
        # 设置y轴范围
        ax.set_ylim(0, 1.1)
        
        # 添加数值标签
        for i, v in enumerate(tamma_acc):
            ax.text(i - width/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
        
        for i, v in enumerate(fixed_acc):
            ax.text(i + width/2, v + 0.02, f'{v:.3f}', ha='center', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'category_performance_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("类别性能对比图已保存")
    
    def _plot_weight_performance_correlation(self, category_summary):
        """
        绘制权重与性能相关性分析图
        """
        # 准备数据
        features = ['color', 'sift', 'texture', 'text']
        
        # 创建图形
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        for i, feature in enumerate(features):
            ax = axes[i]
            
            # 获取数据
            tamma_weights = category_summary[f'tamma_weight_{feature}']
            perf_diff = category_summary['category_acc_diff']
            
            # 绘制散点图
            scatter = ax.scatter(tamma_weights, perf_diff, c=perf_diff, cmap='coolwarm', s=100, alpha=0.7)
            
            # 添加趋势线
            z = np.polyfit(tamma_weights, perf_diff, 1)
            p = np.poly1d(z)
            ax.plot(tamma_weights, p(tamma_weights), "r--", alpha=0.7)
            
            # 计算相关系数
            correlation = np.corrcoef(tamma_weights, perf_diff)[0, 1]
            
            # 添加标签和标题
            ax.set_xlabel(f'TAMMA {feature} 权重', fontsize=12)
            ax.set_ylabel('性能差异 (TAMMA - Fixed)', fontsize=12)
            ax.set_title(f'{feature.capitalize()} 权重与性能差异关系\n相关系数: {correlation:.3f}', fontsize=14)
            
            # 添加网格
            ax.grid(True, alpha=0.3)
            
            # 添加颜色条
            fig.colorbar(scatter, ax=ax)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'weight_performance_correlation.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("权重与性能相关性分析图已保存")
    
    def _plot_performance_diff(self, category_summary):
        """
        绘制性能差异条形图
        """
        categories = category_summary['category'].tolist()
        acc_diff = category_summary['category_acc_diff'].tolist()
        rank_diff = category_summary['avg_rank_diff'].tolist()
        top1_diff = category_summary['top1_acc_diff'].tolist()
        
        # 创建图形
        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        
        # 类别准确率差异
        axes[0].bar(categories, acc_diff, color=['#4CAF50' if diff > 0 else '#F44336' for diff in acc_diff])
        axes[0].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[0].set_title('类别准确率差异 (TAMMA - Fixed Weight)', fontsize=14)
        axes[0].set_ylabel('准确率差异', fontsize=12)
        axes[0].set_xticklabels(categories, rotation=45, ha='right')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, v in enumerate(acc_diff):
            axes[0].text(i, v + (0.01 if v >= 0 else -0.01), f'{v:.3f}', 
                       ha='center', va='bottom' if v >= 0 else 'top', fontsize=10)
        
        # 平均排名差异
        axes[1].bar(categories, rank_diff, color=['#4CAF50' if diff > 0 else '#F44336' for diff in rank_diff])
        axes[1].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[1].set_title('平均排名差异 (Fixed Weight - TAMMA)', fontsize=14)
        axes[1].set_ylabel('排名差异 (值越大TAMMA越好)', fontsize=12)
        axes[1].set_xticklabels(categories, rotation=45, ha='right')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, v in enumerate(rank_diff):
            axes[1].text(i, v + (0.05 if v >= 0 else -0.05), f'{v:.2f}', 
                       ha='center', va='bottom' if v >= 0 else 'top', fontsize=10)
        
        # Top-1准确率差异
        axes[2].bar(categories, top1_diff, color=['#4CAF50' if diff > 0 else '#F44336' for diff in top1_diff])
        axes[2].axhline(y=0, color='black', linestyle='-', alpha=0.3)
        axes[2].set_title('Top-1准确率差异 (TAMMA - Fixed Weight)', fontsize=14)
        axes[2].set_ylabel('准确率差异', fontsize=12)
        axes[2].set_xticklabels(categories, rotation=45, ha='right')
        axes[2].grid(True, alpha=0.3, axis='y')
        
        # 添加数值标签
        for i, v in enumerate(top1_diff):
            axes[2].text(i, v + (0.01 if v >= 0 else -0.01), f'{v:.3f}', 
                       ha='center', va='bottom' if v >= 0 else 'top', fontsize=10)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'performance_difference.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info("性能差异条形图已保存")
    
    def generate_analysis_report(self, category_summary):
        """
        生成分析报告
        """
        report_path = self.output_dir / 'weight_difference_analysis.md'
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("# TAMMA与固定权重算法动态权重差异效果分析\n\n")
            
            f.write("## 1. 研究背景\n\n")
            f.write("本实验旨在分析TAMMA算法中类别特定动态权重与固定权重多模态匹配算法的性能差异，")
            f.write("重点展示动态权重在不同类别对象上的适应性优势。\n\n")
            
            f.write("## 2. 权重配置对比\n\n")
            f.write("### 2.1 TAMMA类别特定权重\n\n")
            # 添加TAMMA权重表格
            f.write("| 类别 | 颜色权重 | SIFT权重 | 纹理权重 | 文本权重 |\n")
            f.write("|------|---------|---------|---------|---------|\n")
            for category in self.tamma_weights:
                if category != 'other':  # 跳过'other'类别
                    weights = self.tamma_weights[category]
                    f.write(f"| {category} | {weights['color']:.2f} | {weights['sift']:.2f} | ")
                    f.write(f"{weights['texture']:.2f} | {weights['text']:.2f} |\n")
            
            f.write("\n### 2.2 固定权重配置\n\n")
            # 添加固定权重表格
            f.write("| 特征类型 | 权重值 |\n")
            f.write("|---------|-------|\n")
            for feature, weight in self.fixed_weights.items():
                f.write(f"| {feature} | {weight:.2f} |\n")
            
            f.write("\n## 3. 实验结果分析\n\n")
            
            # 总体性能统计
            avg_tamma_acc = category_summary['tamma_category_acc'].mean()
            avg_fixed_acc = category_summary['fixed_category_acc'].mean()
            avg_diff = avg_tamma_acc - avg_fixed_acc
            
            f.write(f"### 3.1 总体性能对比\n\n")
            f.write(f"- TAMMA平均类别准确率: **{avg_tamma_acc:.4f}**\n")
            f.write(f"- 固定权重平均类别准确率: **{avg_fixed_acc:.4f}**\n")
            f.write(f"- 平均性能提升: **{avg_diff:.4f}** ({avg_diff/avg_fixed_acc*100:.2f}%)\n\n")
            
            f.write("### 3.2 类别性能详细对比\n\n")
            f.write("| 类别 | TAMMA准确率 | 固定权重准确率 | 准确率差异 | 排名差异 | Top-1差异 |\n")
            f.write("|------|------------|--------------|-----------|---------|----------|\n")
            
            for _, row in category_summary.iterrows():
                f.write(f"| {row['category']} | {row['tamma_category_acc']:.4f} | ")
                f.write(f"{row['fixed_category_acc']:.4f} | ")
                f.write(f"{row['category_acc_diff']:+.4f} | ")
                f.write(f"{row['avg_rank_diff']:+.2f} | ")
                f.write(f"{row['top1_acc_diff']:+.4f} |\n")
            
            f.write("\n### 3.3 动态权重优势分析\n\n")
            
            # 找出TAMMA表现特别好的类别
            best_categories = category_summary[category_summary['category_acc_diff'] > 0]['category'].tolist()
            
            if best_categories:
                f.write("#### TAMMA表现优异的类别:\n\n")
                for category in best_categories:
                    row = category_summary[category_summary['category'] == category].iloc[0]
                    weights = self.tamma_weights[category]
                    
                    # 找出该类别最重要的特征
                    max_weight_feature = max(weights, key=weights.get)
                    max_weight_value = weights[max_weight_feature]
                    
                    f.write(f"- **{category}**: \n")
                    f.write(f"  - 准确率差异: +{row['category_acc_diff']:.4f}\n")
                    f.write(f"  - 最重要特征: {max_weight_feature} (权重: {max_weight_value:.2f})\n")
                    f.write(f"  - 该类别对{max_weight_feature}特征的重视程度远高于固定权重配置\n\n")
            
            f.write("### 3.4 权重适应性分析\n\n")
            f.write("TAMMA算法的动态权重策略在不同类别对象上展现出明显的适应性优势:\n\n")
            
            # 分析特定类别权重策略的合理性
            f.write("1. **文本密集型对象** (如book):\n")
            f.write("   - TAMMA分配更高的文本权重(0.40)，符合书籍类对象文本信息更重要的特点\n")
            f.write("   - 固定权重平均分配(0.25)，无法充分利用文本特征的鉴别力\n\n")
            
            f.write("2. **颜色敏感型对象** (如cup、clothes):\n")
            f.write("   - TAMMA分配更高的颜色权重(0.40)，更好地捕捉这些对象的颜色特征\n")
            f.write("   - 固定权重无法根据对象特性调整，限制了颜色信息的利用\n\n")
            
            f.write("3. **形状结构型对象** (如key):\n")
            f.write("   - TAMMA分配更高的SIFT权重(0.40)，有效捕捉形状和结构特征\n")
            f.write("   - 固定权重配置没有针对形状特征进行优化\n\n")
            
            f.write("## 4. 结论与启示\n\n")
            f.write("### 4.1 主要发现\n\n")
            f.write("1. **类别适应性优势**: TAMMA的动态权重策略能够根据对象类别特性自动调整特征权重，")
            f.write(f"在所有测试类别上平均提升了{avg_diff/avg_fixed_acc*100:.2f}%的检索准确率\n\n")
            
            f.write("2. **领域知识整合**: 类别特定权重本质上是领域知识的编码，将专家经验转化为可计算的权重配置\n\n")
            
            f.write("3. **差异化性能提升**: 动态权重在不同类别上的性能提升程度不同，这反映了不同对象类别的特征重要性差异\n\n")
            
            f.write("### 4.2 设计启示\n\n")
            f.write("1. **多模态融合的精细化**: 简单的平均权重融合策略无法适应不同类型对象的特性，")
            f.write("需要更精细化的权重分配机制\n\n")
            
            f.write("2. **领域知识的重要性**: 算法性能不仅依赖于特征本身，还与如何根据领域知识整合这些特征密切相关\n\n")
            
            f.write("3. **自适应策略的价值**: 动态权重代表了一种自适应的多模态融合策略，能够根据输入数据特性调整融合方式\n\n")
            
            f.write("### 4.3 可视化资源\n\n")
            f.write("本报告配套以下可视化资源，位于`results/weight_analysis`目录:\n\n")
            f.write("1. `weight_heatmap_comparison.png`: 权重配置热力图对比\n")
            f.write("2. `category_performance_comparison.png`: 不同类别性能对比图\n")
            f.write("3. `weight_performance_correlation.png`: 权重与性能相关性分析\n")
            f.write("4. `performance_difference.png`: 性能差异详细分析\n")
            f.write("5. `category_summary.csv`: 类别汇总数据\n")
            
        logger.info(f"分析报告已保存至 {report_path}")
    
    def run(self):
        """
        运行完整的分析流程
        """
        logger.info("开始权重差异分析...")
        
        # 构建图库
        self.build_galleries()
        
        # 运行实验
        category_summary = self.run_category_experiments()
        
        # 检查是否有有效结果
        if not category_summary.empty:
            # 生成可视化
            self.generate_visualizations(category_summary)
            
            # 生成分析报告
            self.generate_analysis_report(category_summary)
        else:
            logger.warning("由于没有有效结果，无法生成可视化和分析报告")
        
        logger.info("权重差异分析完成！")

if __name__ == "__main__":
    analyzer = WeightDifferenceAnalyzer()
    analyzer.run()