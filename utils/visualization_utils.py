#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
可视化工具模块

提供TAMMA多模态检索系统的评估结果可视化功能
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any
import logging
import matplotlib.font_manager
from matplotlib.font_manager import FontProperties

logger = logging.getLogger("visualization_utils")

# 配置matplotlib以支持中文显示
def setup_matplotlib_for_chinese():
    """
    配置matplotlib以正确显示中文，使用系统中已确认存在的字体
    """
    # 忽略字体警告
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib")
    warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")
    
    # 使用Agg后端
    plt.switch_backend('Agg')
    
    # 刷新字体缓存
    matplotlib.font_manager._load_fontmanager()
    
    # 强制设置字体配置
    plt.rcParams['font.family'] = ['sans-serif']
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei Mono', 'Noto Sans CJK SC', 'AR PL UMing CN', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False  # 确保负号正确显示
    plt.rcParams['figure.facecolor'] = 'white'  # 设置图表背景为白色
    plt.rcParams['savefig.facecolor'] = 'white'  # 保存的图像背景为白色
    plt.rcParams['savefig.bbox'] = 'tight'  # 确保保存的图像包含所有元素
    plt.rcParams['savefig.pad_inches'] = 0.1  # 设置保存图像的边距
    plt.rcParams['figure.dpi'] = 300  # 设置默认DPI为300，提高图像质量
    
    logger.info("已配置matplotlib使用系统中文字体支持中文显示")
    return True

# 初始化matplotlib配置
setup_matplotlib_for_chinese()

class SimpleVisualizer:
    """简单可视化工具类"""
    
    def __init__(self, output_dir: str = './visualizations'):
        """
        初始化可视化工具
        
        Args:
            output_dir: 图表输出目录
        """
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 再次设置字体，确保在类实例化时字体配置正确
        setup_matplotlib_for_chinese()
        
        # 设置样式
        sns.set_style("whitegrid")
        sns.set_palette("viridis")
        
        # 创建字体属性对象
        self.font_props = self._get_chinese_font_properties()
        
        logger.info(f"初始化可视化工具: output_dir={self.output_dir}")
        
    def _get_chinese_font_properties(self):
        """获取支持中文的字体属性"""
        # 尝试使用系统中已确认存在的字体
        for font_name in ['WenQuanYi Micro Hei Mono', 'Noto Sans CJK SC', 'AR PL UMing CN']:
            try:
                font_props = FontProperties(family=font_name)
                # 测试字体是否可用
                if font_props.get_name() != 'DejaVu Sans':
                    logger.info(f"成功使用字体: {font_name}")
                    return font_props
            except:
                continue
        
        # 默认回退
        logger.warning("未找到理想的中文字体，使用默认字体")
        return FontProperties()
    
    def load_metrics(self, metrics_path: str) -> Dict[str, Any]:
        """
        加载评估指标
        
        Args:
            metrics_path: 指标文件路径
            
        Returns:
            评估指标数据
        """
        try:
            with open(metrics_path, 'r', encoding='utf-8') as f:
                metrics = json.load(f)
            logger.info(f"成功加载指标文件: {metrics_path}")
            return metrics
        except Exception as e:
            logger.error(f"加载指标文件失败: {str(e)}")
            raise
    
    def plot_top_k_accuracy(self, metrics: Dict[str, Any], save_path: str = None) -> str:
        """
        绘制Top-k准确率对比图
        
        Args:
            metrics: 评估指标数据
            save_path: 保存路径（可选）
            
        Returns:
            保存的文件路径
        """
        logger.info("绘制Top-k准确率对比图")
        
        # 创建新的图形对象
        fig, ax = plt.subplots(figsize=(10, 6))
        
        models = list(metrics.keys())
        k_values = [1, 5, 10, 20]
        
        # 为每个模型准备数据
        for model in models:
            accuracies = [metrics[model]['top_k_accuracy'].get(str(k), 0) for k in k_values]
            ax.plot(k_values, accuracies, marker='o', linewidth=2, label=model)
        
        # 设置图表属性
        ax.set_title('Top-k准确率对比', fontsize=16, fontproperties=self.font_props)
        ax.set_xlabel('k值', fontsize=12, fontproperties=self.font_props)
        ax.set_ylabel('准确率', fontsize=12, fontproperties=self.font_props)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 设置图例
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, title='模型', loc='best', prop=self.font_props)
        
        # 设置刻度标签字体
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(self.font_props)
        
        ax.set_xticks(k_values)
        ax.set_ylim(0, 1.1)
        
        # 保存图表
        if not save_path:
            save_path = os.path.join(self.output_dir, 'top_k_accuracy.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Top-k准确率对比图已保存到: {save_path}")
        return save_path
    
    def plot_metrics_comparison(self, metrics: Dict[str, Any], save_path: str = None) -> str:
        """
        绘制MRR和MAP指标对比图
        
        Args:
            metrics: 评估指标数据
            save_path: 保存路径（可选）
            
        Returns:
            保存的文件路径
        """
        logger.info("绘制MRR和MAP指标对比图")
        
        models = list(metrics.keys())
        metrics_to_plot = ['mrr', 'map']
        metrics_labels = {'mrr': 'MRR', 'map': 'MAP'}
        
        # 创建柱状图
        x = np.arange(len(models))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for i, metric in enumerate(metrics_to_plot):
            values = [metrics[model][metric] for model in models]
            ax.bar(x + i * width, values, width, label=metrics_labels[metric])
        
        # 设置图表属性
        ax.set_title('MRR和MAP指标对比', fontsize=16, fontproperties=self.font_props)
        ax.set_xlabel('模型', fontsize=12, fontproperties=self.font_props)
        ax.set_ylabel('指标值', fontsize=12, fontproperties=self.font_props)
        ax.set_xticks(x + width / 2)
        ax.set_xticklabels(models, fontproperties=self.font_props)
        
        # 设置图例
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles, labels, prop=self.font_props)
        
        # 设置刻度标签字体
        for label in ax.get_yticklabels():
            label.set_fontproperties(self.font_props)
        
        ax.set_ylim(0, 1.1)
        
        # 添加数值标签
        for rect in ax.patches:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9,
                        fontproperties=self.font_props)
        
        # 保存图表
        if not save_path:
            save_path = os.path.join(self.output_dir, 'metrics_comparison.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"MRR和MAP指标对比图已保存到: {save_path}")
        return save_path
    
    def plot_query_time(self, metrics: Dict[str, Any], save_path: str = None) -> str:
        """
        绘制平均查询时间对比图
        
        Args:
            metrics: 评估指标数据
            save_path: 保存路径（可选）
            
        Returns:
            保存的文件路径
        """
        logger.info("绘制平均查询时间对比图")
        
        models = list(metrics.keys())
        times = [metrics[model]['avg_query_time'] * 1000 for model in models]  # 转换为毫秒
        
        # 创建柱状图
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(models, times, color=sns.color_palette("viridis", len(models)))
        
        # 设置图表属性
        ax.set_title('平均查询时间对比（毫秒）', fontsize=16, fontproperties=self.font_props)
        ax.set_xlabel('模型', fontsize=12, fontproperties=self.font_props)
        ax.set_ylabel('平均查询时间（毫秒）', fontsize=12, fontproperties=self.font_props)
        ax.grid(True, linestyle='--', alpha=0.7, axis='y')
        
        # 设置刻度标签字体
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontproperties(self.font_props)
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}ms',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),  # 3点垂直偏移
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9,
                        fontproperties=self.font_props)
        
        # 保存图表
        if not save_path:
            save_path = os.path.join(self.output_dir, 'query_time.png')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"平均查询时间对比图已保存到: {save_path}")
        return save_path
    
    def generate_all_visualizations(self, metrics: Dict[str, Any]) -> List[str]:
        """
        生成所有可视化图表
        
        Args:
            metrics: 评估指标数据
            
        Returns:
            生成的图表文件路径列表
        """
        logger.info("开始生成所有可视化图表")
        
        saved_files = []
        
        # 生成Top-k准确率图
        saved_files.append(self.plot_top_k_accuracy(metrics))
        
        # 生成MRR和MAP对比图
        saved_files.append(self.plot_metrics_comparison(metrics))
        
        # 生成查询时间对比图
        saved_files.append(self.plot_query_time(metrics))
        
        logger.info(f"所有可视化图表已生成，共 {len(saved_files)} 个文件")
        return saved_files
    
    def generate_summary_html(self, metrics: Dict[str, Any], image_paths: List[str]) -> str:
        """
        生成HTML可视化报告
        
        Args:
            metrics: 评估指标数据
            image_paths: 图表文件路径列表
            
        Returns:
            HTML报告文件路径
        """
        logger.info("生成HTML可视化报告")
        
        # 生成HTML内容
        html_content = f'''
        <!DOCTYPE html>
        <html lang="zh-CN">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>TAMMA多模态检索系统 - 评估报告</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    margin: 0;
                    padding: 20px;
                    background-color: #f5f5f5;
                }}
                .container {{
                    max-width: 1200px;
                    margin: 0 auto;
                    background-color: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 0 10px rgba(0,0,0,0.1);
                }}
                h1, h2 {{ color: #333; }}
                table {{
                    width: 100%;
                    border-collapse: collapse;
                    margin: 20px 0;
                }}
                th, td {{ 
                    padding: 12px;
                    text-align: center;
                    border-bottom: 1px solid #ddd;
                }}
                th {{ background-color: #4CAF50; color: white; }}
                tr:hover {{ background-color: #f5f5f5; }}
                .visualization {{
                    margin: 30px 0;
                    text-align: center;
                }}
                .visualization img {{
                    max-width: 100%;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                .metric-highlight {{
                    font-size: 18px;
                    font-weight: bold;
                    color: #2196F3;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>TAMMA多模态检索系统 - 评估报告</h1>
                
                <h2>1. 评估指标摘要</h2>
                <table>
                    <tr>
                        <th>模型</th>
                        <th>Top-1准确率</th>
                        <th>Top-5准确率</th>
                        <th>Top-10准确率</th>
                        <th>MRR</th>
                        <th>MAP</th>
                        <th>平均查询时间</th>
                    </tr>
        '''
        
        # 添加指标数据
        for model, model_metrics in metrics.items():
            html_content += f'''
                    <tr>
                        <td>{model}</td>
                        <td>{model_metrics['top_k_accuracy'].get('1', 0):.4f}</td>
                        <td>{model_metrics['top_k_accuracy'].get('5', 0):.4f}</td>
                        <td>{model_metrics['top_k_accuracy'].get('10', 0):.4f}</td>
                        <td>{model_metrics['mrr']:.4f}</td>
                        <td>{model_metrics['map']:.4f}</td>
                        <td>{model_metrics['avg_query_time'] * 1000:.2f}ms</td>
                    </tr>
            '''
        
        html_content += '''
                </table>
                
                <h2>2. 可视化图表</h2>
        '''
        
        # 添加图表
        chart_titles = {
            'top_k_accuracy.png': 'Top-k准确率对比',
            'metrics_comparison.png': 'MRR和MAP指标对比',
            'query_time.png': '平均查询时间对比'
        }
        
        for img_path in image_paths:
            img_name = os.path.basename(img_path)
            title = chart_titles.get(img_name, img_name)
            rel_path = os.path.relpath(img_path, self.output_dir)
            html_content += f'''
                <div class="visualization">
                    <h3>{title}</h3>
                    <img src="{rel_path}" alt="{title}">
                </div>
            '''
        
        html_content += '''
            </div>
        </body>
        </html>
        '''
        
        # 保存HTML文件
        html_path = os.path.join(self.output_dir, 'evaluation_report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"HTML可视化报告已生成: {html_path}")
        return html_path
    
    def visualize_from_file(self, metrics_path: str) -> str:
        """
        从指标文件生成所有可视化
        
        Args:
            metrics_path: 指标文件路径
            
        Returns:
            HTML报告文件路径
        """
        # 加载指标
        metrics = self.load_metrics(metrics_path)
        
        # 生成所有图表
        image_paths = self.generate_all_visualizations(metrics)
        
        # 生成HTML报告
        html_path = self.generate_summary_html(metrics, image_paths)
        
        return html_path

def visualize_metrics(metrics_path: str, output_dir: str = './visualizations') -> str:
    """
    可视化评估指标的便捷函数
    
    Args:
        metrics_path: 指标文件路径
        output_dir: 输出目录
        
    Returns:
        HTML报告文件路径
    """
    visualizer = SimpleVisualizer(output_dir)
    return visualizer.visualize_from_file(metrics_path)