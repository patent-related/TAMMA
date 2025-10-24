import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any
import logging
import os
from collections import defaultdict

# 设置中文显示
try:
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False    # 用来正常显示负号
except Exception as e:
    logging.warning(f"无法设置中文字体: {e}")

logger = logging.getLogger(__name__)

class PerformanceAnalyzerComplete:
    """
    性能分析器
    
    支持准确率对比、速度对比、难度分析、Recall曲线、MRR对比、时间-准确率权衡、雷达图和箱线图等8种可视化分析方法
    """
    
    def __init__(self, 
                 results_dir: str = './results/figures',
                 dpi: int = 300,
                 figsize: Tuple[int, int] = (10, 6),
                 palette: Optional[str] = None):
        """
        Args:
            results_dir: 结果保存目录
            dpi: 图像分辨率
            figsize: 默认图像大小
            palette: 调色板
        """
        self.results_dir = results_dir
        self.dpi = dpi
        self.figsize = figsize
        self.palette = palette or 'viridis'
        
        # 创建结果目录
        os.makedirs(self.results_dir, exist_ok=True)
        
        # 设置样式
        sns.set_style("whitegrid")
        sns.set_palette(self.palette)
        
        logger.info(f"初始化性能分析器: results_dir={self.results_dir}")
    
    def compare_accuracy(self, 
                        algorithm_results: List[Dict[str, Any]],
                        metrics: List[str] = None,
                        show: bool = True,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        对比不同算法的准确率指标
        
        Args:
            algorithm_results: 算法评估结果列表
            metrics: 要比较的指标列表
            show: 是否显示图像
            save_path: 保存路径
            
        Returns:
            matplotlib Figure对象
        """
        logger.info("开始准确率对比分析")
        
        # 默认指标
        if metrics is None:
            metrics = ['precision@1', 'precision@5', 'precision@10', 'map', 'mrr']
        
        # 准备数据
        data = []
        
        for result in algorithm_results:
            algo_name = result.get('algorithm_name', 'Unknown')
            avg_metrics = result.get('average_metrics', {})
            
            for metric in metrics:
                if metric in avg_metrics:
                    data.append({
                        'algorithm': algo_name,
                        'metric': self._format_metric_name(metric),
                        'value': avg_metrics[metric]['mean'],
                        'std': avg_metrics[metric]['std']
                    })
        
        if not data:
            logger.warning("没有找到足够的数据进行准确率对比")
            return None
        
        # 转换为DataFrame
        df = pd.DataFrame(data)
        
        # 创建图像
        plt.figure(figsize=(12, 6))
        
        # 绘制柱状图
        ax = sns.barplot(
            x='metric', 
            y='value', 
            hue='algorithm',
            data=df,
            capsize=0.1,
            errwidth=1.5
        )
        
        # 添加误差线
        for i, (algorithm, group) in enumerate(df.groupby('algorithm')):
            for j, row in group.iterrows():
                x_pos = metrics.index(row['metric'].replace('@', '@')) * len(algorithm_results) + i
                ax.errorbar(
                    x_pos, 
                    row['value'], 
                    yerr=row['std'],
                    fmt='none',
                    color='black',
                    capsize=3,
                    elinewidth=1
                )
        
        # 设置标签和标题
        plt.title('算法准确率指标对比', fontsize=16)
        plt.xlabel('评估指标', fontsize=12)
        plt.ylabel('指标值', fontsize=12)
        plt.ylim(0, 1.1)  # 设置y轴范围
        
        # 添加数值标签
        for p in ax.patches:
            height = p.get_height()
            ax.text(
                p.get_x() + p.get_width() / 2.,
                height + 0.02,
                f'{height:.3f}',
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        # 调整图例
        plt.legend(title='算法', loc='best')
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            full_path = os.path.join(self.results_dir, save_path)
            plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"准确率对比图已保存到: {full_path}")
        
        if show:
            plt.show()
        
        return plt.gcf()
    
    def compare_speed(self, 
                     algorithm_times: Dict[str, Tuple[float, float]],
                     show: bool = True,
                     save_path: Optional[str] = None) -> plt.Figure:
        """
        对比不同算法的速度
        
        Args:
            algorithm_times: {算法名: (平均时间, 标准差)}
            show: 是否显示图像
            save_path: 保存路径
            
        Returns:
            matplotlib Figure对象
        """
        logger.info("开始速度对比分析")
        
        # 准备数据
        algorithms = list(algorithm_times.keys())
        avg_times = [t[0] for t in algorithm_times.values()]
        std_times = [t[1] for t in algorithm_times.values()]
        
        # 创建图像
        plt.figure(figsize=self.figsize)
        
        # 绘制柱状图
        bars = plt.bar(algorithms, avg_times, yerr=std_times, capsize=5, alpha=0.8)
        
        # 设置颜色
        colors = plt.cm.viridis(np.linspace(0, 1, len(algorithms)))
        for bar, color in zip(bars, colors):
            bar.set_color(color)
        
        # 设置标签和标题
        plt.title('算法速度对比', fontsize=16)
        plt.xlabel('算法', fontsize=12)
        plt.ylabel('平均匹配时间 (秒)', fontsize=12)
        
        # 添加数值标签
        for i, (bar, time) in enumerate(zip(bars, avg_times)):
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2.,
                height + 0.05 * max(avg_times),
                f'{time:.3f}s',
                ha='center',
                va='bottom',
                fontsize=9
            )
        
        # 旋转x轴标签
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            full_path = os.path.join(self.results_dir, save_path)
            plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"速度对比图已保存到: {full_path}")
        
        if show:
            plt.show()
        
        return plt.gcf()
    
    def analyze_difficulty(self, 
                          query_results: List[Dict[str, Any]],
                          difficulty_metric: str = 'map',
                          num_bins: int = 5,
                          show: bool = True,
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        分析查询难度分布
        
        Args:
            query_results: 单个查询的评估结果列表
            difficulty_metric: 难度度量指标
            num_bins: 分箱数量
            show: 是否显示图像
            save_path: 保存路径
            
        Returns:
            matplotlib Figure对象
        """
        logger.info("开始查询难度分析")
        
        # 提取难度指标
        if difficulty_metric == 'map':
            values = [r.get('map', 0) for r in query_results]
            title = '查询难度分布 (基于MAP)'
            xlabel = 'MAP值'
        elif difficulty_metric == 'mrr':
            values = [r.get('mrr', 0) for r in query_results]
            title = '查询难度分布 (基于MRR)'
            xlabel = 'MRR值'
        elif difficulty_metric.startswith('precision'):
            metric_type, k_str = difficulty_metric.split('@')
            values = [r.get(metric_type, {}).get(int(k_str), 0) for r in query_results]
            title = f'查询难度分布 (基于{metric_type.capitalize()}@{k_str})'
            xlabel = f'{metric_type.capitalize()}@{k_str}值'
        else:
            logger.error(f"不支持的难度指标: {difficulty_metric}")
            return None
        
        # 创建图像
        plt.figure(figsize=self.figsize)
        
        # 绘制直方图
        n, bins, patches = plt.hist(
            values, 
            bins=num_bins,
            alpha=0.7,
            edgecolor='black',
            linewidth=1
        )
        
        # 设置颜色
        colors = plt.cm.plasma(np.linspace(0, 1, len(patches)))
        for patch, color in zip(patches, colors):
            patch.set_facecolor(color)
        
        # 设置标签和标题
        plt.title(title, fontsize=16)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel('查询数量', fontsize=12)
        
        # 添加统计信息
        plt.axvline(
            x=np.mean(values),
            color='red',
            linestyle='--',
            label=f'平均值: {np.mean(values):.3f}'
        )
        plt.legend()
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            full_path = os.path.join(self.results_dir, save_path)
            plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"难度分布图已保存到: {full_path}")
        
        if show:
            plt.show()
        
        return plt.gcf()
    
    def plot_recall_curve(self, 
                         algorithm_recalls: Dict[str, List[Tuple[int, float]]],
                         show: bool = True,
                         save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制Recall曲线
        
        Args:
            algorithm_recalls: {算法名: [(k, recall@k)]}
            show: 是否显示图像
            save_path: 保存路径
            
        Returns:
            matplotlib Figure对象
        """
        logger.info("开始绘制Recall曲线")
        
        # 创建图像
        plt.figure(figsize=self.figsize)
        
        # 绘制每个算法的Recall曲线
        colors = plt.cm.tab10(np.linspace(0, 1, len(algorithm_recalls)))
        
        for i, (algorithm, recalls) in enumerate(algorithm_recalls.items()):
            # 排序
            recalls.sort(key=lambda x: x[0])
            k_values = [r[0] for r in recalls]
            recall_values = [r[1] for r in recalls]
            
            # 绘制曲线
            plt.plot(
                k_values,
                recall_values,
                marker='o',
                linewidth=2,
                markersize=8,
                color=colors[i],
                label=algorithm
            )
        
        # 设置标签和标题
        plt.title('Recall@K 曲线', fontsize=16)
        plt.xlabel('K值', fontsize=12)
        plt.ylabel('Recall', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 调整x轴刻度
        plt.xticks(k_values)
        
        # 设置y轴范围
        plt.ylim(0, 1.1)
        
        # 添加图例
        plt.legend(title='算法', loc='best')
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            full_path = os.path.join(self.results_dir, save_path)
            plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"Recall曲线图已保存到: {full_path}")
        
        if show:
            plt.show()
        
        return plt.gcf()
    
    def compare_mrr(self, 
                   algorithm_mrrs: Dict[str, List[float]],
                   show: bool = True,
                   save_path: Optional[str] = None) -> plt.Figure:
        """
        对比不同算法的MRR分布
        
        Args:
            algorithm_mrrs: {算法名: [每个查询的MRR]}
            show: 是否显示图像
            save_path: 保存路径
            
        Returns:
            matplotlib Figure对象
        """
        logger.info("开始MRR对比分析")
        
        # 准备数据
        data = []
        for algorithm, mrrs in algorithm_mrrs.items():
            for mrr in mrrs:
                data.append({
                    'algorithm': algorithm,
                    'mrr': mrr
                })
        
        df = pd.DataFrame(data)
        
        # 创建图像
        plt.figure(figsize=self.figsize)
        
        # 绘制箱线图
        ax = sns.boxplot(
            x='algorithm',
            y='mrr',
            data=df,
            showmeans=True,
            meanprops={
                'marker': 'o',
                'markerfacecolor': 'white',
                'markeredgecolor': 'black',
                'markersize': 8
            }
        )
        
        # 添加散点图显示实际分布
        sns.stripplot(
            x='algorithm',
            y='mrr',
            data=df,
            jitter=True,
            alpha=0.3,
            color='black',
            size=3
        )
        
        # 设置标签和标题
        plt.title('算法MRR分布对比', fontsize=16)
        plt.xlabel('算法', fontsize=12)
        plt.ylabel('MRR值', fontsize=12)
        plt.ylim(0, 1.1)
        
        # 旋转x轴标签
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            full_path = os.path.join(self.results_dir, save_path)
            plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"MRR对比图已保存到: {full_path}")
        
        if show:
            plt.show()
        
        return plt.gcf()
    
    def plot_time_accuracy_tradeoff(self, 
                                  algorithm_data: Dict[str, Tuple[float, float]],
                                  show: bool = True,
                                  save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制时间-准确率权衡图
        
        Args:
            algorithm_data: {算法名: (平均时间, 平均准确率)}
            show: 是否显示图像
            save_path: 保存路径
            
        Returns:
            matplotlib Figure对象
        """
        logger.info("开始绘制时间-准确率权衡图")
        
        # 准备数据
        algorithms = list(algorithm_data.keys())
        times = [data[0] for data in algorithm_data.values()]
        accuracies = [data[1] for data in algorithm_data.values()]
        
        # 创建图像
        plt.figure(figsize=(10, 8))
        
        # 绘制散点图
        scatter = plt.scatter(
            times,
            accuracies,
            s=200,
            alpha=0.7,
            c=np.arange(len(algorithms)),
            cmap='viridis'
        )
        
        # 添加算法标签
        for i, algorithm in enumerate(algorithms):
            plt.annotate(
                algorithm,
                (times[i], accuracies[i]),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=10,
                fontweight='bold'
            )
        
        # 设置标签和标题
        plt.title('时间-准确率权衡分析', fontsize=16)
        plt.xlabel('平均匹配时间 (秒)', fontsize=12)
        plt.ylabel('平均准确率 (Precision@10)', fontsize=12)
        
        # 添加网格
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 设置x轴为对数刻度
        plt.xscale('log')
        
        # 设置y轴范围
        plt.ylim(0, 1.1)
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            full_path = os.path.join(self.results_dir, save_path)
            plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"时间-准确率权衡图已保存到: {full_path}")
        
        if show:
            plt.show()
        
        return plt.gcf()
    
    def plot_radar_chart(self, 
                        algorithm_metrics: Dict[str, Dict[str, float]],
                        show: bool = True,
                        save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制雷达图对比多个算法的性能
        
        Args:
            algorithm_metrics: {算法名: {指标名: 值}}
            show: 是否显示图像
            save_path: 保存路径
            
        Returns:
            matplotlib Figure对象
        """
        logger.info("开始绘制雷达图")
        
        # 获取所有指标名称
        all_metrics = set()
        for metrics in algorithm_metrics.values():
            all_metrics.update(metrics.keys())
        all_metrics = sorted(list(all_metrics))
        
        # 计算角度
        num_vars = len(all_metrics)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += angles[:1]  # 闭合雷达图
        
        # 创建图像
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        # 设置雷达图属性
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # 设置标签位置
        plt.xticks(angles[:-1], [self._format_metric_name(m) for m in all_metrics], fontsize=12)
        
        # 设置y轴范围
        ax.set_ylim(0, 1)
        
        # 绘制网格线
        ax.yaxis.grid(True, linestyle='--', alpha=0.7)
        ax.xaxis.grid(True, linestyle='--', alpha=0.7)
        
        # 为每个算法绘制雷达图
        colors = plt.cm.tab10(np.linspace(0, 1, len(algorithm_metrics)))
        
        for i, (algorithm, metrics) in enumerate(algorithm_metrics.items()):
            # 获取指标值并闭合
            values = [metrics.get(m, 0) for m in all_metrics]
            values += values[:1]  # 闭合雷达图
            
            # 绘制线条
            ax.plot(
                angles,
                values,
                linewidth=2,
                linestyle='solid',
                color=colors[i],
                label=algorithm
            )
            
            # 填充区域
            ax.fill(
                angles,
                values,
                color=colors[i],
                alpha=0.25
            )
        
        # 添加标题
        plt.title('算法性能雷达图', fontsize=16, pad=20)
        
        # 添加图例
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            full_path = os.path.join(self.results_dir, save_path)
            plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"雷达图已保存到: {full_path}")
        
        if show:
            plt.show()
        
        return fig
    
    def plot_boxplot(self, 
                    algorithm_results: List[Dict[str, Any]],
                    metric: str = 'map',
                    show: bool = True,
                    save_path: Optional[str] = None) -> plt.Figure:
        """
        绘制箱线图分析算法性能分布
        
        Args:
            algorithm_results: 算法评估结果列表
            metric: 要分析的指标
            show: 是否显示图像
            save_path: 保存路径
            
        Returns:
            matplotlib Figure对象
        """
        logger.info(f"开始绘制{metric}箱线图")
        
        # 准备数据
        data = []
        
        for result in algorithm_results:
            algo_name = result.get('algorithm_name', 'Unknown')
            
            # 提取每个查询的指标值
            for individual in result.get('individual_results', []):
                if '@' in metric:
                    metric_type, k_str = metric.split('@')
                    k = int(k_str)
                    value = individual.get(metric_type, {}).get(k, 0)
                else:
                    value = individual.get(metric, 0)
                
                data.append({
                    'algorithm': algo_name,
                    'value': value
                })
        
        if not data:
            logger.warning(f"没有找到足够的数据绘制{metric}箱线图")
            return None
        
        df = pd.DataFrame(data)
        
        # 创建图像
        plt.figure(figsize=self.figsize)
        
        # 绘制箱线图
        ax = sns.boxplot(
            x='algorithm',
            y='value',
            data=df,
            showmeans=True,
            meanprops={
                'marker': 'o',
                'markerfacecolor': 'white',
                'markeredgecolor': 'black',
                'markersize': 8
            }
        )
        
        # 添加散点图
        sns.stripplot(
            x='algorithm',
            y='value',
            data=df,
            jitter=True,
            alpha=0.3,
            color='black',
            size=3
        )
        
        # 设置标签和标题
        plt.title(f'{self._format_metric_name(metric)}分布对比', fontsize=16)
        plt.xlabel('算法', fontsize=12)
        plt.ylabel(f'{self._format_metric_name(metric)}值', fontsize=12)
        
        # 设置y轴范围
        plt.ylim(0, 1.1)
        
        # 旋转x轴标签
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        # 保存图像
        if save_path:
            full_path = os.path.join(self.results_dir, save_path)
            plt.savefig(full_path, dpi=self.dpi, bbox_inches='tight')
            logger.info(f"箱线图已保存到: {full_path}")
        
        if show:
            plt.show()
        
        return plt.gcf()
    
    def create_comprehensive_report(self, 
                                  algorithm_results: List[Dict[str, Any]],
                                  algorithm_times: Optional[Dict[str, Tuple[float, float]]] = None,
                                  output_dir: Optional[str] = None) -> str:
        """
        创建综合分析报告
        
        Args:
            algorithm_results: 算法评估结果列表
            algorithm_times: 算法时间数据
            output_dir: 输出目录
            
        Returns:
            报告目录路径
        """
        logger.info("创建综合分析报告")
        
        # 设置输出目录
        report_dir = output_dir or os.path.join(self.results_dir, 'comprehensive_report')
        os.makedirs(report_dir, exist_ok=True)
        
        # 生成所有可视化
        visualizations = []
        
        # 1. 准确率对比
        self.compare_accuracy(
            algorithm_results,
            show=False,
            save_path=os.path.join(report_dir, 'accuracy_comparison.png')
        )
        visualizations.append('accuracy_comparison.png')
        
        # 2. 速度对比（如果有时间数据）
        if algorithm_times:
            self.compare_speed(
                algorithm_times,
                show=False,
                save_path=os.path.join(report_dir, 'speed_comparison.png')
            )
            visualizations.append('speed_comparison.png')
        
        # 3. Recall曲线
        algorithm_recalls = {}
        for result in algorithm_results:
            algo_name = result.get('algorithm_name', 'Unknown')
            avg_metrics = result.get('average_metrics', {})
            recalls = []
            for k in [1, 3, 5, 10]:
                if f'recall@{k}' in avg_metrics:
                    recalls.append((k, avg_metrics[f'recall@{k}']['mean']))
            if recalls:
                algorithm_recalls[algo_name] = recalls
        
        if algorithm_recalls:
            self.plot_recall_curve(
                algorithm_recalls,
                show=False,
                save_path=os.path.join(report_dir, 'recall_curves.png')
            )
            visualizations.append('recall_curves.png')
        
        # 4. 雷达图
        algorithm_metrics = {}
        for result in algorithm_results:
            algo_name = result.get('algorithm_name', 'Unknown')
            avg_metrics = result.get('average_metrics', {})
            metrics = {}
            for metric in ['precision@1', 'precision@5', 'precision@10', 'recall@10', 'mrr', 'map']:
                if metric in avg_metrics:
                    metrics[metric] = avg_metrics[metric]['mean']
            if metrics:
                algorithm_metrics[algo_name] = metrics
        
        if algorithm_metrics:
            self.plot_radar_chart(
                algorithm_metrics,
                show=False,
                save_path=os.path.join(report_dir, 'radar_chart.png')
            )
            visualizations.append('radar_chart.png')
        
        # 5. MAP箱线图
        self.plot_boxplot(
            algorithm_results,
            metric='map',
            show=False,
            save_path=os.path.join(report_dir, 'map_boxplot.png')
        )
        visualizations.append('map_boxplot.png')
        
        # 生成HTML报告
        html_content = self._generate_html_report(algorithm_results, visualizations, report_dir)
        
        # 保存HTML报告
        html_path = os.path.join(report_dir, 'report.html')
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        logger.info(f"综合分析报告已生成: {report_dir}")
        return report_dir
    
    def _format_metric_name(self, metric: str) -> str:
        """
        格式化指标名称
        """
        if '@' in metric:
            name, k = metric.split('@')
            if name == 'precision':
                return f'Precision@{k}'
            elif name == 'recall':
                return f'Recall@{k}'
            elif name == 'f1':
                return f'F1@{k}'
            elif name == 'ndcg':
                return f'NDCG@{k}'
        
        if metric == 'map':
            return 'MAP'
        elif metric == 'mrr':
            return 'MRR'
        
        return metric
    
    def _generate_html_report(self, 
                             algorithm_results: List[Dict[str, Any]],
                             visualizations: List[str],
                             report_dir: str) -> str:
        """
        生成HTML报告内容
        """
        import datetime
        
        # 生成摘要统计
        summary = []
        for result in algorithm_results:
            algo_name = result.get('algorithm_name', 'Unknown')
            avg_metrics = result.get('average_metrics', {})
            
            summary.append({
                'name': algo_name,
                'precision1': avg_metrics.get('precision@1', {}).get('mean', 0),
                'precision5': avg_metrics.get('precision@5', {}).get('mean', 0),
                'precision10': avg_metrics.get('precision@10', {}).get('mean', 0),
                'map': avg_metrics.get('map', {}).get('mean', 0),
                'mrr': avg_metrics.get('mrr', {}).get('mean', 0)
            })
        
        # 构建HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>多模态检索算法性能分析报告</title>
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
                h1, h2, h3 {{
                    color: #333;
                }}
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
                th {{
                    background-color: #4CAF50;
                    color: white;
                }}
                tr:hover {{
                    background-color: #f5f5f5;
                }}
                .metric-value {{
                    font-weight: bold;
                    font-size: 18px;
                }}
                .visualization {{
                    margin: 30px 0;
                    text-align: center;
                }}
                .visualization img {{
                    max-width: 100%;
                    border: 1px solid #ddd;
                    border-radius: 5px;
                }}
                .timestamp {{
                    color: #666;
                    font-style: italic;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>多模态检索算法性能分析报告</h1>
                <p class="timestamp">生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <h2>1. 摘要统计</h2>
                <table>
                    <tr>
                        <th>算法名称</th>
                        <th>Precision@1</th>
                        <th>Precision@5</th>
                        <th>Precision@10</th>
                        <th>MAP</th>
                        <th>MRR</th>
                    </tr>
        """
        
        for s in summary:
            html += f"""
                    <tr>
                        <td>{s['name']}</td>
                        <td>{s['precision1']:.4f}</td>
                        <td>{s['precision5']:.4f}</td>
                        <td>{s['precision10']:.4f}</td>
                        <td>{s['map']:.4f}</td>
                        <td>{s['mrr']:.4f}</td>
                    </tr>
            """
        
        html += """
                </table>
        """
        
        # 添加可视化
        for viz in visualizations:
            viz_name = viz.replace('_', ' ').title().replace('.Png', '')
            html += f"""
                <h2>2. {viz_name}</h2>
                <div class="visualization">
                    <img src="{viz}" alt="{viz_name}">
                </div>
            """
        
        # 算法详细信息
        html += """
                <h2>3. 算法详细信息</h2>
        """
        
        for result in algorithm_results:
            algo_name = result.get('algorithm_name', 'Unknown')
            dataset_name = result.get('dataset_name', 'Unknown')
            query_count = result.get('query_count', 0)
            avg_metrics = result.get('average_metrics', {})
            
            html += f"""
                <h3>{algo_name}</h3>
                <p><strong>数据集:</strong> {dataset_name}</p>
                <p><strong>查询数量:</strong> {query_count}</p>
                
                <h4>平均指标:</h4>
                <table>
                    <tr>
                        <th>指标</th>
                        <th>平均值</th>
                        <th>标准差</th>
                        <th>中位数</th>
                    </tr>
            """
            
            # 获取所有指标
            metrics_list = sorted(avg_metrics.keys())
            for metric in metrics_list:
                m = avg_metrics[metric]
                html += f"""
                    <tr>
                        <td>{self._format_metric_name(metric)}</td>
                        <td>{m['mean']:.4f}</td>
                        <td>{m['std']:.4f}</td>
                        <td>{m['median']:.4f}</td>
                    </tr>
                """
            
            html += """
                </table>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html