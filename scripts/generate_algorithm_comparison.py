#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
算法对比可视化生成脚本
用于生成不同算法的性能对比图表
"""

import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.font_manager import FontProperties

# 简单直接的中文字体配置
import matplotlib

# 直接设置matplotlib全局字体参数
matplotlib.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'WenQuanYi Micro Hei Mono', 'sans-serif']
matplotlib.rcParams['axes.unicode_minus'] = False
print("已配置matplotlib使用文泉驿字体")

# 验证字体是否可用
try:
    # 尝试创建一个简单的图表来验证字体
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, '测试中文显示')
    plt.close()
    print("字体测试成功")
except Exception as e:
    print(f"字体测试出错: {e}")

# 设置图表风格
sns.set(style="whitegrid", font_scale=1.2)
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

def load_experiment_results(results_file):
    """加载实验结果数据"""
    print(f"正在加载实验结果: {results_file}")
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results

def prepare_metrics_data(results):
    """准备指标数据用于可视化"""
    metrics_data = []
    time_data = []
    
    for algo_name, algo_results in results['algorithm_results'].items():
        # 提取评估指标
        avg_metrics = algo_results['average_metrics']
        for metric_name, metric_values in avg_metrics.items():
            metrics_data.append({
                'algorithm': algo_name,
                'metric': metric_name,
                'value': metric_values['mean'],
                'std': metric_values['std']
            })
        
        # 提取时间数据
        avg_time = algo_results['average_time_per_query']['mean']
        time_data.append({
            'algorithm': algo_name,
            'time': avg_time
        })
    
    return pd.DataFrame(metrics_data), pd.DataFrame(time_data)

def plot_precision_comparison(metrics_df, output_dir):
    """绘制Precision@K对比图"""
    # 筛选precision指标
    precision_data = metrics_df[metrics_df['metric'].str.startswith('precision@')].copy()
    precision_data['k_value'] = precision_data['metric'].str.extract('@(\d+)').astype(int)
    
    # 直接设置matplotlib参数，先设置字体后创建图表
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Heiti TC', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(12, 6))
    
    ax = sns.lineplot(
        x='k_value', 
        y='value', 
        hue='algorithm',
        data=precision_data,
        marker='o',
        linewidth=2
    )
    
    # 添加误差线
    for algo in precision_data['algorithm'].unique():
        algo_data = precision_data[precision_data['algorithm'] == algo]
        plt.fill_between(
            algo_data['k_value'],
            algo_data['value'] - algo_data['std'],
            algo_data['value'] + algo_data['std'],
            alpha=0.2
        )
    
    # 设置标题和标签，使用全局字体设置
    plt.title('不同算法的Precision@K对比', fontsize=16)
    plt.xlabel('K值', fontsize=14)
    plt.ylabel('Precision值', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.1)
    plt.legend(title='算法', fontsize=12)
    
    # 添加数值标签
    for line in ax.lines:
        for x, y in zip(line.get_xdata(), line.get_ydata()):
            plt.text(x, y + 0.03, f'{y:.2f}', ha='center', fontsize=9)
    
    output_path = os.path.join(output_dir, 'precision_comparison.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Precision对比图已保存至: {output_path}")
    return output_path

def plot_recall_comparison(metrics_df, output_dir):
    """绘制Recall@K对比图"""
    # 筛选recall指标
    recall_data = metrics_df[metrics_df['metric'].str.startswith('recall@')].copy()
    if recall_data.empty:
        print("未找到Recall指标数据，跳过Recall对比图绘制")
        return None
    
    recall_data['k_value'] = recall_data['metric'].str.extract('@(\d+)').astype(int)
    
    # 直接设置matplotlib参数，先设置字体后创建图表
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Heiti TC', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(12, 6))
    
    ax = sns.barplot(
        x='k_value', 
        y='value', 
        hue='algorithm',
        data=recall_data
    )
    
    # 设置标题和标签，使用全局字体设置
    plt.title('不同算法的Recall@K对比', fontsize=16)
    plt.xlabel('K值', fontsize=14)
    plt.ylabel('Recall值', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.ylim(0, 1.1)
    plt.legend(title='算法', fontsize=12)
    
    # 添加数值标签
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.03,
                f'{height:.2f}', ha='center', fontsize=9)
    
    output_path = os.path.join(output_dir, 'recall_comparison.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Recall对比图已保存至: {output_path}")
    return output_path

def plot_map_mrr_comparison(metrics_df, output_dir):
    """绘制MAP和MRR对比图"""
    # 筛选MAP和MRR指标
    map_mrr_data = metrics_df[metrics_df['metric'].isin(['map', 'mrr'])].copy()
    map_mrr_data['metric_label'] = map_mrr_data['metric'].map({'map': 'MAP', 'mrr': 'MRR'})
    
    # 直接设置matplotlib参数，先设置字体后创建图表
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Heiti TC', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(12, 6))
    
    ax = sns.barplot(
        x='algorithm', 
        y='value', 
        hue='metric_label',
        data=map_mrr_data
    )
    
    # 设置标题和标签，使用全局字体设置
    plt.title('不同算法的MAP和MRR对比', fontsize=16)
    plt.xlabel('算法', fontsize=14)
    plt.ylabel('指标值', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.ylim(0, 1.1)
    plt.legend(title='评估指标', fontsize=12)
    
    # 添加数值标签
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.03,
                f'{height:.3f}', ha='center', fontsize=9)
    
    output_path = os.path.join(output_dir, 'map_mrr_comparison.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"MAP和MRR对比图已保存至: {output_path}")
    return output_path

def plot_query_time_comparison(time_df, output_dir):
    """绘制查询时间对比图"""
    # 转换为毫秒
    time_df['time_ms'] = time_df['time'] * 1000
    
    # 直接设置matplotlib参数，先设置字体后创建图表
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Heiti TC', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.figure(figsize=(12, 6))
    
    ax = sns.barplot(
        x='algorithm', 
        y='time_ms',
        data=time_df
    )
    
    # 设置标题和标签，使用全局字体设置
    plt.title('不同算法的平均查询时间对比（毫秒）', fontsize=16)
    plt.xlabel('算法', fontsize=14)
    plt.ylabel('平均查询时间（毫秒）', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    
    # 添加数值标签
    for p in ax.patches:
        height = p.get_height()
        ax.text(p.get_x() + p.get_width()/2., height + 0.001,
                f'{height:.4f}ms', ha='center', fontsize=9)
    
    output_path = os.path.join(output_dir, 'query_time_comparison.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"查询时间对比图已保存至: {output_path}")
    return output_path

def plot_radar_chart(metrics_df, output_dir):
    """绘制雷达图对比算法综合性能"""
    # 选择主要指标进行雷达图对比
    main_metrics = ['precision@1', 'precision@5', 'map', 'mrr']
    radar_data = metrics_df[metrics_df['metric'].isin(main_metrics)].copy()
    
    # 透视表准备雷达图数据
    pivot_data = radar_data.pivot_table(
        index='algorithm',
        columns='metric',
        values='value'  
    ).reset_index()
    
    # 重命名列以提高可读性
    pivot_data.columns = ['algorithm', 'MAP', 'MRR', 'Precision@1', 'Precision@5']
    
    # 雷达图设置
    categories = ['MAP', 'MRR', 'Precision@1', 'Precision@5']
    N = len(categories)
    
    # 角度设置
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # 闭合雷达图
    
    # 直接设置matplotlib参数，先设置字体后创建图表
    plt.rcParams['font.sans-serif'] = ['WenQuanYi Micro Hei', 'Heiti TC', 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表
    plt.figure(figsize=(10, 10))
    
    ax = plt.subplot(111, polar=True)
    
    # 设置雷达图角度和标签，使用全局字体设置
    plt.xticks(angles[:-1], categories, fontsize=12)
    ax.set_ylim(0, 1.1)
    
    # 绘制每个算法的数据
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, algo in enumerate(pivot_data['algorithm']):
        values = pivot_data.iloc[i, 1:].values.tolist()
        values += values[:1]  # 闭合雷达图
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', color=colors[i], label=algo)
        ax.fill(angles, values, color=colors[i], alpha=0.1)
    
    # 设置标题，使用全局字体设置
    plt.title('算法综合性能雷达图对比', fontsize=16, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    
    output_path = os.path.join(output_dir, 'radar_chart_comparison.png')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"雷达图对比已保存至: {output_path}")
    return output_path

def generate_summary_table(results, output_dir):
    """生成算法性能对比表格并保存为HTML"""
    # 准备数据
    total_rounds = results['experiment_summary']['total_rounds']
    algorithms = ', '.join(results['experiment_summary']['algorithms'])
    num_algorithms = len(results['experiment_summary']['algorithms'])
    
    # 构建HTML内容，使用f-string避免嵌套格式化问题
    html_content = f'''
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>算法性能对比表格</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            margin: 20px;
            line-height: 1.6;
        }}
        h1 {{
            color: #333;
            text-align: center;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            padding: 12px;
            text-align: center;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        tr:hover {{
            background-color: #ddd;
        }}
        .best {{
            background-color: #90EE90;
            font-weight: bold;
        }}
        .summary {{
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <h1>算法性能对比分析</h1>
    <div class="summary">
        <p><strong>实验总结：</strong>本次对比实验包含 {total_rounds} 轮运行，比较了 {num_algorithms} 种算法：{algorithms}</p>
    </div>
    <table>
        <thead>
            <tr>
                <th>算法</th>
                <th>Precision@1</th>
                <th>Precision@5</th>
                <th>Precision@10</th>
                <th>Recall@10</th>
                <th>MAP</th>
                <th>MRR</th>
                <th>平均查询时间</th>
            </tr>
        </thead>
        <tbody>
'''
    
    # 为每个算法添加行数据
    for algo_name, algo_results in results['algorithm_results'].items():
        avg_metrics = algo_results['average_metrics']
        avg_time = algo_results['average_time_per_query']['mean'] * 1000  # 转换为毫秒
        
        html_content += f'''
            <tr>
                <td>{algo_name}</td>
                <td>{avg_metrics['precision@1']['mean']:.3f}</td>
                <td>{avg_metrics['precision@5']['mean']:.3f}</td>
                <td>{avg_metrics['precision@10']['mean']:.3f}</td>
                <td>{avg_metrics['recall@10']['mean']:.3f}</td>
                <td>{avg_metrics['map']['mean']:.3f}</td>
                <td>{avg_metrics['mrr']['mean']:.3f}</td>
                <td>{avg_time:.4f}ms</td>
            </tr>
'''
    
    html_content += '''
        </tbody>
    </table>
</body>
</html>
'''
    
    output_path = os.path.join(output_dir, 'algorithm_comparison_table.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"算法对比表格已保存至: {output_path}")
    return output_path

def main():
    # 配置路径
    results_file = '/home/idata/mtl/code/new-QA/results/comparison/experiments/algorithm_comparison_experiment/final_results.json'
    output_dir = '/home/idata/mtl/code/new-QA/results/comparison/visualizations'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    results = load_experiment_results(results_file)
    
    # 准备数据
    metrics_df, time_df = prepare_metrics_data(results)
    
    # 生成各类对比图表
    print("\n开始生成算法对比图表...")
    chart_paths = []
    
    chart_paths.append(plot_precision_comparison(metrics_df, output_dir))
    
    recall_chart = plot_recall_comparison(metrics_df, output_dir)
    if recall_chart:
        chart_paths.append(recall_chart)
    
    chart_paths.append(plot_map_mrr_comparison(metrics_df, output_dir))
    chart_paths.append(plot_query_time_comparison(time_df, output_dir))
    chart_paths.append(plot_radar_chart(metrics_df, output_dir))
    
    # 生成对比表格
    table_path = generate_summary_table(results, output_dir)
    
    print("\n所有可视化内容生成完成！")
    print(f"\n生成的文件列表：")
    for path in chart_paths:
        print(f"- {path}")
    print(f"- {table_path}")
    
    print("\n您可以在以下目录查看所有可视化结果：")
    print(f"{output_dir}")

if __name__ == "__main__":
    main()