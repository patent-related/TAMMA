#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版算法对比可视化脚本（使用英文标签确保显示正常）
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# 设置图表风格
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.2)

# 使用英文标签避免中文显示问题
LABELS = {
    'precision@1': 'Precision@1',
    'precision@5': 'Precision@5',
    'precision@10': 'Precision@10',
    'recall@10': 'Recall@10',
    'map': 'MAP',
    'mrr': 'MRR',
    'time': 'Query Time (s)'
}

def load_experiment_results(results_file):
    """加载实验结果数据"""
    print(f"Loading experiment results: {results_file}")
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results

def prepare_data_for_comparison(results):
    """Prepare data from experiment results for visualization"""
    metrics_data = []
    time_data = []
    
    # 复制原始脚本的数据准备逻辑，确保数据结构一致
    for algo_name, algo_results in results.get('algorithm_results', {}).items():
        # 提取评估指标
        avg_metrics = algo_results.get('average_metrics', {})
        for metric_name, metric_values in avg_metrics.items():
            metrics_data.append({
                'algorithm': algo_name,
                'metric': metric_name,
                'value': metric_values['mean'],
                'std': metric_values['std']
            })
        
        # 提取时间数据
        avg_time = algo_results.get('average_time_per_query', {}).get('mean', 0)
        time_data.append({
            'algorithm': algo_name,
            'time': avg_time
        })
    
    # 转换为DataFrame
    metrics_df = pd.DataFrame(metrics_data)
    time_df = pd.DataFrame(time_data)
    
    return metrics_df, time_df

def plot_precision_comparison(metrics_df, output_dir):
    """Plot Precision@K comparison"""
    # Filter precision metrics
    precision_data = metrics_df[metrics_df['metric'].str.startswith('precision@')].copy()
    precision_data['k_value'] = precision_data['metric'].str.extract('@(\d+)').astype(int)
    
    plt.figure(figsize=(12, 6))
    ax = sns.lineplot(
        x='k_value', 
        y='value', 
        hue='algorithm',
        data=precision_data,
        marker='o',
        linewidth=2
    )
    
    # Add error bars
    for algo in precision_data['algorithm'].unique():
        algo_data = precision_data[precision_data['algorithm'] == algo]
        plt.fill_between(
            algo_data['k_value'],
            algo_data['value'] - algo_data['std'],
            algo_data['value'] + algo_data['std'],
            alpha=0.2
        )
    
    plt.title('Precision@K Comparison', fontsize=16)
    plt.xlabel('K Value', fontsize=14)
    plt.ylabel('Precision', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0, 1.1)
    plt.legend(title='Algorithm', fontsize=12)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'precision_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Precision comparison saved to: {output_path}")
    
    return output_path

def plot_recall_comparison(metrics_df, output_dir):
    """Plot Recall@K comparison"""
    # Filter recall metrics
    recall_data = metrics_df[metrics_df['metric'].str.startswith('recall@')].copy()
    if recall_data.empty:
        print("No Recall metrics found, skipping Recall comparison plot")
        return None
    
    recall_data['k_value'] = recall_data['metric'].str.extract('@(\d+)').astype(int)
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x='k_value', 
        y='value', 
        hue='algorithm',
        data=recall_data
    )
    
    plt.title('Recall@K Comparison', fontsize=16)
    plt.xlabel('K Value', fontsize=14)
    plt.ylabel('Recall', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.ylim(0, 1.1)
    plt.legend(title='Algorithm', fontsize=12)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'recall_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Recall comparison saved to: {output_path}")
    
    return output_path

def plot_map_mrr_comparison(metrics_df, output_dir):
    """Plot MAP and MRR comparison"""
    # Filter MAP and MRR metrics
    map_mrr_data = metrics_df[metrics_df['metric'].isin(['map', 'mrr'])].copy()
    map_mrr_data['metric_label'] = map_mrr_data['metric'].map({'map': 'MAP', 'mrr': 'MRR'})
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x='algorithm', 
        y='value', 
        hue='metric_label',
        data=map_mrr_data
    )
    
    plt.title('MAP and MRR Comparison', fontsize=16)
    plt.xlabel('Algorithm', fontsize=14)
    plt.ylabel('Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.ylim(0, 1.1)
    plt.legend(title='Metric', fontsize=12)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'map_mrr_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"MAP/MRR comparison saved to: {output_path}")
    
    return output_path

def plot_query_time_comparison(time_df, output_dir):
    """Plot query time comparison"""
    # Convert to milliseconds
    time_df['time_ms'] = time_df['time'] * 1000
    
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x='algorithm', 
        y='time_ms',
        data=time_df
    )
    
    plt.title('Average Query Time Comparison (ms)', fontsize=16)
    plt.xlabel('Algorithm', fontsize=14)
    plt.ylabel('Average Query Time (ms)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7, axis='y')
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'query_time_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Query time comparison saved to: {output_path}")
    
    return output_path

def plot_radar_chart(metrics_df, output_dir):
    """Plot radar chart for algorithm performance comparison"""
    # Select main metrics for radar chart
    main_metrics = ['precision@1', 'precision@5', 'map', 'mrr']
    radar_data = metrics_df[metrics_df['metric'].isin(main_metrics)].copy()
    
    # Pivot table to prepare data for radar chart
    pivot_data = radar_data.pivot_table(
        index='algorithm',
        columns='metric',
        values='value'  
    ).reset_index()
    
    # Rename columns for better readability
    pivot_data.columns = ['algorithm', 'MAP', 'MRR', 'Precision@1', 'Precision@5']
    
    # Radar chart setup
    categories = ['MAP', 'MRR', 'Precision@1', 'Precision@5']
    N = len(categories)
    
    # Angle settings
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]  # Close radar chart
    
    # Create figure
    plt.figure(figsize=(10, 10))
    ax = plt.subplot(111, polar=True)
    
    # Set radar chart angles and labels
    plt.xticks(angles[:-1], categories, fontsize=12)
    ax.set_ylim(0, 1.1)
    
    # Plot data for each algorithm
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, algo in enumerate(pivot_data['algorithm']):
        values = pivot_data.iloc[i, 1:].values.tolist()
        values += values[:1]  # Close radar chart
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', color=colors[i], label=algo)
        ax.fill(angles, values, color=colors[i], alpha=0.1)
    
    plt.title('Algorithm Performance Radar Chart', fontsize=16, pad=20)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    plt.tight_layout()
    
    # Save figure
    output_path = os.path.join(output_dir, 'radar_chart_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Radar chart comparison saved to: {output_path}")
    
    return output_path

def generate_summary_table(metrics_df, time_df, output_dir):
    """Generate HTML table for algorithm comparison"""
    # Create output directory if it doesn't exist
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Prepare HTML content
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>Algorithm Comparison Table</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333333; }
            table { border-collapse: collapse; width: 100%; margin-top: 20px; }
            th, td { text-align: left; padding: 12px; border: 1px solid #ddd; }
            th { background-color: #4CAF50; color: white; }
            tr:nth-child(even) { background-color: #f2f2f2; }
            .metric-highlight { font-weight: bold; color: #0066cc; }
        </style>
    </head>
    <body>
        <h1>Algorithm Performance Comparison</h1>
        <table>
            <tr>
                <th>Algorithm</th>
                <th>Precision@1</th>
                <th>Precision@5</th>
                <th>Precision@10</th>
                <th>Recall@10</th>
                <th>MAP</th>
                <th>MRR</th>
                <th>Query Time (ms)</th>
            </tr>
    """
    
    # Get all algorithms
    algorithms = list(set(metrics_df['algorithm'].tolist() + time_df['algorithm'].tolist()))
    
    for algo in algorithms:
        # Get metrics for this algorithm
        algo_metrics = metrics_df[metrics_df['algorithm'] == algo]
        
        # Get time for this algorithm
        algo_time = time_df[time_df['algorithm'] == algo]
        query_time = algo_time['time'].iloc[0] * 1000 if not algo_time.empty else 'N/A'
        
        # Extract individual metrics
        precision1 = algo_metrics[algo_metrics['metric'] == 'precision@1']['value'].iloc[0] if 'precision@1' in algo_metrics['metric'].values else 'N/A'
        precision5 = algo_metrics[algo_metrics['metric'] == 'precision@5']['value'].iloc[0] if 'precision@5' in algo_metrics['metric'].values else 'N/A'
        precision10 = algo_metrics[algo_metrics['metric'] == 'precision@10']['value'].iloc[0] if 'precision@10' in algo_metrics['metric'].values else 'N/A'
        recall10 = algo_metrics[algo_metrics['metric'] == 'recall@10']['value'].iloc[0] if 'recall@10' in algo_metrics['metric'].values else 'N/A'
        map_score = algo_metrics[algo_metrics['metric'] == 'map']['value'].iloc[0] if 'map' in algo_metrics['metric'].values else 'N/A'
        mrr_score = algo_metrics[algo_metrics['metric'] == 'mrr']['value'].iloc[0] if 'mrr' in algo_metrics['metric'].values else 'N/A'
        
        # Format numbers
        def format_num(num):
            if isinstance(num, float):
                return f"{num:.4f}"
            return str(num)
        
        html_content += f"""
            <tr>
                <td>{algo}</td>
                <td>{format_num(precision1)}</td>
                <td>{format_num(precision5)}</td>
                <td>{format_num(precision10)}</td>
                <td>{format_num(recall10)}</td>
                <td>{format_num(map_score)}</td>
                <td>{format_num(mrr_score)}</td>
                <td>{format_num(query_time)}</td>
            </tr>
        """
    
    # Close HTML
    html_content += """
        </table>
    </body>
    </html>
    """
    
    # Save HTML file
    output_path = os.path.join(output_dir, 'algorithm_comparison_table.html')
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Algorithm comparison table saved to: {output_path}")
    return output_path

def main():
    """Main function to generate all visualizations"""
    # 使用与原始脚本相同的绝对路径
    results_file = '/home/idata/mtl/code/new-QA/results/comparison/experiments/algorithm_comparison_experiment/final_results.json'
    output_dir = '/home/idata/mtl/code/new-QA/results/comparison/visualizations'
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading experiment results...")
    results = load_experiment_results(results_file)
    
    print("Preparing data for comparison...")
    metrics_df, time_df = prepare_data_for_comparison(results)
    
    print("Generating visualizations...")
    # 生成各类对比图表
    chart_paths = []
    
    chart_paths.append(plot_precision_comparison(metrics_df, output_dir))
    
    recall_chart = plot_recall_comparison(metrics_df, output_dir)
    if recall_chart:
        chart_paths.append(recall_chart)
    
    chart_paths.append(plot_map_mrr_comparison(metrics_df, output_dir))
    chart_paths.append(plot_query_time_comparison(time_df, output_dir))
    chart_paths.append(plot_radar_chart(metrics_df, output_dir))
    
    # 生成对比表格
    table_path = generate_summary_table(metrics_df, time_df, output_dir)
    
    print("\nAll visualizations completed!")
    print("\nGenerated files:")
    for file in os.listdir(output_dir):
        print(f"- {os.path.join(output_dir, file)}")
    print(f"\nYou can view all visualization results in: {output_dir}")

if __name__ == "__main__":
    main()