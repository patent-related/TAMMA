#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
可视化演示脚本

直接从评估结果文件生成可视化报告，无需重新运行评估流程
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# 添加项目根目录到Python路径
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.visualization_utils import visualize_metrics

def setup_logging():
    """设置日志配置"""
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y/%m/%d %H:%M:%S'
    )

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='TAMMA多模态检索系统 - 可视化演示工具')
    parser.add_argument('--metrics-path', type=str, 
                       default='./evaluation_results/evaluation_metrics.json',
                       help='评估指标文件路径')
    parser.add_argument('--output-dir', type=str, 
                       default='./visualization_demo',
                       help='可视化输出目录')
    parser.add_argument('--demo', action='store_true', 
                       help='使用示例数据运行演示')
    
    return parser.parse_args()

def main():
    """主函数"""
    # 设置日志
    setup_logging()
    logger = logging.getLogger('generate_visualization')
    
    # 解析参数
    args = parse_args()
    
    logger.info("===== TAMMA多模态检索系统 - 可视化演示 ====")
    logger.info(f"评估指标文件: {args.metrics_path}")
    logger.info(f"输出目录: {args.output_dir}")
    
    # 检查指标文件是否存在
    if not os.path.exists(args.metrics_path):
        if args.demo:
            logger.warning(f"未找到指标文件: {args.metrics_path}")
            logger.warning("创建示例数据用于演示...")
            create_demo_data(args.metrics_path)
        else:
            logger.error(f"错误: 评估指标文件不存在: {args.metrics_path}")
            logger.error("请先运行评估流程或使用 --demo 参数创建示例数据")
            sys.exit(1)
    
    # 生成可视化
    try:
        logger.info("开始生成可视化报告...")
        html_path = visualize_metrics(args.metrics_path, args.output_dir)
        
        logger.info("\n✅ 可视化报告生成成功！")
        logger.info(f"HTML报告: {html_path}")
        logger.info(f"图表文件: {args.output_dir}")
        logger.info("\n📊 生成的可视化内容:")
        logger.info("  - Top-k准确率对比图")
        logger.info("  - MRR和MAP指标对比图")
        logger.info("  - 平均查询时间对比图")
        logger.info("  - 综合HTML报告")
        logger.info("\n💡 提示: 请在浏览器中打开HTML报告查看完整的可视化结果")
        
    except Exception as e:
        logger.error(f"❌ 生成可视化报告时出错: {str(e)}")
        sys.exit(1)

def create_demo_data(metrics_path: str):
    """创建演示数据"""
    import json
    
    # 创建示例指标数据
    demo_metrics = {
        "tamma": {
            "top_k_accuracy": {
                "1": 1.0,
                "5": 1.0,
                "10": 1.0,
                "20": 1.0
            },
            "mrr": 1.0,
            "map": 1.0,
            "avg_query_time": 4.05,
            "total_queries": 50
        },
        "tamma_optimized": {
            "top_k_accuracy": {
                "1": 0.98,
                "5": 0.995,
                "10": 1.0,
                "20": 1.0
            },
            "mrr": 0.985,
            "map": 0.99,
            "avg_query_time": 2.5,
            "total_queries": 50
        },
        "baseline_model": {
            "top_k_accuracy": {
                "1": 0.85,
                "5": 0.92,
                "10": 0.95,
                "20": 0.97
            },
            "mrr": 0.88,
            "map": 0.91,
            "avg_query_time": 3.2,
            "total_queries": 50
        }
    }
    
    # 确保输出目录存在
    os.makedirs(os.path.dirname(metrics_path), exist_ok=True)
    
    # 保存示例数据
    with open(metrics_path, 'w', encoding='utf-8') as f:
        json.dump(demo_metrics, f, indent=2, ensure_ascii=False)
    
    logger.info(f"✅ 已创建示例数据: {metrics_path}")
    logger.info("📊 示例数据包含3个模型的对比: tamma、tamma_optimized、baseline_model")

if __name__ == "__main__":
    main()