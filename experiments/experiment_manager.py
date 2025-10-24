import os
import json
import time
import logging
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional, Any, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ExperimentManager:
    """
    实验管理器
    
    支持SIFT codebook准备、对比实验运行（数据集准备、算法初始化、评估、对比分析、统计检验、结果保存等完整流程）
    """
    
    def __init__(self, 
                 results_dir: str = './results',
                 dataset_dir: str = './data',
                 algorithms_dir: str = './algorithms',
                 save_results: bool = True,
                 random_seed: int = 42):
        """
        Args:
            results_dir: 结果保存目录
            dataset_dir: 数据集目录
            algorithms_dir: 算法目录
            save_results: 是否保存结果
            random_seed: 随机种子
        """
        self.results_dir = results_dir
        self.dataset_dir = dataset_dir
        self.algorithms_dir = algorithms_dir
        self.save_results = save_results
        
        # 创建目录结构
        self.experiment_dir = os.path.join(results_dir, 'experiments')
        self.models_dir = os.path.join(results_dir, 'models')
        self.figures_dir = os.path.join(results_dir, 'figures')
        self.reports_dir = os.path.join(results_dir, 'reports')
        self.datasets_dir = os.path.join(results_dir, 'datasets')
        
        for dir_path in [self.experiment_dir, self.models_dir, self.figures_dir, 
                        self.reports_dir, self.datasets_dir]:
            os.makedirs(dir_path, exist_ok=True)
        
        # 设置随机种子
        np.random.seed(random_seed)
        
        logger.info(f"初始化实验管理器: results_dir={results_dir}")
        logger.info(f"实验目录: {self.experiment_dir}")
    
    def prepare_sift_codebook(self, 
                            dataset_path: str,
                            codebook_size: int = 1024,
                            num_samples: int = 1000,
                            output_path: Optional[str] = None) -> str:
        """
        准备SIFT codebook
        
        Args:
            dataset_path: 数据集路径
            codebook_size: 码本大小
            num_samples: 用于训练的样本数量
            output_path: 输出路径
            
        Returns:
            codebook文件路径
        """
        logger.info(f"开始准备SIFT codebook: size={codebook_size}, samples={num_samples}")
        
        try:
            import cv2
            from sklearn.cluster import KMeans
            import pickle
            
            # 初始化SIFT
            sift = cv2.SIFT_create()
            
            # 收集特征点
            all_features = []
            image_count = 0
            
            # 遍历数据集
            image_paths = []
            for root, _, files in os.walk(dataset_path):
                for file in files:
                    if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_paths.append(os.path.join(root, file))
                        if len(image_paths) >= num_samples:
                            break
                if len(image_paths) >= num_samples:
                    break
            
            logger.info(f"找到 {len(image_paths)} 张图像用于训练SIFT codebook")
            
            # 提取特征
            for img_path in image_paths[:num_samples]:
                try:
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    keypoints, descriptors = sift.detectAndCompute(gray, None)
                    
                    if descriptors is not None and len(descriptors) > 0:
                        all_features.extend(descriptors)
                        image_count += 1
                        
                        if image_count % 100 == 0:
                            logger.info(f"已处理 {image_count}/{num_samples} 张图像")
                            
                except Exception as e:
                    logger.error(f"处理图像 {img_path} 失败: {e}")
                    continue
            
            logger.info(f"共收集 {len(all_features)} 个特征点")
            
            # 如果特征点不足，返回错误
            if len(all_features) < codebook_size:
                logger.error(f"特征点数量不足，需要至少 {codebook_size} 个特征点")
                return None
            
            # 采样特征点以加速聚类
            if len(all_features) > 100000:
                logger.info("特征点数量过多，进行采样")
                sample_indices = np.random.choice(len(all_features), 100000, replace=False)
                sampled_features = [all_features[i] for i in sample_indices]
            else:
                sampled_features = all_features
            
            # 训练KMeans模型
            logger.info(f"开始训练KMeans模型: 特征点数量={len(sampled_features)}")
            start_time = time.time()
            
            kmeans = KMeans(
                n_clusters=codebook_size,
                random_state=42,
                n_init=10,
                max_iter=300
            )
            kmeans.fit(sampled_features)
            
            end_time = time.time()
            logger.info(f"KMeans训练完成，耗时: {end_time - start_time:.2f} 秒")
            
            # 保存codebook
            if output_path is None:
                output_path = os.path.join(self.models_dir, f'sift_codebook_{codebook_size}.pkl')
            
            with open(output_path, 'wb') as f:
                pickle.dump({
                    'codebook': kmeans.cluster_centers_,
                    'size': codebook_size,
                    'trained_on_samples': image_count,
                    'training_time': end_time - start_time
                }, f)
            
            logger.info(f"SIFT codebook已保存到: {output_path}")
            return output_path
            
        except ImportError as e:
            logger.error(f"导入错误: {e}")
            return None
        except Exception as e:
            logger.error(f"准备SIFT codebook失败: {e}")
            return None
    
    def run_comparison_experiment(self,
                                 algorithms: List[Any],
                                 dataset: Dict[str, Any],
                                 num_rounds: int = 5,
                                 query_size: Optional[int] = None,
                                 experiment_name: Optional[str] = None,
                                 metrics: List[str] = None) -> Dict[str, Any]:
        """
        运行对比实验
        
        Args:
            algorithms: 算法列表
            dataset: 数据集信息
            num_rounds: 运行轮数
            query_size: 查询集大小
            experiment_name: 实验名称
            metrics: 评估指标列表
            
        Returns:
            实验结果
        """
        logger.info("开始运行对比实验")
        
        # 生成实验ID
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if experiment_name is None:
            experiment_name = f'exp_{timestamp}'
        
        experiment_dir = os.path.join(self.experiment_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        
        # 默认指标
        if metrics is None:
            metrics = ['precision@1', 'precision@5', 'precision@10', 'recall@10', 'map', 'mrr']
        
        # 记录实验配置
        config = {
            'experiment_name': experiment_name,
            'timestamp': timestamp,
            'algorithms': [algo.__class__.__name__ for algo in algorithms],
            'dataset': dataset.get('name', 'unknown'),
            'num_rounds': num_rounds,
            'query_size': query_size,
            'metrics': metrics
        }
        
        config_path = os.path.join(experiment_dir, 'config.json')
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, ensure_ascii=False, indent=2)
        
        logger.info(f"实验配置已保存到: {config_path}")
        logger.info(f"实验配置: {config}")
        
        # 运行多轮实验
        all_results = []
        algorithm_times = {}
        
        for round_idx in range(num_rounds):
            logger.info(f"\n--- 轮次 {round_idx + 1}/{num_rounds} ---")
            
            # 准备轮次数据
            round_data = self._prepare_round_data(dataset, query_size)
            if not round_data:
                logger.error(f"准备轮次 {round_idx + 1} 数据失败")
                continue
            
            queries, ground_truths = round_data
            
            # 运行每个算法
            for algorithm in algorithms:
                algo_name = algorithm.__class__.__name__
                logger.info(f"运行算法: {algo_name}")
                
                try:
                    # 运行并计时
                    start_time = time.time()
                    results = self._run_algorithm(algorithm, queries, ground_truths)
                    end_time = time.time()
                    
                    run_time = end_time - start_time
                    avg_time_per_query = run_time / len(queries) if queries else 0
                    
                    logger.info(f"算法 {algo_name} 运行完成，总耗时: {run_time:.2f} 秒，平均每个查询: {avg_time_per_query:.4f} 秒")
                    
                    # 记录时间
                    if algo_name not in algorithm_times:
                        algorithm_times[algo_name] = []
                    algorithm_times[algo_name].append(avg_time_per_query)
                    
                    # 评估结果
                    evaluation = self._evaluate_results(results, ground_truths, metrics)
                    
                    # 记录轮次结果
                    round_result = {
                        'round': round_idx + 1,
                        'algorithm': algo_name,
                        'run_time': run_time,
                        'avg_time_per_query': avg_time_per_query,
                        'evaluation': evaluation,
                        'individual_results': results
                    }
                    
                    all_results.append(round_result)
                    
                    # 保存轮次结果
                    round_result_path = os.path.join(experiment_dir, f'round_{round_idx + 1}_{algo_name}.json')
                    with open(round_result_path, 'w', encoding='utf-8') as f:
                        json.dump(round_result, f, ensure_ascii=False, indent=2)
                    
                except Exception as e:
                    logger.error(f"运行算法 {algo_name} 失败: {e}")
                    continue
        
        # 汇总结果
        final_results = self._aggregate_results(all_results, algorithm_times)
        
        # 保存最终结果
        final_result_path = os.path.join(experiment_dir, 'final_results.json')
        with open(final_result_path, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        logger.info(f"实验结果已保存到: {final_result_path}")
        
        # 生成分析报告
        report_path = self._generate_experiment_report(final_results, experiment_dir)
        logger.info(f"实验报告已生成: {report_path}")
        
        return final_results
    
    def _prepare_round_data(self, 
                          dataset: Dict[str, Any],
                          query_size: Optional[int]) -> Tuple[List[Any], List[Any]]:
        """
        准备轮次数据
        
        Args:
            dataset: 数据集信息
            query_size: 查询集大小
            
        Returns:
            (查询列表, 真实标签列表)
        """
        try:
            # 这里应该根据实际数据集格式实现
            # 目前返回示例数据
            dataset_path = dataset.get('root_dir', '')
            
            if not dataset_path or not os.path.exists(dataset_path):
                logger.error(f"数据集路径不存在: {dataset_path}")
                # 返回模拟数据
                queries = [f'query_{i}' for i in range(10)]
                ground_truths = [[f'target_{i}' for i in range(5)] for _ in range(10)]
                return queries, ground_truths
            
            # 从数据集加载
            # 这里应该根据实际数据格式实现
            queries = []
            ground_truths = []
            
            # 示例实现
            annotations_file = os.path.join(dataset_path, 'annotations.json')
            if os.path.exists(annotations_file):
                with open(annotations_file, 'r', encoding='utf-8') as f:
                    annotations = json.load(f)
                
                for ann in annotations:
                    queries.append(ann)
                    # 这里应该根据实际数据格式构建ground truth
                    ground_truths.append([ann])
            
            # 如果设置了查询大小，进行采样
            if query_size and len(queries) > query_size:
                indices = np.random.choice(len(queries), query_size, replace=False)
                queries = [queries[i] for i in indices]
                ground_truths = [ground_truths[i] for i in indices]
            
            return queries, ground_truths
            
        except Exception as e:
            logger.error(f"准备轮次数据失败: {e}")
            # 返回模拟数据
            queries = [f'query_{i}' for i in range(10)]
            ground_truths = [[f'target_{i}' for i in range(5)] for _ in range(10)]
            return queries, ground_truths
    
    def _run_algorithm(self, 
                      algorithm: Any,
                      queries: List[Any],
                      ground_truths: List[Any]) -> List[Dict[str, Any]]:
        """
        运行算法
        
        Args:
            algorithm: 算法实例
            queries: 查询列表
            ground_truths: 真实标签列表
            
        Returns:
            算法结果列表
        """
        results = []
        
        for i, query in enumerate(queries):
            try:
                # 根据算法接口调用
                if hasattr(algorithm, 'match'):
                    # 单个匹配
                    matches = algorithm.match(query)
                elif hasattr(algorithm, 'batch_match'):
                    # 批量匹配（如果是最后一个或每10个查询）
                    if i == len(queries) - 1 or (i + 1) % 10 == 0:
                        batch_results = algorithm.batch_match(queries[i-9:i+1] if i > 9 else queries[:i+1])
                        for j, q in enumerate(queries[i-9:i+1] if i > 9 else queries[:i+1]):
                            if j == len(batch_results) - 1 or (i+1) % 10 == 0 and j == (i % 10):
                                matches = batch_results[j]
                                break
                    else:
                        continue
                else:
                    raise ValueError(f"算法 {algorithm.__class__.__name__} 没有match或batch_match方法")
                
                # 构建结果
                result = {
                    'query_id': i,
                    'matches': matches,
                    'ground_truth': ground_truths[i]
                }
                results.append(result)
                
                # 记录进度
                if (i + 1) % 100 == 0:
                    logger.info(f"已处理 {i + 1}/{len(queries)} 个查询")
                    
            except Exception as e:
                logger.error(f"处理查询 {i} 失败: {e}")
                # 添加空结果
                results.append({
                    'query_id': i,
                    'matches': [],
                    'ground_truth': ground_truths[i],
                    'error': str(e)
                })
        
        return results
    
    def _evaluate_results(self, 
                         results: List[Dict[str, Any]],
                         ground_truths: List[Any],
                         metrics: List[str]) -> Dict[str, Any]:
        """
        评估算法结果
        
        Args:
            results: 算法结果
            ground_truths: 真实标签
            metrics: 评估指标
            
        Returns:
            评估结果
        """
        # 导入评估器
        try:
            from evaluation.advanced_evaluator import AdvancedEvaluator
            
            evaluator = AdvancedEvaluator()
            evaluation = evaluator.evaluate(results, metrics)
            return evaluation
            
        except ImportError:
            logger.warning("AdvancedEvaluator导入失败，使用简单评估")
            # 使用简单评估
            evaluation = self._simple_evaluate(results, metrics)
            return evaluation
        except Exception as e:
            logger.error(f"评估失败: {e}")
            return {}
    
    def _simple_evaluate(self, 
                        results: List[Dict[str, Any]],
                        metrics: List[str]) -> Dict[str, Any]:
        """
        简单评估实现
        """
        evaluation = {}
        
        # 计算Precision@K
        for metric in metrics:
            if metric.startswith('precision@'):
                k = int(metric.split('@')[1])
                precisions = []
                
                for result in results:
                    matches = result.get('matches', [])[:k]
                    ground_truth = result.get('ground_truth', [])
                    
                    if not ground_truth:
                        continue
                    
                    # 这里需要根据实际数据格式实现匹配判断
                    # 简化版本：假设匹配项和真实项有共同的ID
                    true_positives = 0
                    for match in matches:
                        # 假设match是字典，有id字段
                        if isinstance(match, dict) and 'id' in match:
                            match_id = match['id']
                            for gt in ground_truth:
                                if isinstance(gt, dict) and 'id' in gt and gt['id'] == match_id:
                                    true_positives += 1
                                    break
                    
                    precision = true_positives / k if k > 0 else 0
                    precisions.append(precision)
                
                if precisions:
                    evaluation[metric] = {
                        'mean': np.mean(precisions),
                        'std': np.std(precisions),
                        'median': np.median(precisions)
                    }
            
            elif metric.startswith('recall@'):
                k = int(metric.split('@')[1])
                recalls = []
                
                for result in results:
                    matches = result.get('matches', [])[:k]
                    ground_truth = result.get('ground_truth', [])
                    
                    if not ground_truth:
                        continue
                    
                    true_positives = 0
                    for match in matches:
                        if isinstance(match, dict) and 'id' in match:
                            match_id = match['id']
                            for gt in ground_truth:
                                if isinstance(gt, dict) and 'id' in gt and gt['id'] == match_id:
                                    true_positives += 1
                                    break
                    
                    recall = true_positives / len(ground_truth) if ground_truth else 0
                    recalls.append(recall)
                
                if recalls:
                    evaluation[metric] = {
                        'mean': np.mean(recalls),
                        'std': np.std(recalls),
                        'median': np.median(recalls)
                    }
            
            elif metric == 'map':
                aps = []
                
                for result in results:
                    matches = result.get('matches', [])
                    ground_truth = result.get('ground_truth', [])
                    
                    if not ground_truth:
                        continue
                    
                    true_positives = 0
                    precision_sum = 0
                    relevant_count = 0
                    
                    for i, match in enumerate(matches):
                        is_relevant = False
                        if isinstance(match, dict) and 'id' in match:
                            match_id = match['id']
                            for gt in ground_truth:
                                if isinstance(gt, dict) and 'id' in gt and gt['id'] == match_id:
                                    is_relevant = True
                                    relevant_count += 1
                                    break
                        
                        if is_relevant:
                            true_positives += 1
                            precision = true_positives / (i + 1)
                            precision_sum += precision
                    
                    ap = precision_sum / len(ground_truth) if ground_truth else 0
                    aps.append(ap)
                
                if aps:
                    evaluation[metric] = {
                        'mean': np.mean(aps),
                        'std': np.std(aps),
                        'median': np.median(aps)
                    }
            
            elif metric == 'mrr':
                reciprocal_ranks = []
                
                for result in results:
                    matches = result.get('matches', [])
                    ground_truth = result.get('ground_truth', [])
                    
                    if not ground_truth:
                        continue
                    
                    first_relevant = -1
                    for i, match in enumerate(matches):
                        if isinstance(match, dict) and 'id' in match:
                            match_id = match['id']
                            for gt in ground_truth:
                                if isinstance(gt, dict) and 'id' in gt and gt['id'] == match_id:
                                    first_relevant = i + 1  # 排名从1开始
                                    break
                        if first_relevant > 0:
                            break
                    
                    if first_relevant > 0:
                        reciprocal_rank = 1 / first_relevant
                    else:
                        reciprocal_rank = 0
                    
                    reciprocal_ranks.append(reciprocal_rank)
                
                if reciprocal_ranks:
                    evaluation[metric] = {
                        'mean': np.mean(reciprocal_ranks),
                        'std': np.std(reciprocal_ranks),
                        'median': np.median(reciprocal_ranks)
                    }
        
        return evaluation
    
    def _aggregate_results(self, 
                          all_results: List[Dict[str, Any]],
                          algorithm_times: Dict[str, List[float]]) -> Dict[str, Any]:
        """
        汇总多轮实验结果
        
        Args:
            all_results: 所有轮次的结果
            algorithm_times: 算法运行时间
            
        Returns:
            汇总结果
        """
        # 按算法分组
        algo_results = {}
        for result in all_results:
            algo_name = result['algorithm']
            if algo_name not in algo_results:
                algo_results[algo_name] = []
            algo_results[algo_name].append(result)
        
        # 汇总每个算法的结果
        aggregated = {
            'experiment_summary': {
                'total_rounds': len(set(r['round'] for r in all_results)),
                'algorithms': list(algo_results.keys()),
                'total_results': len(all_results)
            },
            'algorithm_results': {}
        }
        
        for algo_name, results in algo_results.items():
            # 汇总评估指标
            avg_metrics = {}
            all_metrics = defaultdict(list)
            
            # 收集所有轮次的指标
            for result in results:
                for metric, values in result['evaluation'].items():
                    all_metrics[metric].append(values)
            
            # 计算平均值
            for metric, values_list in all_metrics.items():
                means = [v['mean'] for v in values_list]
                stds = [v['std'] for v in values_list]
                medians = [v['median'] for v in values_list]
                
                avg_metrics[metric] = {
                    'mean': np.mean(means),
                    'std': np.mean(stds),
                    'median': np.median(medians),
                    'rounds_mean': means,
                    'rounds_std': stds
                }
            
            # 汇总时间
            if algo_name in algorithm_times:
                time_mean = np.mean(algorithm_times[algo_name])
                time_std = np.std(algorithm_times[algo_name])
            else:
                time_mean = 0
                time_std = 0
            
            # 保存每个查询的结果（最后一轮）
            last_round_result = None
            for result in sorted(results, key=lambda x: x['round'], reverse=True):
                if 'individual_results' in result:
                    last_round_result = result['individual_results']
                    break
            
            aggregated['algorithm_results'][algo_name] = {
                'average_metrics': avg_metrics,
                'average_time_per_query': {
                    'mean': time_mean,
                    'std': time_std
                },
                'rounds_run': len(results),
                'last_round_results': last_round_result
            }
        
        # 计算最佳算法
        best_algo_per_metric = {}
        for metric in ['precision@1', 'precision@5', 'precision@10', 'map', 'mrr']:
            best_value = -1
            best_algo = None
            
            for algo_name, results in aggregated['algorithm_results'].items():
                if metric in results['average_metrics']:
                    value = results['average_metrics'][metric]['mean']
                    if value > best_value:
                        best_value = value
                        best_algo = algo_name
            
            if best_algo:
                best_algo_per_metric[metric] = {
                    'algorithm': best_algo,
                    'value': best_value
                }
        
        aggregated['best_algorithms'] = best_algo_per_metric
        
        return aggregated
    
    def _generate_experiment_report(self, 
                                  results: Dict[str, Any],
                                  experiment_dir: str) -> str:
        """
        生成实验报告
        
        Args:
            results: 实验结果
            experiment_dir: 实验目录
            
        Returns:
            报告路径
        """
        try:
            # 创建HTML报告
            html_content = self._generate_html_report(results, experiment_dir)
            
            # 保存HTML报告
            html_path = os.path.join(experiment_dir, 'experiment_report.html')
            with open(html_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            # 复制到reports目录
            report_filename = os.path.basename(experiment_dir) + '_report.html'
            report_path = os.path.join(self.reports_dir, report_filename)
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            return report_path
            
        except Exception as e:
            logger.error(f"生成报告失败: {e}")
            return None
    
    def _generate_html_report(self, 
                             results: Dict[str, Any],
                             experiment_dir: str) -> str:
        """
        生成HTML报告内容
        """
        import datetime
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>对比实验报告</title>
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
                .metric-best {{
                    background-color: #d4edda;
                    font-weight: bold;
                }}
                .summary-stats {{
                    display: flex;
                    justify-content: space-around;
                    margin: 20px 0;
                    flex-wrap: wrap;
                }}
                .stat-card {{
                    background-color: #f8f9fa;
                    padding: 20px;
                    border-radius: 5px;
                    text-align: center;
                    margin: 10px;
                    min-width: 200px;
                }}
                .stat-value {{
                    font-size: 24px;
                    font-weight: bold;
                    color: #4CAF50;
                }}
                .timestamp {{
                    color: #666;
                    font-style: italic;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>多模态检索算法对比实验报告</h1>
                <p class="timestamp">生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                
                <div class="summary-stats">
                    <div class="stat-card">
                        <div class="stat-label">实验轮数</div>
                        <div class="stat-value">{results['experiment_summary']['total_rounds']}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">算法数量</div>
                        <div class="stat-value">{len(results['experiment_summary']['algorithms'])}</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">实验目录</div>
                        <div class="stat-value">{os.path.basename(experiment_dir)}</div>
                    </div>
                </div>
                
                <h2>1. 实验结果摘要</h2>
                <table>
                    <tr>
                        <th>算法</th>
                        <th>Precision@1</th>
                        <th>Precision@5</th>
                        <th>Precision@10</th>
                        <th>Recall@10</th>
                        <th>MAP</th>
                        <th>MRR</th>
                        <th>平均时间(秒/查询)</th>
                    </tr>
        """
        
        # 构建结果表格
        for algo_name, algo_results in results['algorithm_results'].items():
            html += f"<tr>\n"
            html += f"<td>{algo_name}</td>\n"
            
            metrics = ['precision@1', 'precision@5', 'precision@10', 'recall@10', 'map', 'mrr']
            for metric in metrics:
                if metric in algo_results['average_metrics']:
                    value = algo_results['average_metrics'][metric]['mean']
                    # 检查是否是最佳
                    is_best = False
                    if metric in results['best_algorithms']:
                        is_best = results['best_algorithms'][metric]['algorithm'] == algo_name
                    
                    class_attr = ' class="metric-best"' if is_best else ''
                    html += f"<td{class_attr}>{value:.4f}</td>\n"
                else:
                    html += "<td>-</td>\n"
            
            # 添加时间
            time_value = algo_results['average_time_per_query']['mean']
            html += f"<td>{time_value:.6f}</td>\n"
            html += "</tr>\n"
        
        html += """
                </table>
        """
        
        # 最佳算法总结
        html += """
                <h2>2. 最佳算法总结</h2>
                <table>
                    <tr>
                        <th>指标</th>
                        <th>最佳算法</th>
                        <th>最佳值</th>
                    </tr>
        """
        
        for metric, best_info in results['best_algorithms'].items():
            html += f"""
                    <tr>
                        <td>{metric}</td>
                        <td>{best_info['algorithm']}</td>
                        <td>{best_info['value']:.4f}</td>
                    </tr>
            """
        
        html += """
                </table>
        """
        
        # 详细算法结果
        html += """
                <h2>3. 算法详细结果</h2>
        """
        
        for algo_name, algo_results in results['algorithm_results'].items():
            html += f"""
                <h3>{algo_name}</h3>
                <table>
                    <tr>
                        <th>指标</th>
                        <th>平均值</th>
                        <th>标准差</th>
                        <th>中位数</th>
                    </tr>
            """
            
            for metric, values in algo_results['average_metrics'].items():
                html += f"""
                    <tr>
                        <td>{metric}</td>
                        <td>{values['mean']:.4f}</td>
                        <td>{values['std']:.4f}</td>
                        <td>{values['median']:.4f}</td>
                    </tr>
                """
            
            # 添加时间信息
            time_info = algo_results['average_time_per_query']
            html += f"""
                    <tr>
                        <td>平均查询时间</td>
                        <td>{time_info['mean']:.6f}秒</td>
                        <td>{time_info['std']:.6f}秒</td>
                        <td>-</td>
                    </tr>
                </table>
                <p><strong>运行轮次:</strong> {algo_results['rounds_run']}</p>
            """
        
        html += """
            </div>
        </body>
        </html>
        """
        
        return html
    
    def visualize_comparison(self, 
                           results: Dict[str, Any],
                           metrics: List[str] = None,
                           output_dir: Optional[str] = None) -> List[str]:
        """
        可视化对比结果
        
        Args:
            results: 实验结果
            metrics: 要可视化的指标
            output_dir: 输出目录
            
        Returns:
            生成的图表路径列表
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 设置样式
            sns.set_style("whitegrid")
            sns.set_palette("husl")
            
            # 设置输出目录
            viz_dir = output_dir or os.path.join(self.figures_dir, 'comparison')
            os.makedirs(viz_dir, exist_ok=True)
            
            # 默认指标
            if metrics is None:
                metrics = ['precision@1', 'precision@5', 'precision@10', 'map', 'mrr']
            
            # 准备数据
            data = []
            for algo_name, algo_results in results['algorithm_results'].items():
                for metric in metrics:
                    if metric in algo_results['average_metrics']:
                        data.append({
                            'algorithm': algo_name,
                            'metric': metric,
                            'value': algo_results['average_metrics'][metric]['mean'],
                            'std': algo_results['average_metrics'][metric]['std']
                        })
            
            if not data:
                logger.warning("没有足够的数据进行可视化")
                return []
            
            # 创建DataFrame
            import pandas as pd
            df = pd.DataFrame(data)
            
            # 生成图表
            image_paths = []
            
            # 1. 柱状图对比
            plt.figure(figsize=(12, 6))
            ax = sns.barplot(x='metric', y='value', hue='algorithm', data=df, capsize=0.1)
            plt.title('算法性能指标对比')
            plt.xlabel('评估指标')
            plt.ylabel('指标值')
            plt.ylim(0, 1.1)
            
            # 添加数值标签
            for p in ax.patches:
                height = p.get_height()
                ax.text(p.get_x() + p.get_width() / 2., height + 0.02, 
                       f'{height:.3f}', ha='center', va='bottom')
            
            bar_path = os.path.join(viz_dir, 'metrics_comparison.png')
            plt.tight_layout()
            plt.savefig(bar_path, dpi=300, bbox_inches='tight')
            image_paths.append(bar_path)
            
            # 2. 时间对比图
            time_data = []
            for algo_name, algo_results in results['algorithm_results'].items():
                time_data.append({
                    'algorithm': algo_name,
                    'time': algo_results['average_time_per_query']['mean'],
                    'std': algo_results['average_time_per_query']['std']
                })
            
            if time_data:
                time_df = pd.DataFrame(time_data)
                plt.figure(figsize=(10, 6))
                ax = sns.barplot(x='algorithm', y='time', data=time_df, capsize=0.1)
                plt.title('算法速度对比')
                plt.xlabel('算法')
                plt.ylabel('平均查询时间 (秒)')
                plt.xticks(rotation=45, ha='right')
                
                # 添加数值标签
                for p in ax.patches:
                    height = p.get_height()
                    ax.text(p.get_x() + p.get_width() / 2., height + 0.0001, 
                           f'{height:.6f}', ha='center', va='bottom', fontsize=9)
                
                time_path = os.path.join(viz_dir, 'speed_comparison.png')
                plt.tight_layout()
                plt.savefig(time_path, dpi=300, bbox_inches='tight')
                image_paths.append(time_path)
            
            logger.info(f"已生成 {len(image_paths)} 个可视化图表")
            return image_paths
            
        except ImportError as e:
            logger.error(f"导入可视化库失败: {e}")
            return []
        except Exception as e:
            logger.error(f"可视化失败: {e}")
            return []

# 示例用法
if __name__ == '__main__':
    # 创建实验管理器
    manager = ExperimentManager(
        results_dir='./results',
        dataset_dir='./data',
        save_results=True
    )
    
    # 示例：准备SIFT codebook
    # codebook_path = manager.prepare_sift_codebook(
    #     dataset_path='./data/synthetic/images',
    #     codebook_size=1024,
    #     num_samples=100
    # )
    
    # 示例：运行对比实验
    # algorithms = []  # 这里应该初始化算法实例
    # dataset = {'name': 'Synthetic Dataset', 'path': './data/synthetic'}
    # 
    # results = manager.run_comparison_experiment(
    #     algorithms=algorithms,
    #     dataset=dataset,
    #     num_rounds=3,
    #     query_size=100,
    #     experiment_name='example_comparison'
    # )
    
    logger.info("实验管理器已初始化，准备运行实验")