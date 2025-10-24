import numpy as np
from typing import List, Dict, Tuple, Optional, Callable, Any
import pandas as pd
import logging
from scipy import stats
from collections import defaultdict

logger = logging.getLogger(__name__)

class AdvancedEvaluator:
    """
    高级评估系统
    
    支持Top-K准确率、MRR、MAP等多种评估指标，包含单次评估、多轮聚合和算法比较功能
    """
    
    def __init__(self, 
                 metrics: Optional[List[str]] = None,
                 max_k: int = 10,
                 save_results: bool = False,
                 output_dir: str = './results'):
        """
        Args:
            metrics: 评估指标列表
            max_k: 最大的K值
            save_results: 是否保存结果
            output_dir: 输出目录
        """
        # 默认评估指标
        default_metrics = ['precision_at_k', 'recall_at_k', 'mrr', 'map', 'ndcg']
        
        self.metrics = metrics or default_metrics
        self.max_k = max_k
        self.save_results = save_results
        self.output_dir = output_dir
        
        # 预定义的K值列表
        self.k_values = [1, 3, 5, 10] if max_k >= 10 else [k for k in [1, 3, 5, 10] if k <= max_k]
        
        # 结果存储
        self.results_history = []
        self.comparison_results = []
        
        logger.info(f"初始化高级评估器: metrics={self.metrics}, max_k={self.max_k}")
    
    def evaluate_single(self, 
                       query_results: List[Tuple[int, float]],
                       ground_truth_indices: List[int],
                       query_id: Optional[str] = None) -> Dict[str, Any]:
        """
        评估单个查询的结果
        
        Args:
            query_results: 查询结果 [(索引, 相似度)]
            ground_truth_indices: 真实正样本索引列表
            query_id: 查询ID
            
        Returns:
            评估指标字典
        """
        result = {
            'query_id': query_id,
            'total_relevant': len(ground_truth_indices),
            'retrieved_count': len(query_results)
        }
        
        # 提取排序后的索引
        retrieved_indices = [idx for idx, _ in query_results]
        retrieved_scores = [score for _, score in query_results]
        
        # 标记相关结果
        is_relevant = [idx in ground_truth_indices for idx in retrieved_indices]
        
        # 计算各种指标
        if 'precision_at_k' in self.metrics:
            precision_values = self._calculate_precision_at_k(is_relevant)
            result['precision'] = precision_values
        
        if 'recall_at_k' in self.metrics:
            recall_values = self._calculate_recall_at_k(is_relevant, len(ground_truth_indices))
            result['recall'] = recall_values
        
        if 'mrr' in self.metrics:
            mrr = self._calculate_mrr(is_relevant)
            result['mrr'] = mrr
        
        if 'map' in self.metrics:
            ap = self._calculate_average_precision(is_relevant)
            result['map'] = ap
        
        if 'ndcg' in self.metrics:
            ndcg_values = self._calculate_ndcg(retrieved_scores, is_relevant)
            result['ndcg'] = ndcg_values
        
        if 'f1_at_k' in self.metrics:
            f1_values = self._calculate_f1_at_k(is_relevant, len(ground_truth_indices))
            result['f1'] = f1_values
        
        # 存储详细结果
        result['retrieved_indices'] = retrieved_indices
        result['retrieved_scores'] = retrieved_scores
        result['is_relevant'] = is_relevant
        
        return result
    
    def evaluate_batch(self, 
                      all_query_results: List[List[Tuple[int, float]]],
                      all_ground_truth_indices: List[List[int]],
                      query_ids: Optional[List[str]] = None,
                      algorithm_name: Optional[str] = None,
                      dataset_name: Optional[str] = None) -> Dict[str, Any]:
        """
        评估一批查询的结果
        
        Args:
            all_query_results: 所有查询的结果列表
            all_ground_truth_indices: 所有查询的真实正样本索引列表
            query_ids: 查询ID列表
            algorithm_name: 算法名称
            dataset_name: 数据集名称
            
        Returns:
            聚合评估指标
        """
        logger.info(f"开始批量评估: {len(all_query_results)}个查询")
        
        # 验证输入
        if len(all_query_results) != len(all_ground_truth_indices):
            raise ValueError("查询结果数量与真实值数量不匹配")
        
        # 初始化聚合结果
        aggregate_result = {
            'algorithm_name': algorithm_name,
            'dataset_name': dataset_name,
            'query_count': len(all_query_results),
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'individual_results': []
        }
        
        # 批量评估
        metrics_list = defaultdict(list)
        
        for i, (query_results, ground_truth) in enumerate(zip(all_query_results, all_ground_truth_indices)):
            query_id = query_ids[i] if query_ids else f"query_{i}"
            
            # 评估单个查询
            single_result = self.evaluate_single(
                query_results, ground_truth, query_id=query_id
            )
            
            aggregate_result['individual_results'].append(single_result)
            
            # 收集指标
            if 'precision' in single_result:
                for k, v in single_result['precision'].items():
                    metrics_list[f'precision@{k}'].append(v)
            
            if 'recall' in single_result:
                for k, v in single_result['recall'].items():
                    metrics_list[f'recall@{k}'].append(v)
            
            if 'mrr' in single_result:
                metrics_list['mrr'].append(single_result['mrr'])
            
            if 'map' in single_result:
                metrics_list['map'].append(single_result['map'])
            
            if 'ndcg' in single_result:
                for k, v in single_result['ndcg'].items():
                    metrics_list[f'ndcg@{k}'].append(v)
            
            if 'f1' in single_result:
                for k, v in single_result['f1'].items():
                    metrics_list[f'f1@{k}'].append(v)
        
        # 计算平均值
        aggregate_result['average_metrics'] = {}
        for metric_name, values in metrics_list.items():
            aggregate_result['average_metrics'][metric_name] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'median': np.median(values),
                'min': np.min(values),
                'max': np.max(values)
            }
        
        # 存储历史记录
        self.results_history.append(aggregate_result)
        
        logger.info(f"批量评估完成")
        self._print_evaluation_summary(aggregate_result)
        
        # 保存结果
        if self.save_results:
            self._save_evaluation_results(aggregate_result)
        
        return aggregate_result
    
    def _calculate_precision_at_k(self, is_relevant: List[bool]) -> Dict[int, float]:
        """
        计算不同K值的精确率
        """
        precision_dict = {}
        
        for k in self.k_values:
            if k > len(is_relevant):
                # 如果K大于检索结果数量，使用实际数量
                relevant_at_k = sum(is_relevant)
                precision = relevant_at_k / len(is_relevant) if len(is_relevant) > 0 else 0.0
            else:
                relevant_at_k = sum(is_relevant[:k])
                precision = relevant_at_k / k
            
            precision_dict[k] = precision
        
        return precision_dict
    
    def _calculate_recall_at_k(self, is_relevant: List[bool], total_relevant: int) -> Dict[int, float]:
        """
        计算不同K值的召回率
        """
        recall_dict = {}
        
        if total_relevant == 0:
            # 如果没有相关样本，召回率为1
            return {k: 1.0 for k in self.k_values}
        
        for k in self.k_values:
            if k > len(is_relevant):
                relevant_at_k = sum(is_relevant)
            else:
                relevant_at_k = sum(is_relevant[:k])
            
            recall = relevant_at_k / total_relevant
            recall_dict[k] = recall
        
        return recall_dict
    
    def _calculate_f1_at_k(self, is_relevant: List[bool], total_relevant: int) -> Dict[int, float]:
        """
        计算不同K值的F1分数
        """
        f1_dict = {}
        
        precision_dict = self._calculate_precision_at_k(is_relevant)
        recall_dict = self._calculate_recall_at_k(is_relevant, total_relevant)
        
        for k in self.k_values:
            p = precision_dict[k]
            r = recall_dict[k]
            
            if p + r > 0:
                f1 = 2 * p * r / (p + r)
            else:
                f1 = 0.0
            
            f1_dict[k] = f1
        
        return f1_dict
    
    def _calculate_mrr(self, is_relevant: List[bool]) -> float:
        """
        计算平均倒数排名 (MRR)
        """
        for i, relevant in enumerate(is_relevant):
            if relevant:
                return 1.0 / (i + 1)
        
        return 0.0
    
    def _calculate_average_precision(self, is_relevant: List[bool]) -> float:
        """
        计算平均精度 (AP)
        """
        if not any(is_relevant):
            return 0.0
        
        # 计算精确率和累积相关样本
        precision_values = []
        relevant_count = 0
        
        for i, relevant in enumerate(is_relevant):
            if relevant:
                relevant_count += 1
                # 计算到当前位置的精确率
                precision = relevant_count / (i + 1)
                precision_values.append(precision)
        
        # 平均精确率
        return sum(precision_values) / len(precision_values)
    
    def _calculate_ndcg(self, scores: List[float], is_relevant: List[bool]) -> Dict[int, float]:
        """
        计算归一化折损累积增益 (NDCG)
        """
        ndcg_dict = {}
        
        for k in self.k_values:
            if k > len(scores):
                actual_k = len(scores)
            else:
                actual_k = k
            
            # 计算DCG
            dcg = 0.0
            for i in range(actual_k):
                if is_relevant[i]:
                    dcg += 1.0 / np.log2(i + 2)  # 排名从1开始，所以+2
            
            # 计算理想DCG（按相关性降序排列）
            sorted_relevant = sorted(is_relevant[:actual_k], reverse=True)
            idcg = 0.0
            for i, relevant in enumerate(sorted_relevant):
                if relevant:
                    idcg += 1.0 / np.log2(i + 2)
            
            # 计算NDCG
            if idcg > 0:
                ndcg = dcg / idcg
            else:
                ndcg = 0.0
            
            ndcg_dict[k] = ndcg
        
        return ndcg_dict
    
    def _print_evaluation_summary(self, aggregate_result: Dict[str, Any]):
        """
        打印评估摘要
        """
        algo_name = aggregate_result.get('algorithm_name', 'Unknown')
        dataset_name = aggregate_result.get('dataset_name', 'Unknown')
        
        logger.info(f"===== 评估摘要 ====-")
        logger.info(f"算法: {algo_name}")
        logger.info(f"数据集: {dataset_name}")
        logger.info(f"查询数量: {aggregate_result['query_count']}")
        logger.info("平均指标:")
        
        metrics = aggregate_result.get('average_metrics', {})
        
        # 按指标类型分组打印
        precision_metrics = {k: v for k, v in metrics.items() if k.startswith('precision')}
        recall_metrics = {k: v for k, v in metrics.items() if k.startswith('recall')}
        f1_metrics = {k: v for k, v in metrics.items() if k.startswith('f1')}
        ndcg_metrics = {k: v for k, v in metrics.items() if k.startswith('ndcg')}
        single_metrics = {k: v for k, v in metrics.items() if k not in precision_metrics and 
                          k not in recall_metrics and k not in f1_metrics and k not in ndcg_metrics}
        
        # 打印单值指标
        for metric, stats_dict in single_metrics.items():
            logger.info(f"  {metric:10s}: {stats_dict['mean']:.4f} ± {stats_dict['std']:.4f}")
        
        # 打印K值相关指标
        if precision_metrics:
            logger.info("  精确率:")
            for k in self.k_values:
                if f'precision@{k}' in precision_metrics:
                    stats_dict = precision_metrics[f'precision@{k}']
                    logger.info(f"    Precision@{k}: {stats_dict['mean']:.4f} ± {stats_dict['std']:.4f}")
        
        if recall_metrics:
            logger.info("  召回率:")
            for k in self.k_values:
                if f'recall@{k}' in recall_metrics:
                    stats_dict = recall_metrics[f'recall@{k}']
                    logger.info(f"    Recall@{k}: {stats_dict['mean']:.4f} ± {stats_dict['std']:.4f}")
        
        if f1_metrics:
            logger.info("  F1分数:")
            for k in self.k_values:
                if f'f1@{k}' in f1_metrics:
                    stats_dict = f1_metrics[f'f1@{k}']
                    logger.info(f"    F1@{k}: {stats_dict['mean']:.4f} ± {stats_dict['std']:.4f}")
        
        if ndcg_metrics:
            logger.info("  NDCG:")
            for k in self.k_values:
                if f'ndcg@{k}' in ndcg_metrics:
                    stats_dict = ndcg_metrics[f'ndcg@{k}']
                    logger.info(f"    NDCG@{k}: {stats_dict['mean']:.4f} ± {stats_dict['std']:.4f}")
        
        logger.info(f"==================")
    
    def _save_evaluation_results(self, aggregate_result: Dict[str, Any]):
        """
        保存评估结果
        """
        try:
            # 创建输出目录
            import os
            os.makedirs(self.output_dir, exist_ok=True)
            
            # 生成文件名
            algo_name = aggregate_result.get('algorithm_name', 'Unknown').replace(' ', '_')
            dataset_name = aggregate_result.get('dataset_name', 'Unknown').replace(' ', '_')
            timestamp = aggregate_result.get('timestamp', '').replace(':', '-').replace(' ', '_')
            
            filename = f"{algo_name}_{dataset_name}_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            # 保存为JSON
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(aggregate_result, f, indent=2, ensure_ascii=False)
            
            logger.info(f"评估结果已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"保存评估结果失败: {e}")
    
    def compare_algorithms(self, 
                          algorithm_results_list: List[Dict[str, Any]],
                          metrics_to_compare: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        比较多个算法的性能
        
        Args:
            algorithm_results_list: 多个算法的评估结果列表
            metrics_to_compare: 要比较的指标列表
            
        Returns:
            比较结果
        """
        logger.info(f"开始算法比较: {len(algorithm_results_list)}个算法")
        
        comparison = {
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'algorithms': [],
            'metric_comparison': {}
        }
        
        # 提取所有指标名称
        all_metrics = set()
        for result in algorithm_results_list:
            if 'average_metrics' in result:
                all_metrics.update(result['average_metrics'].keys())
        
        # 过滤要比较的指标
        if metrics_to_compare:
            metrics_to_compare = [m for m in metrics_to_compare if m in all_metrics]
        else:
            metrics_to_compare = list(all_metrics)
        
        # 比较每个指标
        for metric in metrics_to_compare:
            comparison['metric_comparison'][metric] = {}
            
            # 提取所有算法的该指标值
            metric_values = []
            algorithm_names = []
            
            for result in algorithm_results_list:
                algo_name = result.get('algorithm_name', 'Unknown')
                algorithm_names.append(algo_name)
                
                if algo_name not in comparison['algorithms']:
                    comparison['algorithms'].append(algo_name)
                
                if 'average_metrics' in result and metric in result['average_metrics']:
                    metric_values.append(result['average_metrics'][metric]['mean'])
                else:
                    metric_values.append(None)
            
            # 记录每个算法的值
            comparison['metric_comparison'][metric]['values'] = {}
            for algo_name, value in zip(algorithm_names, metric_values):
                comparison['metric_comparison'][metric]['values'][algo_name] = value
            
            # 找出最佳算法
            valid_values = [(algo, val) for algo, val in zip(algorithm_names, metric_values) if val is not None]
            if valid_values:
                best_algo, best_value = max(valid_values, key=lambda x: x[1])
                comparison['metric_comparison'][metric]['best_algorithm'] = best_algo
                comparison['metric_comparison'][metric]['best_value'] = best_value
        
        # 存储比较结果
        self.comparison_results.append(comparison)
        
        logger.info(f"算法比较完成")
        self._print_comparison_summary(comparison)
        
        # 保存结果
        if self.save_results:
            self._save_comparison_results(comparison)
        
        return comparison
    
    def _print_comparison_summary(self, comparison: Dict[str, Any]):
        """
        打印比较摘要
        """
        logger.info(f"===== 算法比较摘要 ====-")
        
        for metric, metric_data in comparison['metric_comparison'].items():
            logger.info(f"指标: {metric}")
            for algo_name, value in metric_data['values'].items():
                if value is not None:
                    logger.info(f"  {algo_name}: {value:.4f}")
                else:
                    logger.info(f"  {algo_name}: N/A")
            
            if 'best_algorithm' in metric_data:
                logger.info(f"  最佳: {metric_data['best_algorithm']} ({metric_data['best_value']:.4f})")
            
            logger.info()
        
        logger.info(f"=====================")
    
    def _save_comparison_results(self, comparison: Dict[str, Any]):
        """
        保存比较结果
        """
        try:
            # 创建输出目录
            import os
            os.makedirs(self.output_dir, exist_ok=True)
            
            # 生成文件名
            timestamp = comparison.get('timestamp', '').replace(':', '-').replace(' ', '_')
            algorithms_str = '_vs_'.join([a.replace(' ', '_')[:10] for a in comparison['algorithms']])
            filename = f"comparison_{algorithms_str}_{timestamp}.json"
            filepath = os.path.join(self.output_dir, filename)
            
            # 保存为JSON
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(comparison, f, indent=2, ensure_ascii=False)
            
            logger.info(f"比较结果已保存到: {filepath}")
            
        except Exception as e:
            logger.error(f"保存比较结果失败: {e}")
    
    def run_statistical_test(self, 
                           result1: Dict[str, Any],
                           result2: Dict[str, Any],
                           metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        运行统计显著性检验
        
        Args:
            result1: 第一个算法的结果
            result2: 第二个算法的结果
            metrics: 要测试的指标列表
            
        Returns:
            统计检验结果
        """
        logger.info(f"运行统计显著性检验")
        
        # 验证两个结果集的查询数量是否相同
        if result1['query_count'] != result2['query_count']:
            logger.warning("两个算法的查询数量不同，无法进行配对t检验")
            return None
        
        test_results = {
            'algorithm1': result1.get('algorithm_name', 'Algorithm1'),
            'algorithm2': result2.get('algorithm_name', 'Algorithm2'),
            'query_count': result1['query_count'],
            'tests': {}
        }
        
        # 获取所有可能的指标
        all_metrics = set()
        for i, result in enumerate([result1, result2]):
            if 'individual_results' not in result:
                logger.warning(f"结果{i+1}缺少详细的单个查询结果")
                return None
        
        # 确定要测试的指标
        if not metrics:
            metrics = ['precision@1', 'precision@5', 'precision@10', 
                       'recall@1', 'recall@5', 'recall@10', 
                       'mrr', 'map', 'ndcg@1', 'ndcg@5', 'ndcg@10']
        
        # 对每个指标进行测试
        for metric in metrics:
            try:
                # 提取每个查询的指标值
                values1 = []
                values2 = []
                
                # 根据指标类型提取值
                if '@' in metric:
                    # Precision@k, Recall@k, F1@k, NDCG@k
                    metric_type, k_str = metric.split('@')
                    k = int(k_str)
                    
                    for i, result in enumerate([result1, result2]):
                        for individual in result['individual_results']:
                            if metric_type in individual:
                                value = individual[metric_type].get(k, 0.0)
                                if i == 0:
                                    values1.append(value)
                                else:
                                    values2.append(value)
                else:
                    # MRR, MAP
                    for i, result in enumerate([result1, result2]):
                        for individual in result['individual_results']:
                            if metric in individual:
                                value = individual[metric]
                                if i == 0:
                                    values1.append(value)
                                else:
                                    values2.append(value)
                
                # 确保有足够的数据
                if len(values1) != len(values2) or len(values1) == 0:
                    logger.warning(f"无法为指标{metric}提取足够的数据点")
                    continue
                
                # 配对t检验
                t_stat, p_value = stats.ttest_rel(values1, values2)
                
                # Cohen's d效应量
                mean_diff = np.mean([v1 - v2 for v1, v2 in zip(values1, values2)])
                pooled_std = np.sqrt((np.std(values1)**2 + np.std(values2)**2) / 2)
                cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
                
                test_results['tests'][metric] = {
                    't_statistic': t_stat,
                    'p_value': p_value,
                    'cohens_d': cohens_d,
                    'significant': p_value < 0.05,
                    'algorithm1_mean': np.mean(values1),
                    'algorithm2_mean': np.mean(values2),
                    'mean_difference': mean_diff
                }
                
                logger.info(f"指标{metric}的统计检验完成")
                
            except Exception as e:
                logger.error(f"为指标{metric}运行统计检验时出错: {e}")
        
        # 打印统计检验结果
        self._print_statistical_test_results(test_results)
        
        return test_results
    
    def _print_statistical_test_results(self, test_results: Dict[str, Any]):
        """
        打印统计检验结果
        """
        logger.info(f"===== 统计显著性检验结果 ====-")
        logger.info(f"算法1: {test_results['algorithm1']}")
        logger.info(f"算法2: {test_results['algorithm2']}")
        
        for metric, result in test_results['tests'].items():
            logger.info(f"指标: {metric}")
            logger.info(f"  t统计量: {result['t_statistic']:.4f}")
            logger.info(f"  p值: {result['p_value']:.6f}")
            logger.info(f"  Cohen's d: {result['cohens_d']:.4f}")
            logger.info(f"  显著性: {'是' if result['significant'] else '否'}")
            logger.info(f"  算法1均值: {result['algorithm1_mean']:.4f}")
            logger.info(f"  算法2均值: {result['algorithm2_mean']:.4f}")
            logger.info(f"  均值差: {result['mean_difference']:.4f}")
            
            # 解释结果
            if result['significant']:
                if result['mean_difference'] > 0:
                    logger.info(f"  结论: {test_results['algorithm1']}显著优于{test_results['algorithm2']}")
                else:
                    logger.info(f"  结论: {test_results['algorithm2']}显著优于{test_results['algorithm1']}")
            else:
                logger.info(f"  结论: 两个算法性能没有显著差异")
            
            logger.info()
        
        logger.info(f"=============================")
    
    def export_detailed_results(self, 
                               results: Dict[str, Any],
                               filepath: str,
                               format: str = 'csv'):
        """
        导出详细结果
        
        Args:
            results: 评估结果
            filepath: 导出路径
            format: 导出格式 ('csv', 'excel')
        """
        try:
            if 'individual_results' not in results:
                logger.error("结果中缺少详细的单个查询结果")
                return
            
            # 准备数据
            data = []
            
            for individual in results['individual_results']:
                row = {
                    'query_id': individual.get('query_id', ''),
                    'total_relevant': individual.get('total_relevant', 0),
                    'retrieved_count': individual.get('retrieved_count', 0)
                }
                
                # 添加各种指标
                if 'precision' in individual:
                    for k, v in individual['precision'].items():
                        row[f'precision@{k}'] = v
                
                if 'recall' in individual:
                    for k, v in individual['recall'].items():
                        row[f'recall@{k}'] = v
                
                if 'f1' in individual:
                    for k, v in individual['f1'].items():
                        row[f'f1@{k}'] = v
                
                if 'ndcg' in individual:
                    for k, v in individual['ndcg'].items():
                        row[f'ndcg@{k}'] = v
                
                if 'mrr' in individual:
                    row['mrr'] = individual['mrr']
                
                if 'map' in individual:
                    row['map'] = individual['map']
                
                data.append(row)
            
            # 创建DataFrame
            df = pd.DataFrame(data)
            
            # 导出
            if format == 'csv':
                df.to_csv(filepath, index=False, encoding='utf-8')
                logger.info(f"详细结果已导出到CSV: {filepath}")
            elif format == 'excel':
                df.to_excel(filepath, index=False)
                logger.info(f"详细结果已导出到Excel: {filepath}")
            else:
                logger.error(f"不支持的导出格式: {format}")
                
        except Exception as e:
            logger.error(f"导出详细结果失败: {e}")
    
    def get_summary_report(self) -> str:
        """
        获取摘要报告
        
        Returns:
            摘要报告字符串
        """
        report = "===== 评估器摘要报告 =====\n"
        report += f"评估的算法总数: {len(self.results_history)}\n"
        report += f"比较的算法组总数: {len(self.comparison_results)}\n"
        
        if self.results_history:
            report += "\n最近的评估结果:\n"
            for i, result in enumerate(self.results_history[-5:][::-1]):
                algo_name = result.get('algorithm_name', 'Unknown')
                dataset_name = result.get('dataset_name', 'Unknown')
                metrics = result.get('average_metrics', {})
                
                report += f"  {i+1}. {algo_name} on {dataset_name}:\n"
                if 'map' in metrics:
                    report += f"     MAP: {metrics['map']['mean']:.4f}\n"
                if 'mrr' in metrics:
                    report += f"     MRR: {metrics['mrr']['mean']:.4f}\n"
                if 'precision@1' in metrics:
                    report += f"     Precision@1: {metrics['precision@1']['mean']:.4f}\n"
        
        report += "\n=======================\n"
        
        return report