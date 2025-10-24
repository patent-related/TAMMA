import os
import json
import yaml
import copy
from typing import Dict, Any, Optional, List, Tuple, Union
import logging
import argparse
import importlib.util
import inspect

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ConfigManager:
    """
    配置管理工具类
    
    提供配置加载、保存、验证、参数解析等功能
    """
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        加载配置文件
        
        Args:
            config_path: 配置文件路径
            
        Returns:
            配置字典
        """
        try:
            # 检查文件是否存在
            if not os.path.exists(config_path):
                logger.error(f"配置文件不存在: {config_path}")
                return {}
            
            # 根据文件扩展名选择加载方法
            ext = os.path.splitext(config_path)[1].lower()
            
            with open(config_path, 'r', encoding='utf-8') as f:
                if ext == '.json':
                    config = json.load(f)
                elif ext in ['.yaml', '.yml']:
                    config = yaml.safe_load(f)
                else:
                    logger.error(f"不支持的配置文件格式: {ext}")
                    return {}
            
            logger.info(f"成功加载配置文件: {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"加载配置文件失败 {config_path}: {e}")
            return {}
    
    @staticmethod
    def save_config(config: Dict[str, Any],
                   output_path: str,
                   format: str = 'json') -> bool:
        """
        保存配置文件
        
        Args:
            config: 配置字典
            output_path: 输出路径
            format: 输出格式 (json/yaml)
            
        Returns:
            是否保存成功
        """
        try:
            # 确保目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 根据格式保存
            with open(output_path, 'w', encoding='utf-8') as f:
                if format.lower() == 'yaml':
                    yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
                else:  # json
                    json.dump(config, f, ensure_ascii=False, indent=2)
            
            logger.info(f"成功保存配置文件: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"保存配置文件失败 {output_path}: {e}")
            return False
    
    @staticmethod
    def validate_config(config: Dict[str, Any],
                       schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        验证配置
        
        Args:
            config: 配置字典
            schema: 配置模式
            
        Returns:
            (是否有效, 错误列表)
        """
        errors = []
        
        def validate_section(config_section: Dict[str, Any],
                            schema_section: Dict[str, Any],
                            path: str = '') -> None:
            """验证配置的某个部分"""
            # 检查必需字段
            for key, properties in schema_section.items():
                if isinstance(properties, dict):
                    # 字段属性
                    required = properties.get('required', False)
                    field_type = properties.get('type', Any)
                    default = properties.get('default', None)
                    
                    # 检查字段是否存在
                    if key not in config_section:
                        if required:
                            errors.append(f"{path}{key} 是必需的")
                        elif default is not None:
                            # 使用默认值
                            config_section[key] = default
                        continue
                    
                    # 检查字段类型
                    value = config_section[key]
                    if field_type is not Any and not isinstance(value, field_type):
                        # 尝试转换类型
                        try:
                            if field_type == int:
                                config_section[key] = int(value)
                            elif field_type == float:
                                config_section[key] = float(value)
                            elif field_type == bool:
                                # 处理字符串形式的布尔值
                                if isinstance(value, str):
                                    value_lower = value.lower()
                                    if value_lower in ['true', 'yes', '1']:
                                        config_section[key] = True
                                    elif value_lower in ['false', 'no', '0']:
                                        config_section[key] = False
                                    else:
                                        errors.append(f"{path}{key} 必须是布尔类型，当前值: {value}")
                                else:
                                    errors.append(f"{path}{key} 必须是布尔类型，当前类型: {type(value).__name__}")
                            elif field_type == str:
                                config_section[key] = str(value)
                            elif field_type == list:
                                if isinstance(value, (list, tuple)):
                                    config_section[key] = list(value)
                                else:
                                    errors.append(f"{path}{key} 必须是列表类型")
                            elif field_type == dict:
                                if isinstance(value, dict):
                                    # 递归验证嵌套字典
                                    validate_section(
                                        config_section[key], 
                                        properties.get('schema', {}),
                                        f"{path}{key}."
                                    )
                                else:
                                    errors.append(f"{path}{key} 必须是字典类型")
                        except Exception:
                            errors.append(f"{path}{key} 无法转换为 {field_type.__name__} 类型")
                    
                    # 检查值范围
                    min_val = properties.get('min')
                    max_val = properties.get('max')
                    if isinstance(value, (int, float)):
                        if min_val is not None and value < min_val:
                            errors.append(f"{path}{key} 的值 {value} 小于最小值 {min_val}")
                        if max_val is not None and value > max_val:
                            errors.append(f"{path}{key} 的值 {value} 大于最大值 {max_val}")
                    
                    # 检查枚举值
                    enum = properties.get('enum')
                    if enum is not None and value not in enum:
                        errors.append(f"{path}{key} 的值 {value} 不在允许的枚举值中: {enum}")
            
            # 检查额外字段（可选）
            allow_extra = schema_section.get('allow_extra', True)
            if not allow_extra:
                schema_keys = [k for k in schema_section if isinstance(schema_section[k], dict)]
                for key in config_section:
                    if key not in schema_keys:
                        errors.append(f"{path}{key} 是未知字段")
        
        try:
            validate_section(config, schema)
            
            if errors:
                for error in errors:
                    logger.error(f"配置验证失败: {error}")
                return False, errors
            else:
                logger.info("配置验证成功")
                return True, []
                
        except Exception as e:
            logger.error(f"配置验证过程出错: {e}")
            return False, [str(e)]
    
    @staticmethod
    def merge_configs(base_config: Dict[str, Any],
                     override_config: Dict[str, Any],
                     deep_merge: bool = True) -> Dict[str, Any]:
        """
        合并配置
        
        Args:
            base_config: 基础配置
            override_config: 覆盖配置
            deep_merge: 是否深度合并
            
        Returns:
            合并后的配置
        """
        try:
            result = copy.deepcopy(base_config)
            
            if not deep_merge:
                # 浅合并
                result.update(override_config)
            else:
                # 深度合并
                for key, value in override_config.items():
                    if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                        # 递归合并嵌套字典
                        result[key] = ConfigManager.merge_configs(result[key], value, True)
                    else:
                        # 覆盖值
                        result[key] = copy.deepcopy(value)
            
            return result
            
        except Exception as e:
            logger.error(f"合并配置失败: {e}")
            return base_config
    
    @staticmethod
    def create_parser(config_schema: Dict[str, Any]) -> argparse.ArgumentParser:
        """
        根据配置模式创建命令行参数解析器
        
        Args:
            config_schema: 配置模式
            
        Returns:
            参数解析器
        """
        parser = argparse.ArgumentParser(description='配置参数解析器')
        
        def add_arguments(schema: Dict[str, Any],
                         prefix: str = '') -> None:
            """递归添加参数"""
            for key, properties in schema.items():
                if isinstance(properties, dict):
                    field_type = properties.get('type', Any)
                    default = properties.get('default')
                    help_text = properties.get('help', f'{key} parameter')
                    
                    # 构建参数名
                    arg_name = f'--{prefix}{key}'
                    
                    # 决定参数类型
                    if field_type == bool:
                        # 布尔参数使用store_true或store_false
                        if default is False or default is None:
                            parser.add_argument(arg_name, action='store_true', help=help_text)
                        else:
                            parser.add_argument(arg_name, action='store_false', help=help_text)
                    elif field_type in [int, float, str]:
                        parser.add_argument(arg_name, type=field_type, default=default, help=help_text)
                    elif field_type == list:
                        parser.add_argument(arg_name, type=str, nargs='+', default=default, help=help_text)
                    
                    # 处理嵌套配置
                    nested_schema = properties.get('schema', {})
                    if nested_schema:
                        add_arguments(nested_schema, f'{key}.')
        
        add_arguments(config_schema)
        
        # 添加配置文件参数
        parser.add_argument('--config', type=str, help='配置文件路径')
        
        return parser
    
    @staticmethod
    def parse_arguments(config_schema: Dict[str, Any]) -> Dict[str, Any]:
        """
        解析命令行参数
        
        Args:
            config_schema: 配置模式
            
        Returns:
            配置字典
        """
        parser = ConfigManager.create_parser(config_schema)
        args = parser.parse_args()
        
        # 加载配置文件（如果指定）
        config = {}
        if args.config and os.path.exists(args.config):
            config = ConfigManager.load_config(args.config)
        
        # 覆盖命令行参数
        arg_dict = vars(args)
        
        def update_config_from_args(config_dict: Dict[str, Any],
                                  args_dict: Dict[str, str],
                                  prefix: str = '') -> None:
            """从命令行参数更新配置"""
            for key in list(args_dict.keys()):
                if key == 'config':
                    continue
                
                value = args_dict[key]
                if value is not None:
                    # 处理嵌套参数
                    if '.' in key:
                        parts = key.split('.')
                        current = config_dict
                        for part in parts[:-1]:
                            if part not in current:
                                current[part] = {}
                            current = current[part]
                        current[parts[-1]] = value
                    else:
                        config_dict[key] = value
        
        update_config_from_args(config, arg_dict)
        
        return config
    
    @staticmethod
    def get_class_from_config(config: Dict[str, Any],
                            class_key: str = 'class',
                            module_key: str = 'module',
                            default_module: str = None) -> Optional[Any]:
        """
        从配置中加载类
        
        Args:
            config: 配置字典
            class_key: 类名的键
            module_key: 模块名的键
            default_module: 默认模块名
            
        Returns:
            类对象
        """
        try:
            class_name = config.get(class_key)
            module_name = config.get(module_key, default_module)
            
            if not class_name or not module_name:
                logger.error(f"缺少类名或模块名: class={class_name}, module={module_name}")
                return None
            
            # 导入模块
            module = importlib.import_module(module_name)
            
            # 获取类
            cls = getattr(module, class_name)
            
            logger.info(f"成功加载类: {module_name}.{class_name}")
            return cls
            
        except Exception as e:
            logger.error(f"加载类失败 {config.get(module_key)}.{config.get(class_key)}: {e}")
            return None
    
    @staticmethod
    def instantiate_from_config(config: Dict[str, Any],
                              class_key: str = 'class',
                              module_key: str = 'module',
                              default_module: str = None,
                              **kwargs) -> Optional[Any]:
        """
        从配置实例化对象
        
        Args:
            config: 配置字典
            class_key: 类名的键
            module_key: 模块名的键
            default_module: 默认模块名
            **kwargs: 额外参数
            
        Returns:
            实例化的对象
        """
        try:
            # 加载类
            cls = ConfigManager.get_class_from_config(
                config, class_key, module_key, default_module
            )
            
            if cls is None:
                return None
            
            # 准备参数
            params = copy.deepcopy(config)
            # 移除类和模块信息
            params.pop(class_key, None)
            params.pop(module_key, None)
            # 添加额外参数
            params.update(kwargs)
            
            # 过滤参数
            signature = inspect.signature(cls.__init__)
            valid_params = {}
            for key, value in params.items():
                if key in signature.parameters:
                    valid_params[key] = value
            
            # 实例化
            instance = cls(**valid_params)
            
            logger.info(f"成功实例化对象: {cls.__name__}")
            return instance
            
        except Exception as e:
            logger.error(f"实例化对象失败: {e}")
            return None
    
    @staticmethod
    def get_config_schema(cls: Any) -> Dict[str, Any]:
        """
        从类获取配置模式
        
        Args:
            cls: 类对象
            
        Returns:
            配置模式
        """
        schema = {}
        
        try:
            # 获取__init__方法的签名
            signature = inspect.signature(cls.__init__)
            
            # 遍历参数
            for name, param in list(signature.parameters.items())[1:]:  # 跳过self
                param_schema = {
                    'type': param.annotation if param.annotation != inspect.Parameter.empty else Any,
                    'required': param.default == inspect.Parameter.empty,
                }
                
                if param.default != inspect.Parameter.empty:
                    param_schema['default'] = param.default
                
                # 获取参数文档
                doc = cls.__init__.__doc__ or ''
                if doc:
                    # 简单的文档解析
                    lines = doc.split('\n')
                    for line in lines:
                        line = line.strip()
                        if line.startswith(name + ':'):
                            param_schema['help'] = line[len(name) + 1:].strip()
                            break
                
                schema[name] = param_schema
            
            return schema
            
        except Exception as e:
            logger.error(f"获取配置模式失败: {e}")
            return {}
    
    @staticmethod
    def export_config(config: Dict[str, Any],
                     output_format: str = 'dict',
                     exclude_keys: Optional[List[str]] = None) -> Union[Dict[str, Any], str]:
        """
        导出配置
        
        Args:
            config: 配置字典
            output_format: 输出格式 (dict/json/yaml)
            exclude_keys: 排除的键列表
            
        Returns:
            导出的配置
        """
        try:
            # 复制配置
            export_config = copy.deepcopy(config)
            
            # 排除键
            if exclude_keys:
                for key in exclude_keys:
                    if '.' in key:
                        # 处理嵌套键
                        parts = key.split('.')
                        current = export_config
                        for part in parts[:-1]:
                            if part in current:
                                current = current[part]
                            else:
                                break
                        else:
                            current.pop(parts[-1], None)
                    else:
                        export_config.pop(key, None)
            
            # 格式化输出
            if output_format == 'json':
                return json.dumps(export_config, ensure_ascii=False, indent=2)
            elif output_format == 'yaml':
                return yaml.dump(export_config, allow_unicode=True, default_flow_style=False, sort_keys=False)
            else:  # dict
                return export_config
                
        except Exception as e:
            logger.error(f"导出配置失败: {e}")
            return config if output_format == 'dict' else ''
    
    @staticmethod
    def validate_config_types(config: Dict[str, Any]) -> Dict[str, Any]:
        """
        验证并转换配置类型
        
        Args:
            config: 配置字典
            
        Returns:
            处理后的配置
        """
        try:
            def process_value(value: Any) -> Any:
                """处理单个值"""
                if isinstance(value, str):
                    # 尝试转换字符串为其他类型
                    # 布尔值
                    if value.lower() in ['true', 'yes', 'y', '1']:
                        return True
                    elif value.lower() in ['false', 'no', 'n', '0']:
                        return False
                    # 整数
                    try:
                        return int(value)
                    except ValueError:
                        pass
                    # 浮点数
                    try:
                        return float(value)
                    except ValueError:
                        pass
                    # 列表（如果格式为 '[a,b,c]'）
                    if value.startswith('[') and value.endswith(']'):
                        try:
                            return json.loads(value)
                        except json.JSONDecodeError:
                            pass
                    # 字典（如果格式为 '{...}'）
                    if value.startswith('{') and value.endswith('}'):
                        try:
                            return json.loads(value)
                        except json.JSONDecodeError:
                            pass
                
                return value
            
            def process_config(config_section: Dict[str, Any]) -> Dict[str, Any]:
                """递归处理配置"""
                result = {}
                
                for key, value in config_section.items():
                    if isinstance(value, dict):
                        result[key] = process_config(value)
                    elif isinstance(value, list):
                        result[key] = [process_value(item) for item in value]
                    else:
                        result[key] = process_value(value)
                
                return result
            
            return process_config(config)
            
        except Exception as e:
            logger.error(f"验证配置类型失败: {e}")
            return config
    
    @staticmethod
    def get_nested_value(config: Dict[str, Any],
                        keys_path: str,
                        default: Any = None) -> Any:
        """
        获取嵌套配置值
        
        Args:
            config: 配置字典
            keys_path: 键路径，如 'model.parameters.hidden_size'
            default: 默认值
            
        Returns:
            配置值
        """
        try:
            keys = keys_path.split('.')
            current = config
            
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return default
            
            return current
            
        except Exception as e:
            logger.error(f"获取嵌套配置值失败: {e}")
            return default
    
    @staticmethod
    def set_nested_value(config: Dict[str, Any],
                        keys_path: str,
                        value: Any) -> bool:
        """
        设置嵌套配置值
        
        Args:
            config: 配置字典
            keys_path: 键路径
            value: 新值
            
        Returns:
            是否设置成功
        """
        try:
            keys = keys_path.split('.')
            current = config
            
            # 遍历到倒数第二个键
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                elif not isinstance(current[key], dict):
                    logger.error(f"路径 {'.'.join(keys[:-1])} 不是字典")
                    return False
                current = current[key]
            
            # 设置值
            current[keys[-1]] = value
            
            return True
            
        except Exception as e:
            logger.error(f"设置嵌套配置值失败: {e}")
            return False

# 默认配置模式
def get_default_tamma_config() -> Dict[str, Any]:
    """
    获取TAMMA算法的默认配置模式
    
    Returns:
        配置模式
    """
    return {
        'feature_extraction': {
            'color': {
                'enabled': {'type': bool, 'default': True, 'help': '是否启用颜色特征'},
                'color_space': {'type': str, 'default': 'hsv', 'enum': ['hsv', 'rgb', 'lab', 'ycrcb'], 'help': '颜色空间'},
                'bins': {'type': int, 'default': 8, 'min': 2, 'max': 64, 'help': '直方图分箱数'},
                'spatial_pyramid_levels': {'type': int, 'default': 2, 'min': 1, 'max': 4, 'help': '空间金字塔层级'}
            },
            'texture': {
                'enabled': {'type': bool, 'default': True, 'help': '是否启用纹理特征'},
                'texture_type': {'type': str, 'default': 'lbp', 'enum': ['lbp', 'glcm', 'gabor', 'hog'], 'help': '纹理类型'},
                'grid_size': {'type': int, 'default': 8, 'min': 2, 'max': 32, 'help': '网格大小'}
            },
            'text': {
                'enabled': {'type': bool, 'default': True, 'help': '是否启用文字特征'},
                'use_ocr': {'type': bool, 'default': True, 'help': '是否使用OCR'},
                'text_embedding': {'type': str, 'default': 'tfidf', 'enum': ['tfidf', 'keyword'], 'help': '文本嵌入方法'}
            },
            'sift': {
                'enabled': {'type': bool, 'default': True, 'help': '是否启用SIFT特征'},
                'n_features': {'type': int, 'default': 200, 'min': 10, 'max': 1000, 'help': 'SIFT特征点数量'},
                'codebook_size': {'type': int, 'default': 1024, 'min': 128, 'max': 4096, 'help': '词袋码本大小'}
            }
        },
        'matching': {
            'color_threshold': {'type': float, 'default': 0.5, 'min': 0.0, 'max': 1.0, 'help': '颜色匹配阈值'},
            'spatial_threshold': {'type': float, 'default': 0.3, 'min': 0.0, 'max': 1.0, 'help': '空间约束阈值'},
            'final_threshold': {'type': float, 'default': 0.6, 'min': 0.0, 'max': 1.0, 'help': '最终匹配阈值'},
            'k_neighbors': {'type': int, 'default': 100, 'min': 10, 'max': 500, 'help': 'K近邻数量'}
        },
        'fusion': {
            'method': {'type': str, 'default': 'weighted_sum', 'enum': ['weighted_sum', 'max', 'min', 'average'], 'help': '融合方法'},
            'weights': {
                'color': {'type': float, 'default': 0.3, 'min': 0.0, 'max': 1.0, 'help': '颜色权重'},
                'texture': {'type': float, 'default': 0.2, 'min': 0.0, 'max': 1.0, 'help': '纹理权重'},
                'text': {'type': float, 'default': 0.3, 'min': 0.0, 'max': 1.0, 'help': '文字权重'},
                'sift': {'type': float, 'default': 0.2, 'min': 0.0, 'max': 1.0, 'help': 'SIFT权重'}
            },
            'category_weights': {
                'book': {'type': dict, 'help': '书籍类别的权重配置'},
                'wallet': {'type': dict, 'help': '钱包类别的权重配置'},
                'cup': {'type': dict, 'help': '水杯类别的权重配置'},
                'phone': {'type': dict, 'help': '手机类别的权重配置'},
                'bag': {'type': dict, 'help': '背包类别的权重配置'},
                'clothes': {'type': dict, 'help': '衣物类别的权重配置'},
                'shoes': {'type': dict, 'help': '鞋子类别的权重配置'},
                'keys': {'type': dict, 'help': '钥匙类别的权重配置'},
                'glasses': {'type': dict, 'help': '眼镜类别的权重配置'}
            }
        },
        'performance': {
            'use_gpu': {'type': bool, 'default': False, 'help': '是否使用GPU'},
            'batch_size': {'type': int, 'default': 32, 'min': 1, 'max': 512, 'help': '批处理大小'},
            'index_type': {'type': str, 'default': 'flat', 'enum': ['flat', 'ivf', 'hnsw'], 'help': '索引类型'}
        },
        'evaluation': {
            'k_values': {'type': list, 'default': [1, 5, 10], 'help': 'Top-K值'},
            'metrics': {'type': list, 'default': ['accuracy', 'mrr', 'map'], 'help': '评估指标'}
        }
    }

# 示例用法
if __name__ == '__main__':
    # 示例：加载配置
    # config = ConfigManager.load_config('config.yaml')
    # 
    # # 验证配置
    # schema = get_default_tamma_config()
    # valid, errors = ConfigManager.validate_config(config, schema)
    # 
    # # 保存配置
    # ConfigManager.save_config(config, 'output_config.json')
    # 
    # # 从命令行解析参数
    # config = ConfigManager.parse_arguments(schema)
    # 
    # # 实例化对象
    # from feature_extraction.color_extractor import ColorFeatureExtractor
    # color_extractor = ConfigManager.instantiate_from_config(
    #     config['feature_extraction']['color'],
    #     class_key='class',
    #     module_key='module',
    #     default_module='feature_extraction.color_extractor'
    # )
    # 
    logger.info("配置管理工具类已加载")