import os
import random
import json
import numpy as np
import cv2
import logging
from typing import List, Dict, Tuple, Optional, Any
from PIL import Image, ImageDraw, ImageFont, ImageFilter

logger = logging.getLogger(__name__)

class LostFoundDatasetGeneratorComplete:
    """
    失物招领数据集生成器
    支持合成数据生成、真实数据加载、数据增强和多样化场景模拟
    """
    
    # 物品类别
    ITEM_CLASSES = [
        'book', 'wallet', 'phone', 'keys', 'cup', 
        'umbrella', 'bag', 'glasses', 'clothes', 'document'
    ]
    
    # 背景类型
    BACKGROUND_TYPES = [
        'indoor', 'outdoor', 'office', 'classroom', 
        'cafeteria', 'library', 'park', 'street'
    ]
    
    # 光照条件
    LIGHTING_CONDITIONS = [
        'normal', 'low', 'high', 'backlit', 'shadowed'
    ]
    
    # 物品状态
    ITEM_STATES = ['normal', 'damaged', 'dirty', 'partially_covered']
    
    def __init__(self, 
                 output_dir: str = './data',
                 image_size: Tuple[int, int] = (256, 256),
                 max_objects_per_image: int = 3,
                 random_seed: Optional[int] = None):
        """
        Args:
            output_dir: 输出目录
            image_size: 图像大小
            max_objects_per_image: 每张图像最多物品数量
            random_seed: 随机种子
        """
        self.output_dir = output_dir
        self.image_size = image_size
        self.max_objects_per_image = max_objects_per_image
        
        # 创建目录
        self.synthetic_dir = os.path.join(output_dir, 'synthetic')
        self.real_dir = os.path.join(output_dir, 'real')
        self.merged_dir = os.path.join(output_dir, 'merged')
        
        for dir_path in [self.synthetic_dir, self.real_dir, self.merged_dir]:
            os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)
            os.makedirs(os.path.join(dir_path, 'annotations'), exist_ok=True)
        
        # 设置随机种子
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
        
        logger.info(f"初始化数据集生成器: output_dir={output_dir}")
        logger.info(f"支持的物品类别: {', '.join(self.ITEM_CLASSES)}")
    
    def generate_synthetic_dataset(self, 
                                  num_samples: int,
                                  output_json: bool = True,
                                  output_dir: Optional[str] = None) -> str:
        """
        生成合成数据集
        
        Args:
            num_samples: 生成样本数量
            output_json: 是否输出JSON格式
            output_dir: 输出目录
            
        Returns:
            数据集目录路径
        """
        logger.info(f"开始生成合成数据集: {num_samples}个样本")
        
        # 设置输出目录
        dataset_dir = output_dir or self.synthetic_dir
        images_dir = os.path.join(dataset_dir, 'images')
        annotations_dir = os.path.join(dataset_dir, 'annotations')
        
        # 生成样本
        all_annotations = []
        
        for i in range(num_samples):
            try:
                # 生成图像和标注
                image_path, annotation = self._generate_single_synthetic_image(
                    i,
                    images_dir,
                    annotations_dir
                )
                
                if image_path and annotation:
                    all_annotations.append(annotation)
                    
                    if (i + 1) % 100 == 0:
                        logger.info(f"已生成 {i + 1}/{num_samples} 个合成样本")
                        
            except Exception as e:
                logger.error(f"生成样本 {i} 失败: {e}")
                continue
        
        # 输出JSON格式
        if output_json:
            json_path = os.path.join(dataset_dir, 'annotations.json')
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(all_annotations, f, ensure_ascii=False, indent=2)
            logger.info(f"JSON标注已保存到: {json_path}")
        
        logger.info(f"合成数据集生成完成，共 {len(all_annotations)} 个样本")
        return dataset_dir
    
    def _generate_single_synthetic_image(self, 
                                        sample_id: int,
                                        images_dir: str,
                                        annotations_dir: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """
        生成单个合成图像
        
        Args:
            sample_id: 样本ID
            images_dir: 图像目录
            annotations_dir: 标注目录
            
        Returns:
            (图像路径, 标注信息)
        """
        # 创建空白图像
        width, height = self.image_size
        image = Image.new('RGB', self.image_size, color='white')
        draw = ImageDraw.Draw(image)
        
        # 选择背景
        background_type = random.choice(self.BACKGROUND_TYPES)
        self._draw_background(image, draw, background_type)
        
        # 选择光照条件
        lighting = random.choice(self.LIGHTING_CONDITIONS)
        
        # 随机决定物品数量
        num_objects = random.randint(1, self.max_objects_per_image)
        
        # 物品信息列表
        objects = []
        
        # 记录已放置的区域，避免重叠
        placed_regions = []
        
        for obj_idx in range(num_objects):
            # 选择物品类别
            item_class = random.choice(self.ITEM_CLASSES)
            
            # 选择物品状态
            item_state = random.choice(self.ITEM_STATES)
            
            # 生成物品图像
            obj_width = random.randint(30, min(width // 3, 100))
            obj_height = random.randint(30, min(height // 3, 100))
            
            # 尝试放置物品，避免重叠
            placed = False
            for _ in range(10):  # 尝试10次
                x = random.randint(10, width - obj_width - 10)
                y = random.randint(10, height - obj_height - 10)
                
                # 检查重叠
                overlap = False
                new_region = (x, y, x + obj_width, y + obj_height)
                
                for region in placed_regions:
                    if self._check_overlap(new_region, region):
                        overlap = True
                        break
                
                if not overlap:
                    placed_regions.append(new_region)
                    placed = True
                    break
            
            if not placed:
                continue
            
            # 绘制物品
            self._draw_item(draw, item_class, x, y, obj_width, obj_height, item_state)
            
            # 添加噪声和变换
            self._apply_noise(draw, x, y, obj_width, obj_height, item_state)
            
            # 记录物品信息
            obj_info = {
                'class': item_class,
                'state': item_state,
                'bbox': [x, y, obj_width, obj_height],
                'id': f'obj_{sample_id}_{obj_idx}'
            }
            
            # 生成文字描述
            obj_info['description'] = self._generate_item_description(item_class, item_state)
            
            # 生成特征描述
            obj_info['features'] = self._generate_item_features(item_class)
            
            objects.append(obj_info)
        
        # 应用光照效果
        if lighting != 'normal':
            image = self._apply_lighting(image, lighting)
        
        # 添加全局噪声
        image = self._add_global_noise(image)
        
        # 保存图像
        image_filename = f'sample_{sample_id:06d}.jpg'
        image_path = os.path.join(images_dir, image_filename)
        image.save(image_path, 'JPEG', quality=95)
        
        # 创建标注信息
        annotation = {
            'id': sample_id,
            'image_path': image_filename,
            'width': width,
            'height': height,
            'background_type': background_type,
            'lighting': lighting,
            'objects': objects,
            'metadata': {
                'generator': 'LostFoundDatasetGenerator',
                'version': '1.0'
            }
        }
        
        # 保存单独的标注文件
        annotation_filename = f'sample_{sample_id:06d}.json'
        annotation_path = os.path.join(annotations_dir, annotation_filename)
        with open(annotation_path, 'w', encoding='utf-8') as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)
        
        return image_path, annotation
    
    def _draw_background(self, 
                        image: Image.Image,
                        draw: ImageDraw.ImageDraw,
                        background_type: str):
        """
        绘制背景
        """
        width, height = image.size
        
        if background_type == 'indoor':
            # 室内背景 - 浅色调
            color = (random.randint(200, 255), 
                     random.randint(200, 255), 
                     random.randint(200, 255))
            draw.rectangle([(0, 0), (width, height)], fill=color)
        
        elif background_type == 'outdoor':
            # 室外背景 - 蓝天绿地
            sky_color = (random.randint(150, 200), 
                         random.randint(180, 220), 
                         random.randint(220, 255))
            ground_color = (random.randint(50, 100), 
                           random.randint(150, 200), 
                           random.randint(50, 100))
            
            draw.rectangle([(0, 0), (width, height//2)], fill=sky_color)
            draw.rectangle([(0, height//2), (width, height)], fill=ground_color)
        
        elif background_type == 'office':
            # 办公室背景
            desk_color = (random.randint(100, 150), 
                         random.randint(80, 130), 
                         random.randint(50, 100))
            wall_color = (random.randint(220, 240), 
                         random.randint(220, 240), 
                         random.randint(220, 240))
            
            draw.rectangle([(0, 0), (width, height//3)], fill=wall_color)
            draw.rectangle([(0, height//3), (width, height)], fill=desk_color)
        
        elif background_type == 'classroom':
            # 教室背景
            color = (random.randint(230, 250), 
                     random.randint(230, 250), 
                     random.randint(230, 250))
            draw.rectangle([(0, 0), (width, height)], fill=color)
            
            # 添加黑板
            draw.rectangle([(width//4, height//4), (width*3//4, height*3//4)], 
                          fill=(random.randint(30, 80), random.randint(30, 80), random.randint(30, 80)))
        
        else:
            # 默认背景
            color = (random.randint(200, 255), 
                     random.randint(200, 255), 
                     random.randint(200, 255))
            draw.rectangle([(0, 0), (width, height)], fill=color)
    
    def _draw_item(self, 
                  draw: ImageDraw.ImageDraw,
                  item_class: str,
                  x: int,
                  y: int,
                  width: int,
                  height: int,
                  item_state: str):
        """
        绘制物品
        """
        # 选择颜色
        color = self._get_random_color(item_class)
        
        if item_class == 'book':
            # 绘制书本
            draw.rectangle([(x, y), (x + width, y + height)], fill=color)
            
            # 添加书页
            page_width = width // 10
            draw.rectangle([(x, y), (x + page_width, y + height)], 
                          fill=(random.randint(200, 255), random.randint(200, 255), random.randint(180, 220)))
            
            # 添加标题
            title_height = height // 5
            draw.rectangle([(x + page_width, y + height//4), 
                           (x + width - 10, y + height//4 + title_height)], 
                          fill=(min(color[0]-30, 255), min(color[1]-30, 255), min(color[2]-30, 255)))
        
        elif item_class == 'wallet':
            # 绘制钱包
            if random.random() > 0.5:  # 横版钱包
                draw.rectangle([(x, y), (x + width, y + height)], fill=color)
                # 添加钱包边缘
                draw.rectangle([(x, y), (x + width, y + 5)], 
                              fill=(min(color[0]-20, 255), min(color[1]-20, 255), min(color[2]-20, 255)))
            else:  # 竖版钱包
                draw.rectangle([(x, y), (x + width*2//3, y + height)], fill=color)
        
        elif item_class == 'phone':
            # 绘制手机
            phone_width = width
            phone_height = height
            
            # 手机主体
            draw.rectangle([(x, y), (x + phone_width, y + phone_height)], fill=color)
            
            # 屏幕
            screen_width = phone_width - 10
            screen_height = phone_height - 20
            screen_x = x + 5
            screen_y = y + 10
            
            screen_color = (random.randint(200, 230), random.randint(200, 230), random.randint(200, 230))
            draw.rectangle([(screen_x, screen_y), (screen_x + screen_width, screen_y + screen_height)], 
                          fill=screen_color)
            
            # 摄像头
            cam_size = 5
            draw.ellipse([(x + phone_width//2 - cam_size//2, y + 5), 
                         (x + phone_width//2 + cam_size//2, y + 5 + cam_size)], 
                        fill=(0, 0, 0))
        
        elif item_class == 'keys':
            # 绘制钥匙串
            key_width = width // 3
            key_height = height // 2
            
            # 钥匙环
            ring_radius = 10
            ring_x = x + key_width
            ring_y = y + ring_radius
            draw.ellipse([(ring_x - ring_radius, ring_y - ring_radius), 
                         (ring_x + ring_radius, ring_y + ring_radius)], 
                        fill=(200, 200, 200), outline=(150, 150, 150), width=2)
            
            # 绘制2-3把钥匙
            num_keys = random.randint(2, 3)
            for i in range(num_keys):
                kx = ring_x + ring_radius
                ky = ring_y + i * key_height
                
                # 钥匙柄
                draw.rectangle([(kx, ky), (kx + key_width, ky + key_height//2)], 
                              fill=color)
                
                # 钥匙齿
                teeth_height = key_height // 4
                for j in range(3):
                    tooth_width = key_width // 4
                    if random.random() > 0.3:
                        draw.rectangle([(kx + key_width, ky + j * teeth_height * 1.5), 
                                       (kx + key_width + tooth_width, ky + j * teeth_height * 1.5 + teeth_height)], 
                                      fill=color)
        
        elif item_class == 'cup':
            # 绘制杯子
            cup_width = width // 2
            cup_height = height
            
            # 杯子主体
            draw.ellipse([(x + cup_width//2, y + cup_height - 10), 
                         (x + cup_width*3//2, y + cup_height)], 
                        fill=color)
            
            # 杯身
            draw.rectangle([(x + cup_width//2, y), 
                           (x + cup_width*3//2, y + cup_height)], 
                          fill=color)
            
            # 杯柄
            handle_x = x + cup_width*3//2
            handle_y = y + cup_height//4
            handle_radius = cup_width // 4
            draw.ellipse([(handle_x, handle_y), 
                         (handle_x + handle_radius*2, handle_y + handle_radius*2)], 
                        fill=(255, 255, 255))
            draw.ellipse([(handle_x + handle_radius//2, handle_y + handle_radius//2), 
                         (handle_x + handle_radius*1.5, handle_y + handle_radius*1.5)], 
                        fill=color)
        
        elif item_class == 'umbrella':
            # 绘制雨伞
            canopy_radius = min(width, height) // 2
            handle_height = height // 2
            
            # 伞面
            draw.ellipse([(x + width//2 - canopy_radius, y), 
                         (x + width//2 + canopy_radius, y + canopy_radius*2)], 
                        fill=color)
            
            # 伞柄
            handle_x = x + width//2
            handle_y = y + canopy_radius
            draw.line([(handle_x, handle_y), (handle_x, handle_y + handle_height)], 
                     fill=(100, 100, 100), width=3)
        
        elif item_class == 'bag':
            # 绘制包
            bag_width = width
            bag_height = height
            
            # 包主体
            draw.rectangle([(x, y), (x + bag_width, y + bag_height)], fill=color)
            
            # 包带
            strap_width = width // 10
            draw.rectangle([(x + bag_width//4, y), 
                           (x + bag_width//4 + strap_width, y + bag_height//2)], 
                          fill=(min(color[0]-30, 255), min(color[1]-30, 255), min(color[2]-30, 255)))
            
            # 包扣
            button_size = strap_width
            draw.ellipse([(x + bag_width//4, y + bag_height//4 - button_size//2), 
                         (x + bag_width//4 + button_size, y + bag_height//4 + button_size//2)], 
                        fill=(200, 200, 200))
        
        elif item_class == 'glasses':
            # 绘制眼镜
            lens_radius = min(width, height) // 4
            bridge_width = lens_radius
            
            # 左镜片
            left_x = x + lens_radius
            left_y = y + height//2
            draw.ellipse([(left_x - lens_radius, left_y - lens_radius), 
                         (left_x + lens_radius, left_y + lens_radius)], 
                        fill=(200, 220, 240))
            
            # 右镜片
            right_x = left_x + lens_radius*2 + bridge_width
            draw.ellipse([(right_x - lens_radius, left_y - lens_radius), 
                         (right_x + lens_radius, left_y + lens_radius)], 
                        fill=(200, 220, 240))
            
            # 鼻梁架
            draw.line([(left_x + lens_radius, left_y), (right_x - lens_radius, left_y)], 
                     fill=color, width=3)
            
            # 镜腿
            leg_length = lens_radius * 1.5
            draw.line([(left_x - lens_radius, left_y), 
                      (left_x - lens_radius - leg_length, left_y - leg_length//2)], 
                     fill=color, width=2)
            draw.line([(right_x + lens_radius, left_y), 
                      (right_x + lens_radius + leg_length, left_y - leg_length//2)], 
                     fill=color, width=2)
        
        elif item_class == 'clothes':
            # 绘制衣服
            self._draw_clothes(draw, x, y, width, height, color, item_class)
        
        elif item_class == 'document':
            # 绘制文档
            draw.rectangle([(x, y), (x + width, y + height)], 
                          fill=(random.randint(220, 255), random.randint(220, 255), random.randint(220, 255)))
            
            # 添加文字线条
            line_spacing = height // 8
            line_color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
            for i in range(3, 8):
                line_y = y + i * line_spacing
                draw.line([(x + 10, line_y), (x + width - 10, line_y)], 
                         fill=line_color, width=1)
        
        # 应用物品状态效果
        if item_state == 'damaged':
            self._apply_damage_effect(draw, x, y, width, height)
        elif item_state == 'dirty':
            self._apply_dirty_effect(draw, x, y, width, height)
        elif item_state == 'partially_covered':
            self._apply_covering_effect(draw, x, y, width, height)
    
    def _draw_clothes(self, 
                     draw: ImageDraw.ImageDraw,
                     x: int,
                     y: int,
                     width: int,
                     height: int,
                     color: Tuple[int, int, int],
                     item_type: str):
        """
        绘制衣物
        """
        if random.random() > 0.5:
            # T恤样式
            # 领口
            neck_y = y + height // 5
            draw.ellipse([(x + width//4, neck_y - height//10), 
                         (x + width*3//4, neck_y)], 
                        fill=color)
            
            # 主体
            draw.polygon([
                (x, neck_y),
                (x + width//4, y + height),
                (x + width*3//4, y + height),
                (x + width, neck_y)
            ], fill=color)
            
            # 袖子
            sleeve_length = height // 3
            draw.polygon([
                (x, neck_y),
                (x - sleeve_length, neck_y + sleeve_length),
                (x - sleeve_length + width//5, neck_y + sleeve_length*2),
                (x + width//10, neck_y + height//4)
            ], fill=color)
            
            draw.polygon([
                (x + width, neck_y),
                (x + width + sleeve_length, neck_y + sleeve_length),
                (x + width + sleeve_length - width//5, neck_y + sleeve_length*2),
                (x + width - width//10, neck_y + height//4)
            ], fill=color)
        else:
            # 衬衫样式
            draw.rectangle([(x, y), (x + width, y + height)], fill=color)
            
            # 领子
            collar_height = height // 8
            draw.polygon([
                (x + width//4, y),
                (x + width//2, y + collar_height),
                (x + width//2, y)
            ], fill=(min(color[0]-20, 255), min(color[1]-20, 255), min(color[2]-20, 255)))
            
            draw.polygon([
                (x + width*3//4, y),
                (x + width//2, y + collar_height),
                (x + width//2, y)
            ], fill=(min(color[0]-20, 255), min(color[1]-20, 255), min(color[2]-20, 255)))
            
            # 纽扣
            for i in range(3):
                button_y = y + collar_height + i * height // 6
                draw.ellipse([
                    (x + width//2 - 3, button_y - 3),
                    (x + width//2 + 3, button_y + 3)
                ], fill=(200, 200, 200))
    
    def _draw_zipper(self, 
                    draw: ImageDraw.ImageDraw,
                    x: int,
                    y: int,
                    length: int,
                    color: Tuple[int, int, int]):
        """
        绘制拉链
        """
        # 拉链线
        draw.line([(x, y), (x, y + length)], fill=color, width=2)
        
        # 拉链齿
        tooth_size = 3
        for i in range(0, length, tooth_size*2):
            # 左齿
            draw.rectangle([
                (x - tooth_size, y + i),
                (x, y + i + tooth_size)
            ], fill=color)
            
            # 右齿
            draw.rectangle([
                (x, y + i + tooth_size),
                (x + tooth_size, y + i + tooth_size*2)
            ], fill=color)
        
        # 拉链头
        slider_height = tooth_size * 4
        draw.rectangle([
            (x - tooth_size*2, y + length//2 - slider_height//2),
            (x + tooth_size*2, y + length//2 + slider_height//2)
        ], fill=(min(color[0]-30, 255), min(color[1]-30, 255), min(color[2]-30, 255)))
    
    def _draw_strap(self, 
                   draw: ImageDraw.ImageDraw,
                   start_x: int,
                   start_y: int,
                   end_x: int,
                   end_y: int,
                   width: int,
                   color: Tuple[int, int, int]):
        """
        绘制肩带或背带
        """
        draw.line([(start_x, start_y), (end_x, end_y)], fill=color, width=width)
        
        # 添加细节
        detail_spacing = 10
        steps = max(abs(end_x - start_x), abs(end_y - start_y)) // detail_spacing
        
        if steps > 0:
            dx = (end_x - start_x) / steps
            dy = (end_y - start_y) / steps
            
            for i in range(steps + 1):
                x = start_x + dx * i
                y = start_y + dy * i
                draw.ellipse([
                    (x - 2, y - 2),
                    (x + 2, y + 2)
                ], fill=(min(color[0]-20, 255), min(color[1]-20, 255), min(color[2]-20, 255)))
    
    def _get_random_color(self, item_class: str) -> Tuple[int, int, int]:
        """
        根据物品类别获取随机颜色
        """
        if item_class == 'book':
            return (random.randint(100, 200), random.randint(50, 150), random.randint(50, 150))
        elif item_class == 'wallet':
            return (random.randint(50, 150), random.randint(50, 150), random.randint(50, 150))
        elif item_class == 'phone':
            return (random.randint(0, 50), random.randint(0, 50), random.randint(0, 50))
        elif item_class == 'keys':
            return (random.randint(150, 200), random.randint(150, 200), random.randint(150, 200))
        elif item_class == 'cup':
            return (random.randint(100, 200), random.randint(100, 200), random.randint(150, 250))
        elif item_class == 'umbrella':
            return (random.randint(150, 250), random.randint(50, 150), random.randint(50, 150))
        elif item_class == 'bag':
            return (random.randint(50, 150), random.randint(50, 150), random.randint(50, 150))
        elif item_class == 'glasses':
            return (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
        elif item_class == 'clothes':
            return (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
        elif item_class == 'document':
            return (random.randint(220, 255), random.randint(220, 255), random.randint(220, 255))
        else:
            return (random.randint(100, 200), random.randint(100, 200), random.randint(100, 200))
    
    def _apply_noise(self, 
                    draw: ImageDraw.ImageDraw,
                    x: int,
                    y: int,
                    width: int,
                    height: int,
                    item_state: str):
        """
        应用噪声效果
        """
        if item_state == 'normal':
            return
        
        # 添加噪点
        noise_level = random.randint(10, 30)
        for _ in range(noise_level):
            nx = random.randint(x, x + width)
            ny = random.randint(y, y + height)
            size = random.randint(1, 3)
            noise_color = (random.randint(0, 100), random.randint(0, 100), random.randint(0, 100))
            draw.rectangle([(nx, ny), (nx + size, ny + size)], fill=noise_color)
    
    def _apply_damage_effect(self, 
                            draw: ImageDraw.ImageDraw,
                            x: int,
                            y: int,
                            width: int,
                            height: int):
        """
        应用损坏效果
        """
        # 添加裂痕
        num_cracks = random.randint(1, 3)
        
        for _ in range(num_cracks):
            start_x = random.randint(x, x + width)
            start_y = random.randint(y, y + height)
            length = random.randint(width // 4, width // 2)
            angle = random.uniform(0, 2 * np.pi)
            
            end_x = start_x + int(length * np.cos(angle))
            end_y = start_y + int(length * np.sin(angle))
            
            draw.line([(start_x, start_y), (end_x, end_y)], 
                     fill=(0, 0, 0), width=2)
            
            # 添加碎片效果
            for _ in range(5):
                piece_x = random.randint(min(start_x, end_x), max(start_x, end_x))
                piece_y = random.randint(min(start_y, end_y), max(start_y, end_y))
                piece_angle = angle + random.uniform(-np.pi/4, np.pi/4)
                piece_length = random.randint(5, 10)
                
                piece_end_x = piece_x + int(piece_length * np.cos(piece_angle))
                piece_end_y = piece_y + int(piece_length * np.sin(piece_angle))
                
                draw.line([(piece_x, piece_y), (piece_end_x, piece_end_y)], 
                         fill=(0, 0, 0), width=1)
    
    def _apply_dirty_effect(self, 
                           draw: ImageDraw.ImageDraw,
                           x: int,
                           y: int,
                           width: int,
                           height: int):
        """
        应用脏污效果
        """
        # 添加污渍
        num_stains = random.randint(2, 5)
        
        for _ in range(num_stains):
            stain_x = random.randint(x, x + width)
            stain_y = random.randint(y, y + height)
            stain_radius = random.randint(width // 10, width // 5)
            
            # 随机污渍颜色
            stain_color = (random.randint(50, 100), random.randint(50, 100), random.randint(0, 50))
            
            # 绘制不规则污渍
            for _ in range(10):
                offset_x = random.randint(-stain_radius // 2, stain_radius // 2)
                offset_y = random.randint(-stain_radius // 2, stain_radius // 2)
                sub_radius = random.randint(stain_radius // 3, stain_radius)
                
                draw.ellipse([
                    (stain_x + offset_x - sub_radius, stain_y + offset_y - sub_radius),
                    (stain_x + offset_x + sub_radius, stain_y + offset_y + sub_radius)
                ], fill=stain_color, outline=None)
    
    def _apply_covering_effect(self, 
                              draw: ImageDraw.ImageDraw,
                              x: int,
                              y: int,
                              width: int,
                              height: int):
        """
        应用部分遮挡效果
        """
        # 遮挡区域大小
        cover_width = random.randint(width // 4, width // 2)
        cover_height = random.randint(height // 4, height // 2)
        
        # 选择遮挡位置
        cover_position = random.choice(['top', 'bottom', 'left', 'right', 'corner'])
        
        if cover_position == 'top':
            cover_x = x
            cover_y = y - cover_height // 2
        elif cover_position == 'bottom':
            cover_x = x
            cover_y = y + height - cover_height // 2
        elif cover_position == 'left':
            cover_x = x - cover_width // 2
            cover_y = y
        elif cover_position == 'right':
            cover_x = x + width - cover_width // 2
            cover_y = y
        else:  # corner
            corner = random.choice(['top-left', 'top-right', 'bottom-left', 'bottom-right'])
            if corner == 'top-left':
                cover_x = x - cover_width // 2
                cover_y = y - cover_height // 2
            elif corner == 'top-right':
                cover_x = x + width - cover_width // 2
                cover_y = y - cover_height // 2
            elif corner == 'bottom-left':
                cover_x = x - cover_width // 2
                cover_y = y + height - cover_height // 2
            else:
                cover_x = x + width - cover_width // 2
                cover_y = y + height - cover_height // 2
        
        # 遮挡物颜色
        cover_color = (random.randint(150, 200), random.randint(150, 200), random.randint(150, 200))
        
        # 绘制遮挡物
        draw.rectangle([
            (cover_x, cover_y),
            (cover_x + cover_width, cover_y + cover_height)
        ], fill=cover_color)
    
    def _apply_lighting(self, 
                       image: Image.Image,
                       lighting: str) -> Image.Image:
        """
        应用光照效果
        """
        if lighting == 'low':
            # 低光照
            return Image.eval(image, lambda x: max(0, x - 50))
        
        elif lighting == 'high':
            # 高光照
            return Image.eval(image, lambda x: min(255, x + 50))
        
        elif lighting == 'backlit':
            # 逆光效果
            width, height = image.size
            mask = Image.new('L', (width, height), 0)
            mask_draw = ImageDraw.Draw(mask)
            
            # 光晕效果
            for i in range(20):
                opacity = 255 - i * 10
                radius = width // 2 - i * 5
                if radius <= 0:
                    break
                
                mask_draw.ellipse([
                    (width // 2 - radius, height // 2 - radius),
                    (width // 2 + radius, height // 2 + radius)
                ], fill=opacity)
            
            backlit = Image.new('RGB', (width, height), (255, 255, 255))
            return Image.blend(image, backlit, 0.3)
        
        elif lighting == 'shadowed':
            # 阴影效果
            width, height = image.size
            shadow_region = random.choice([
                [(0, 0), (width//2, height)],
                [(width//2, 0), (width, height)],
                [(0, 0), (width, height//2)],
                [(0, height//2), (width, height)]
            ])
            
            shadow = Image.new('RGBA', (width, height), (0, 0, 0, 0))
            shadow_draw = ImageDraw.Draw(shadow)
            shadow_draw.rectangle(shadow_region, fill=(0, 0, 0, 80))
            
            result = image.copy()
            result.paste(shadow, (0, 0), shadow)
            return result
        
        return image
    
    def _add_global_noise(self, image: Image.Image) -> Image.Image:
        """
        添加全局噪声
        """
        # 转换为numpy数组
        img_array = np.array(image)
        
        # 添加高斯噪声
        mean = 0
        sigma = random.randint(1, 5)
        gaussian_noise = np.random.normal(mean, sigma, img_array.shape)
        noisy_image = img_array + gaussian_noise
        noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
        
        # 转回PIL图像
        result = Image.fromarray(noisy_image)
        
        # 轻微模糊
        if random.random() > 0.7:
            result = result.filter(ImageFilter.GaussianBlur(radius=0.5))
        
        return result
    
    def _generate_item_description(self, 
                                 item_class: str,
                                 item_state: str) -> str:
        """
        生成物品描述
        """
        descriptions = {
            'book': [
                '一本{}的书',
                '一个{}的书籍',
                '带有封面的{}书籍'
            ],
            'wallet': [
                '一个{}的钱包',
                '一个{}的皮夹',
                '{}的钱包'
            ],
            'phone': [
                '一部{}的手机',
                '一个{}的智能手机',
                '{}的手机'
            ],
            'keys': [
                '一串{}的钥匙',
                '几把{}的钥匙',
                '{}的钥匙串'
            ],
            'cup': [
                '一个{}的杯子',
                '一个{}的水杯',
                '{}的杯子'
            ],
            'umbrella': [
                '一把{}的雨伞',
                '一个{}的雨伞',
                '{}的雨伞'
            ],
            'bag': [
                '一个{}的包',
                '一个{}的袋子',
                '{}的包'
            ],
            'glasses': [
                '一副{}的眼镜',
                '一个{}的眼镜',
                '{}的眼镜'
            ],
            'clothes': [
                '一件{}的衣服',
                '一个{}的衣物',
                '{}的衣服'
            ],
            'document': [
                '一份{}的文档',
                '一些{}的文件',
                '{}的文档'
            ]
        }
        
        state_adjectives = {
            'normal': ['普通', '常见', '标准'],
            'damaged': ['损坏的', '破损的', '坏了的'],
            'dirty': ['脏的', '污秽的', '有污渍的'],
            'partially_covered': ['部分被遮挡的', '半遮半掩的', '部分可见的']
        }
        
        base_desc = random.choice(descriptions.get(item_class, ['{}物品']))
        adjective = random.choice(state_adjectives.get(item_state, ['']))
        
        return base_desc.format(adjective)
    
    def _generate_item_features(self, item_class: str) -> Dict[str, str]:
        """
        生成物品特征描述
        """
        features = {
            'book': {
                'size': random.choice(['小', '中', '大']),
                'color': random.choice(['红色', '蓝色', '绿色', '黄色', '黑色', '白色']),
                'cover_type': random.choice(['硬皮', '软皮', '精装'])
            },
            'wallet': {
                'size': random.choice(['小', '中']),
                'color': random.choice(['黑色', '棕色', '蓝色', '红色']),
                'material': random.choice(['皮革', '布料', '塑料'])
            },
            'phone': {
                'size': random.choice(['小', '中', '大']),
                'color': random.choice(['黑色', '白色', '银色', '金色', '蓝色']),
                'type': random.choice(['智能手机', '普通手机'])
            },
            'keys': {
                '数量': str(random.randint(2, 5)) + '把',
                'color': random.choice(['银色', '金色', '黑色']),
                'has_keyring': random.choice(['是', '否'])
            },
            'cup': {
                'size': random.choice(['小', '中', '大']),
                'color': random.choice(['红色', '蓝色', '绿色', '黄色', '白色']),
                'has_handle': random.choice(['是', '否'])
            },
            'umbrella': {
                'size': random.choice(['折叠', '标准', '大号']),
                'color': random.choice(['红色', '蓝色', '黑色', '花色']),
                'type': random.choice(['直柄', '折叠'])
            },
            'bag': {
                'size': random.choice(['小', '中', '大']),
                'color': random.choice(['黑色', '棕色', '蓝色', '红色', '白色']),
                'type': random.choice(['背包', '手提包', '单肩包'])
            },
            'glasses': {
                'frame_type': random.choice(['全框', '半框', '无框']),
                'color': random.choice(['黑色', '金色', '银色', '棕色']),
                'lens_type': random.choice(['近视', '远视', '太阳镜'])
            },
            'clothes': {
                'type': random.choice(['T恤', '衬衫', '外套', '裤子']),
                'color': random.choice(['红色', '蓝色', '绿色', '黑色', '白色', '灰色']),
                'size': random.choice(['S', 'M', 'L', 'XL'])
            },
            'document': {
                'type': random.choice(['纸张', '证件', '文件']),
                'color': random.choice(['白色', '黄色', '蓝色']),
                'number_of_pages': str(random.randint(1, 10)) + '页'
            }
        }
        
        return features.get(item_class, {})
    
    def _check_overlap(self, region1: Tuple[int, int, int, int], 
                      region2: Tuple[int, int, int, int]) -> bool:
        """
        检查两个区域是否重叠
        """
        x1, y1, x2, y2 = region1
        a1, b1, a2, b2 = region2
        
        # 如果一个区域的左边界大于另一个区域的右边界，不重叠
        if x1 > a2 or a1 > x2:
            return False
        
        # 如果一个区域的上边界大于另一个区域的下边界，不重叠
        if y1 > b2 or b1 > y2:
            return False
        
        return True
    
    def load_real_dataset(self, 
                         dataset_path: str,
                         output_dir: Optional[str] = None) -> str:
        """
        加载真实数据集
        
        Args:
            dataset_path: 数据集路径
            output_dir: 输出目录
            
        Returns:
            数据集目录路径
        """
        logger.info(f"加载真实数据集: {dataset_path}")
        
        # 设置输出目录
        dataset_dir = output_dir or self.real_dir
        
        # 这里可以实现真实数据集的加载逻辑
        # 目前只是创建一个示例
        
        # 创建示例JSON文件
        example_data = {
            'dataset_info': {
                'name': 'Real Lost and Found Dataset',
                'description': '真实失物招领数据集示例',
                'total_samples': 0,
                'classes': self.ITEM_CLASSES
            }
        }
        
        json_path = os.path.join(dataset_dir, 'dataset_info.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(example_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"真实数据集信息已保存到: {json_path}")
        return dataset_dir
    
    def merge_datasets(self, 
                      synthetic_dir: Optional[str] = None,
                      real_dir: Optional[str] = None,
                      output_dir: Optional[str] = None,
                      synthetic_ratio: float = 0.5) -> str:
        """
        合并合成数据集和真实数据集
        
        Args:
            synthetic_dir: 合成数据集目录
            real_dir: 真实数据集目录
            output_dir: 输出目录
            synthetic_ratio: 合成数据比例
            
        Returns:
            合并后数据集目录路径
        """
        logger.info("合并数据集")
        
        # 设置目录
        synthetic_dir = synthetic_dir or self.synthetic_dir
        real_dir = real_dir or self.real_dir
        merged_dir = output_dir or self.merged_dir
        
        # 创建合并数据集信息
        merged_info = {
            'dataset_info': {
                'name': 'Merged Lost and Found Dataset',
                'description': '合并的失物招领数据集',
                'synthetic_ratio': synthetic_ratio,
                'source_datasets': [
                    {'type': 'synthetic', 'path': synthetic_dir},
                    {'type': 'real', 'path': real_dir}
                ]
            }
        }
        
        json_path = os.path.join(merged_dir, 'dataset_info.json')
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(merged_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"合并数据集信息已保存到: {json_path}")
        return merged_dir
    
    def generate_texture(self, 
                        width: int,
                        height: int,
                        texture_type: str = 'random') -> np.ndarray:
        """
        生成纹理图像
        
        Args:
            width: 宽度
            height: 高度
            texture_type: 纹理类型
            
        Returns:
            纹理图像数组
        """
        if texture_type == 'wood':
            # 木纹纹理
            x, y = np.meshgrid(np.linspace(0, 1, width), np.linspace(0, 1, height))
            noise = np.random.normal(0, 0.1, (height, width))
            wood = np.sin(x * 10 + noise) * 0.5 + 0.5
            wood = np.dstack([wood * 180 + 75, wood * 100 + 50, wood * 30 + 20])
            return np.clip(wood, 0, 255).astype(np.uint8)
        
        elif texture_type == 'marble':
            # 大理石纹理
            x, y = np.meshgrid(np.linspace(0, 10, width), np.linspace(0, 10, height))
            noise = np.random.normal(0, 0.5, (height, width))
            marble = np.sin(x + noise) * np.cos(y + noise) * 0.5 + 0.5
            marble = np.dstack([marble * 50 + 200, marble * 50 + 200, marble * 50 + 200])
            return np.clip(marble, 0, 255).astype(np.uint8)
        
        elif texture_type == 'fabric':
            # 布料纹理
            texture = np.random.normal(128, 30, (height, width, 3))
            return np.clip(texture, 0, 255).astype(np.uint8)
        
        else:
            # 随机纹理
            return np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    
    def apply_transforms(self, 
                        image: Image.Image,
                        transforms: List[str]) -> Image.Image:
        """
        应用图像变换
        
        Args:
            image: 输入图像
            transforms: 变换列表
            
        Returns:
            变换后的图像
        """
        result = image.copy()
        
        for transform in transforms:
            if transform == 'rotate':
                angle = random.randint(-30, 30)
                result = result.rotate(angle, expand=True)
            
            elif transform == 'scale':
                scale_factor = random.uniform(0.8, 1.2)
                new_width = int(result.width * scale_factor)
                new_height = int(result.height * scale_factor)
                result = result.resize((new_width, new_height))
            
            elif transform == 'flip':
                if random.random() > 0.5:
                    result = result.transpose(Image.FLIP_LEFT_RIGHT)
                else:
                    result = result.transpose(Image.FLIP_TOP_BOTTOM)
            
            elif transform == 'crop':
                crop_size = random.randint(10, min(result.width, result.height) // 4)
                left = random.randint(0, crop_size)
                top = random.randint(0, crop_size)
                right = result.width - random.randint(0, crop_size)
                bottom = result.height - random.randint(0, crop_size)
                result = result.crop((left, top, right, bottom))
            
            elif transform == 'blur':
                radius = random.uniform(0.5, 2.0)
                result = result.filter(ImageFilter.GaussianBlur(radius=radius))
        
        # 调整回原始大小
        if result.size != image.size:
            result = result.resize(image.size)
        
        return result
    
    def save_dataset_info(self, 
                         dataset_dir: str,
                         dataset_type: str,
                         num_samples: int,
                         **kwargs) -> str:
        """
        保存数据集信息
        
        Args:
            dataset_dir: 数据集目录
            dataset_type: 数据集类型
            num_samples: 样本数量
            
        Returns:
            信息文件路径
        """
        info = {
            'dataset_type': dataset_type,
            'num_samples': num_samples,
            'image_size': self.image_size,
            'item_classes': self.ITEM_CLASSES,
            **kwargs
        }
        
        info_path = os.path.join(dataset_dir, 'dataset_info.json')
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
        
        return info_path
    
    def load_dataset_info(self, dataset_dir: str) -> Dict[str, Any]:
        """
        加载数据集信息
        
        Args:
            dataset_dir: 数据集目录
            
        Returns:
            数据集信息
        """
        info_path = os.path.join(dataset_dir, 'dataset_info.json')
        
        if os.path.exists(info_path):
            with open(info_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        
        logger.warning(f"数据集信息文件不存在: {info_path}")
        return {}

# 示例用法
if __name__ == '__main__':
    import sys
    
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('dataset_generator.log')
        ]
    )
    
    # 创建生成器
    generator = LostFoundDatasetGeneratorComplete(
        output_dir='./data',
        image_size=(256, 256),
        max_objects_per_image=3,
        random_seed=42
    )
    
    # 生成合成数据集
    logger.info("开始生成示例数据集")
    dataset_dir = generator.generate_synthetic_dataset(
        num_samples=10,
        output_json=True
    )
    
    logger.info(f"示例数据集生成完成: {dataset_dir}")