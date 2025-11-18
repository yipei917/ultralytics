# _*_ coding : utf-8 _*_
# @Time : 2025-01-12
# @Description : 整合labelme转换、数据集分割、配置生成的完整YOLO训练管道

import os
import json
import shutil
import random
import argparse
from datetime import datetime
from sklearn.model_selection import train_test_split




def convert_labelme_to_yolo(json_path, output_dir, label_map):
    """
    将 Labelme 格式的 JSON 文件转换为 YOLO11 格式的 TXT 文件。
    """
    try:
        # 检查文件是否为空
        if os.path.getsize(json_path) == 0:
            print(f"Warning: 跳过空文件 {json_path}")
            return False
        
        # 打开 Labelme 格式的 JSON 文件
        with open(json_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
            if not content:
                print(f"Warning: 跳过空内容文件 {json_path}")
                return False
            
            try:
                labelme_data = json.loads(content)
            except json.JSONDecodeError as e:
                print(f"Warning: JSON解析失败，跳过文件 {json_path}: {e}")
                return False
    except Exception as e:
        print(f"Warning: 读取文件失败，跳过文件 {json_path}: {e}")
        return False

    # 验证JSON结构
    required_keys = ['imageWidth', 'imageHeight', 'shapes']
    for key in required_keys:
        if key not in labelme_data:
            print(f"Warning: JSON文件缺少必需字段 '{key}'，跳过文件 {json_path}")
            return False
    
    # 获取图像的宽度和高度
    try:
        image_width = labelme_data['imageWidth']
        image_height = labelme_data['imageHeight']
        
        if image_width <= 0 or image_height <= 0:
            print(f"Warning: 图像尺寸无效 (宽:{image_width}, 高:{image_height})，跳过文件 {json_path}")
            return False
            
    except (KeyError, TypeError, ValueError) as e:
        print(f"Warning: 获取图像尺寸失败，跳过文件 {json_path}: {e}")
        return False

    yolo_annotations = []  # 存储 YOLO11 格式的标注

    # 遍历所有的标注形状
    for shape in labelme_data['shapes']:
        label = shape['label'].strip()  # 获取标签名称
        if label not in label_map:
            print(f"Warning: 标签 '{label}' 未在标签映射中，跳过此标注")
            continue  # 如果标签未定义，则忽略

        class_id = label_map[label]  # 获取对应的类别 ID
        points = shape['points']  # 获取标注的坐标点

        if shape['shape_type'] == 'rectangle':  # 如果是矩形
            x1, y1 = min(point[0] for point in points), min(point[1] for point in points)
            x2, y2 = max(point[0] for point in points), max(point[1] for point in points)

        elif shape['shape_type'] == 'polygon':  # 如果是多边形
            x1, y1 = min(point[0] for point in points), min(point[1] for point in points)
            x2, y2 = max(point[0] for point in points), max(point[1] for point in points)

        elif shape['shape_type'] == 'circle':  # 处理圆形标注
            # 圆形的两个点分别是圆心和圆上的某个点，计算圆的半径
            (cx, cy), (x, y) = points
            # 计算半径
            r = ((x - cx) ** 2 + (y - cy) ** 2) ** 0.5
            # 计算最小外接矩形
            x1 = cx - r
            y1 = cy - r
            x2 = cx + r
            y2 = cy + r

        else:
            print(f"Warning: 不支持的标注类型 '{shape['shape_type']}'，跳过此标注")
            continue  # 其他类型不处理

        # 计算 YOLO11 格式所需的中心点和宽高
        x_center = max(0, (x1 + x2) / 2.0 / image_width)
        y_center = max(0, (y1 + y2) / 2.0 / image_height)
        width = max(0, (x2 - x1) / image_width)
        height = max(0, (y2 - y1) / image_height)

        # 添加 YOLO11 格式的标注到列表中
        yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # 构建输出文件的路径
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(json_path))[0] + '.txt')
    # 将 YOLO11 格式的标注写入输出文件
    with open(output_file, 'w') as f:
        f.write('\n'.join(yolo_annotations))
    
    return True


def process_labelme_folder(input_folder, output_folder, label_map, verbose=True):
    """
    处理输入文件夹中的所有 JSON 文件，并将其转换为 YOLO11 格式的 TXT 文件。
    """
    os.makedirs(output_folder, exist_ok=True)  # 创建输出文件夹
    converted_count = 0
    skipped_count = 0
    
    # 获取所有JSON文件
    json_files = [f for f in os.listdir(input_folder) if f.endswith(".json")]
    if verbose:
        print(f"找到 {len(json_files)} 个JSON文件")
    
    for filename in json_files:
        json_path = os.path.join(input_folder, filename)
        if verbose:
            print(f"处理文件: {filename}")
        
        # 调用转换函数并检查返回值
        success = convert_labelme_to_yolo(json_path, output_folder, label_map)
        if success:
            converted_count += 1
        else:
            skipped_count += 1
    
    if verbose:
        print(f"成功转换 {converted_count} 个JSON标注文件")
        if skipped_count > 0:
            print(f"跳过 {skipped_count} 个有问题的文件")
    
    return converted_count


def split_dataset(image_dir, label_dir, output_dir, train_rate=0.8, val_rate=0.1, test_rate=0.1):
    """
    按比例分割数据集为训练集、验证集和测试集
    """
    # 支持多种图片格式
    image_exts = ('.jpg', '.jpeg', '.png', '.bmp')
    images = [f for f in os.listdir(image_dir) if f.lower().endswith(image_exts)]
    
    # 根据图片文件名生成对应的标签文件名（假设标签格式为 .txt）
    labels = [os.path.splitext(f)[0] + '.txt' for f in images]  

    # 确保图片和标签文件一一对应
    valid_images = []
    for image, label in zip(images, labels):
        if os.path.exists(os.path.join(label_dir, label)):
            valid_images.append(image)
        else:
            print(f"Warning: 图片 {image} 对应的标签文件未找到，跳过此图片")
    
    print(f"找到 {len(valid_images)} 个有效的图片-标签对")
    
    if len(valid_images) == 0:
        raise ValueError("没有找到有效的图片-标签对！")
    
    # 使用有效的图片列表进行划分
    train_images, val_test_images = train_test_split(valid_images, test_size=(val_rate + test_rate), random_state=42)
    val_images, test_images = train_test_split(val_test_images, test_size=(test_rate / (val_rate + test_rate)), random_state=42)

    subsets = [('train', train_images), ('val', val_images), ('test', test_images)]

    # 创建输出文件夹并复制文件
    for subset, subset_images in subsets:
        os.makedirs(f"{output_dir}/images/{subset}", exist_ok=True)
        os.makedirs(f"{output_dir}/labels/{subset}", exist_ok=True)
        
        for image in subset_images:
            # 复制图片文件
            shutil.copy(os.path.join(image_dir, image), f"{output_dir}/images/{subset}/{image}")
            # 复制对应的标签文件
            label_file = os.path.splitext(image)[0] + '.txt'
            shutil.copy(os.path.join(label_dir, label_file), f"{output_dir}/labels/{subset}/{label_file}")
    
    print(f"数据集分割完成:")
    print(f"  训练集: {len(train_images)} 个样本")
    print(f"  验证集: {len(val_images)} 个样本") 
    print(f"  测试集: {len(test_images)} 个样本")


def generate_data_yaml(output_dir, dataset_path, label_names):
    """
    生成data.yaml配置文件
    """
    # 将路径转换为绝对路径
    train_path = os.path.abspath(os.path.join(dataset_path, "images/train"))
    val_path = os.path.abspath(os.path.join(dataset_path, "images/val"))
    test_path = os.path.abspath(os.path.join(dataset_path, "images/test"))
    
    # 格式化类别名称列表，确保YAML格式正确（单类别时也要是列表格式）
    names_str = str(label_names) if len(label_names) > 1 else f"['{label_names[0]}']"
    
    data_yaml_content = f"""train: {train_path}
val: {val_path}
test: {test_path}

nc: {len(label_names)}  # 类别数量
names: {names_str}  # 类别名称，需与标签文件中的类别一致
"""
    
    data_yaml_path = os.path.join(output_dir, 'data.yaml')
    with open(data_yaml_path, 'w', encoding='utf-8') as f:
        f.write(data_yaml_content)
    
    print(f"已生成 data.yaml 文件: {data_yaml_path}")
    print(f"  - 类别数量(nc): {len(label_names)}")
    print(f"  - 类别名称(names): {label_names}")
    return data_yaml_path


def generate_train_script(output_dir, dataset_name):
    """
    生成train.py训练脚本
    """
    train_script_content = f"""from ultralytics.models import YOLO
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
 
if __name__ == '__main__':
    model = YOLO(model='/home/gdw/train-center/ultralytics/yolo11n.pt')
    model.train(
        data='./data.yaml',         # 数据集配置文件路径
        epochs=100,                 # 训练轮数
        batch=48,                   # 批次大小
        device='1,2,3',             # 使用的GPU设备编号
        workers=16,                 # 数据加载线程数
        lr0=0.01,                   # 初始学习率
        lrf=0.01,                   # 最终学习率比例
        momentum=0.937,             # 优化器动量
        weight_decay=0.0005,        # 权重衰减
        optimizer='SGD',            # 优化器类型
        amp=False,                  # 是否启用混合精度训练
        project='train',       # 训练结果保存目录
        name='{dataset_name}'
    )
"""
    
    train_script_path = os.path.join(output_dir, 'train.py')
    with open(train_script_path, 'w', encoding='utf-8') as f:
        f.write(train_script_content)
    
    print(f"已生成 train.py 文件: {train_script_path}")
    return train_script_path


def main():
    parser = argparse.ArgumentParser(description='YOLO训练数据准备完整管道')
    parser.add_argument('-i', '--json_dir', default='../rack_data', help='Labelme标注JSON文件夹路径')
    parser.add_argument('-n', '--dataset_name', default='rack', help='数据集名称')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例 (默认: 0.8)')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例 (默认: 0.1)')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='测试集比例 (默认: 0.1)')
    
    args = parser.parse_args()
    
    # 验证参数
    if not os.path.exists(args.json_dir):
        raise FileNotFoundError(f"JSON文件夹不存在: {args.json_dir}")
    
    if abs(args.train_ratio + args.val_ratio + args.test_ratio - 1.0) > 0.001:
        raise ValueError("训练集、验证集、测试集比例之和必须等于1.0")
    
    # 生成数据集名称
    if args.dataset_name:
        dataset_name = args.dataset_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        dataset_name = f"dataset_{timestamp}"
    
    print(f"开始处理数据集: {dataset_name}")
    
    # 创建输出目录结构
    datasets_dir = "./datasets"
    os.makedirs(datasets_dir, exist_ok=True)
    
    dataset_output_dir = os.path.join(datasets_dir, dataset_name)
    os.makedirs(dataset_output_dir, exist_ok=True)
    
    # 获取脚本所在目录，用于生成配置文件
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if not script_dir:
        script_dir = os.getcwd()
    
    print(f"配置文件将保存到: {script_dir}")
    
    try:
        # 1. 设置固定的标签映射（只有一个类别：NG）
        label_map = {'NG': 0}
        label_names = ['NG']
        print(f"✅ 使用固定标签映射: NG -> 0")
        
        # 2. 转换labelme标注为YOLO格式
        print("\n正在转换Labelme标注为YOLO格式...")
        temp_labels_dir = os.path.join(dataset_output_dir, "temp_labels")
        convert_count = process_labelme_folder(args.json_dir, temp_labels_dir, label_map)
        
        if convert_count == 0:
            raise ValueError("没有找到可转换的JSON标注文件！")
        
        # 3. 分割数据集
        print("正在分割数据集...")
        split_dataset(
            args.json_dir, 
            temp_labels_dir, 
            dataset_output_dir,
            args.train_ratio, 
            args.val_ratio, 
            args.test_ratio
        )
        
        # 4. 清理临时标签文件夹
        shutil.rmtree(temp_labels_dir)
        
        # 5. 生成配置文件（直接生成在脚本同级目录）
        print("正在生成配置文件...")
        data_yaml_path = generate_data_yaml(script_dir, dataset_output_dir, label_names)
        train_script_path = generate_train_script(script_dir, dataset_name)
        
        print(f"\n✅ 数据集准备完成！")
        print(f"📁 数据集位置: {dataset_output_dir}")
        print(f"📄 数据配置: {data_yaml_path}")
        print(f"📄 训练脚本: {train_script_path}")
        print(f"\n🚀 下一步: 在当前目录运行 python train.py")
        
    except Exception as e:
        print(f"❌ 处理过程中出现错误: {str(e)}")
        # 清理可能创建的目录
        if os.path.exists(dataset_output_dir) and not os.listdir(dataset_output_dir):
            os.rmdir(dataset_output_dir)
        raise


if __name__ == '__main__':
    main()
