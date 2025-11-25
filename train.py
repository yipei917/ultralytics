from ultralytics.models import YOLO
from ultralytics.models import RTDETR
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# 将本地代码以开发模式安装
# pip install -e /home/gdw/object_detection/ultralytics

if __name__ == '__main__':
    # 方式1: 使用本地ultralytics中的yaml配置文件（从头开始训练，不加载预训练权重）
    # model = YOLO(model='./ultralytics/cfg/models/11/yolo11n.yaml')
    
    # 方式2: 使用预训练权重文件（基于预训练模型fine-tune，推荐）
    # model = YOLO(model='yolo11n.pt')

    model = RTDETR(model='./ultralytics/cfg/models/my-detr/rtdetr-l-eva.yaml')
    model.info()
    model.train(
        data='./data.yaml',         # 数据集配置文件路径
        epochs=100,                 # 训练轮数
        batch=12,                   # 批次大小
        device='6,7',
        workers=4,                 # 数据加载线程数
        lr0=0.01,                   # 初始学习率
        lrf=0.01,                   # 最终学习率比例
        momentum=0.937,             # 优化器动量
        weight_decay=0.0005,        # 权重衰减
        optimizer='SGD',            # 优化器类型
        amp=False,                  # 是否启用混合精度训练
        deterministic=False,        # 关闭确定性算法，避免 grid_sampler_2d 警告
        project='train',            # 训练结果保存目录
        name='local_rtdetr-l-eva_e100'
    )
