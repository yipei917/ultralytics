from ultralytics.models import RTDETR
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

if __name__ == '__main__':
    model = RTDETR(model='../ultralytics/cfg/models/rt-detr/rtdetr-l.yaml')
    model.info()
    model.train(
        data='./data.yaml',         # 数据集配置文件路径
        epochs=1,                 # 训练轮数
        batch=1,                   # 批次大小
        device='8',
        workers=1,                 # 数据加载线程数
        project='debug',
        amp=False,
        deterministic=False
    )
