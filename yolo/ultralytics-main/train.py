import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(model=r'ultralytics/cfg/models/11/yolo11-cbam-seg.yaml')
    model.load('yolo11m-seg.pt')  # 迁移预训练权重，没有就先 pip install ultralytics 让它自动下载

    model.train(data=r'data.yaml',
                imgsz=640,
                epochs=150,
                batch=16,        # 显存不够改成 8
                workers=4,
                device='',
                # 优化器
                optimizer='AdamW',
                lr0=0.001,
                cos_lr=True,
                warmup_epochs=5,
                close_mosaic=20,
                # 几何增强
                degrees=15.0,
                scale=0.6,
                perspective=0.0005,
                # 混合增强
                mixup=0.15,
                copy_paste=0.1,
                # 亮度增强（应对比赛场地光照）
                hsv_v=0.5,
                # 其他
                resume=False,
                amp=True,
                project='runs/train',
                name='v2_cbam_seg',
                single_cls=False,
                cache=False,
                )
