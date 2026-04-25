import pyrealsense2 as rs
import numpy as np
import cv2
import os
from datetime import datetime

# 当前时间处理 (UTC 2026-04-07 09:05:56 → 北京时间 2026-04-07 17:05:56)
utc_time = "2026-04-07 17:05:56"  # 实际使用时替换为datetime.now()获取当前时间
current_time = datetime.strptime(utc_time, "%Y-%m-%d %H:%M:%S").strftime("%Y%m%d_%H%M%S")

# 创建保存目录
color_dir = f"color_images_{current_time}"
depth_dir = f"depth_images_{current_time}"
os.makedirs(color_dir, exist_ok=True)
os.makedirs(depth_dir, exist_ok=True)

# 初始化RealSense管道
pipeline = rs.pipeline()
config = rs.config()

# 启用彩色和深度流
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

# 创建对齐对象（深度对齐到彩色）
align_to = rs.stream.color
align = rs.align(align_to)

try:
    # 启动管道
    pipeline.start(config)
    
    frame_count = 0
    print("开始采集图像 (按ESC退出)...")
    
    while True:
        # 等待一组连贯的帧
        frames = pipeline.wait_for_frames()
        
        # 对齐深度帧到彩色帧
        aligned_frames = align.process(frames)
        
        # 获取对齐后的帧
        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()
        
        if not color_frame or not depth_frame:
            continue
        
        # 转换为OpenCV格式
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        
        # 显示图像
        cv2.imshow('Color Image', color_image)
        
        # 应用颜色映射到深度图像以便可视化
        depth_colormap = cv2.applyColorMap(
            cv2.convertScaleAbs(depth_image, alpha=0.03), 
            cv2.COLORMAP_JET
        )
        cv2.imshow('Depth Image', depth_colormap)
        
        # 保存图像
        color_filename = os.path.join(color_dir, f"color_{frame_count:04d}.png")
        depth_filename = os.path.join(depth_dir, f"depth_{frame_count:04d}.png")
        
        cv2.imwrite(color_filename, color_image)
        cv2.imwrite(depth_filename, depth_image)
        
        print(f"已保存: {color_filename} 和 {depth_filename}")
        frame_count += 1
        
        # 按ESC退出
        key = cv2.waitKey(1)
        if key == 27:  # ESC键
            break

finally:
    # 停止管道
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"采集完成! 彩色图像保存在: {color_dir}")
    print(f"深度图像保存在: {depth_dir}")
