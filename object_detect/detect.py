import cv2
import pyrealsense2 as rs
import time
import numpy as np
import math
from ultralytics import YOLO

model = YOLO("runs/train/exp/weights/best.pt")  # 换成 seg 训练后的 best.pt

pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 848, 480, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 848, 480, rs.format.bgr8, 30)

pipeline.start(config)
align_to = rs.stream.color
align = rs.align(align_to)

# 获取深度比例尺（米/单位）
profile = pipeline.get_active_profile()
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale = depth_sensor.get_depth_scale()

def get_aligned_images():
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    aligned_depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    intr = color_frame.profile.as_video_stream_profile().intrinsics
    depth_intrin = aligned_depth_frame.profile.as_video_stream_profile().intrinsics

    depth_image = np.asanyarray(aligned_depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())

    return intr, depth_intrin, color_image, depth_image, aligned_depth_frame

def get_3d_camera_coordinate(depth_pixel, aligned_depth_frame, depth_intrin):
    x = depth_pixel[0]
    y = depth_pixel[1]
    dis = aligned_depth_frame.get_distance(x, y)
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, depth_pixel, dis)
    return dis, camera_coordinate

def get_mask_depth(mask_pts, depth_image, depth_intrin, h, w):
    """用 mask 区域深度中位数替代单点深度，更稳定"""
    mask_img = np.zeros((h, w), dtype=np.uint8)
    pts = np.array(mask_pts, dtype=np.int32)
    cv2.fillPoly(mask_img, [pts], 1)

    raw_depths = depth_image[mask_img == 1].astype(np.float32) * depth_scale
    valid = raw_depths[(raw_depths > 0.1) & (raw_depths < 3.5)]
    if len(valid) == 0:
        return None, None

    dis = float(np.median(valid))
    # 用 mask 重心作为像素坐标
    M = cv2.moments(mask_img)
    if M["m00"] == 0:
        return None, None
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    camera_coordinate = rs.rs2_deproject_pixel_to_point(depth_intrin, [cx, cy], dis)
    return dis, camera_coordinate, (cx, cy)

fps = 0
start_time = time.time()

try:
    while True:
        intr, depth_intrin, color_image, depth_image, aligned_depth_frame = get_aligned_images()
        if not depth_image.any() or not color_image.any():
            continue

        h, w = color_image.shape[:2]
        time1 = time.time()

        results = model.predict(color_image, conf=0.35, augment=True)
        annotated_frame = results[0].plot()
        names_dic = results[0].names
        boxes = results[0].boxes

        has_masks = results[0].masks is not None

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            name = names_dic[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if has_masks and i < len(results[0].masks.xy):
                # --- seg 模式：用 mask 区域取深度 ---
                mask_pts = results[0].masks.xy[i]
                result = get_mask_depth(mask_pts, depth_image, depth_intrin, h, w)
                if result[0] is None:
                    continue
                dis, camera_coordinate, (cx, cy) = result
            else:
                # --- detect 模式：用 bbox 中心点取深度（兼容旧模型）---
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                dis, camera_coordinate = get_3d_camera_coordinate([cx, cy], aligned_depth_frame, depth_intrin)
                # 过滤距离异常的误检
                if dis < 0.1 or dis > 3.5:
                    continue

            formatted_coord = f"({camera_coordinate[0]:.2f}, {camera_coordinate[1]:.2f}, {camera_coordinate[2]:.2f})"

            with open("object_detect/point_cloud_data.txt", "a") as f:
                f.write(f"\nTime: {time.time()}\n")
                f.write(f"{name}\n")
                f.write(f"{formatted_coord}\n")

            cv2.circle(annotated_frame, (cx, cy), 4, (255, 255, 255), 5)
            cv2.putText(annotated_frame, formatted_coord, (cx + 20, cy + 10),
                        0, 0.6, [225, 255, 255], thickness=1, lineType=cv2.LINE_AA)

        time2 = time.time()
        fps = int(1 / (time2 - time1))
        cv2.putText(annotated_frame, f'FPS: {fps}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('YOLOv8-seg RealSense', annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
