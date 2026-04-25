import cv2
import numpy as np
import time
import os
import sys
from datetime import datetime, timedelta
import pyrealsense2 as rs

# 限差参数
cosine_similar_thresh = 0.363
l2norm_similar_thresh = 1.128
scale = 1.0

# 画框可视化函数
def visualize(input_img, frame, faces, fps, thickness=2):
    fps_string = f"FPS : {fps:.2f}"
    if frame >= 0:
        print(f"Frame {frame}, ", end="")
    print(f"FPS: {fps_string}")
    
    if faces is None:
        return
    
    for i in range(faces.shape[0]):
        face = faces[i]
        print(f"Face {i}, top-left coordinates: ({face[0]}, {face[1]}), "
              f"box width: {face[2]}, box height: {face[3]}, "
              f"score: {face[14]:.2f}")
        
        # 绘制边界框
        cv2.rectangle(input_img, 
                     (int(face[0]), int(face[1])),
                     (int(face[0] + face[2]), int(face[1] + face[3])),
                     (0, 255, 0), thickness)
        
        # 绘制特征点
        cv2.circle(input_img, (int(face[4]), int(face[5])), 2, (255, 0, 0), thickness)
        cv2.circle(input_img, (int(face[6]), int(face[7])), 2, (0, 0, 255), thickness)
        cv2.circle(input_img, (int(face[8]), int(face[9])), 2, (0, 255, 0), thickness)
        cv2.circle(input_img, (int(face[10]), int(face[11])), 2, (255, 0, 255), thickness)
        cv2.circle(input_img, (int(face[12]), int(face[13])), 2, (0, 255, 255), thickness)
    
    # 添加FPS文本
    cv2.putText(input_img, fps_string, (0, 15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# 在图像上写入识别结果
def write_names(input_img, faces, names):
    if faces is None:
        return
    
    for i in range(faces.shape[0]):
        org = (int(faces[i][0]), int(faces[i][1]))
        font_face = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1
        color = (255, 0, 0)
        thickness = 2
        
        cv2.putText(input_img, names[i], org, font_face, 
                   font_scale, color, thickness)

# 保存人脸特征数据
def save_face_data(image, faces, names):
    face_recognizer = cv2.FaceRecognizerSF_create(
        "face_detect/model/face_recognition_sface_2021dec.onnx", "")
    
    for i in range(faces.shape[0]):
        aligned_face = face_recognizer.alignCrop(image, faces[i])
        feature = face_recognizer.feature(aligned_face)
        
        # 保存特征到XML
        fs = cv2.FileStorage("face_detect/feature/vocabulary.xml", cv2.FILE_STORAGE_APPEND)
        fs.write(names[i], feature)
        fs.release()
        
        # 保存名字到文本文件
        try:
            with open("face_detect/feature/name.txt", "a") as f:
                f.write("\n" + names[i])
        except Exception as e:
            print(f"无法保存名字文件: {e}")

# 初始化RealSense相机
def initialize_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 配置颜色流
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    # 启动管道
    pipeline.start(config)
    
    # 对齐对象（用于对齐深度到颜色）
    align_to = rs.stream.color
    align = rs.align(align_to)
    
    return pipeline, align

# 从本地数据库识别人脸（使用RealSense）
def recognize_people():
    # 初始化RealSense相机
    pipeline, align = initialize_realsense()
    
    # 初始化人脸检测器（稍后设置尺寸）
    detector = cv2.FaceDetectorYN_create(
        "face_detect/model/face_detection_yunet_2023mar.onnx", "", 
        (320, 320), 0.9, 0.3, 5000)
    
    # 初始化人脸识别器
    face_recognizer = cv2.FaceRecognizerSF_create(
        "face_detect/model/face_recognition_sface_2021dec.onnx", "")
    
    # 加载已知人名
    names = []
    try:
        with open("face_detect/feature/name.txt", "r") as f:
            names = [line.strip() for line in f if line.strip()]
    except Exception as e:
        print(f"无法加载名字文件: {e}")
    
    print(f"已加载 {len(names)} 个名字")
    
    frame_count = 0
    tm = cv2.TickMeter()
    
    try:
        while True:
            # 等待RealSense帧
            frames = pipeline.wait_for_frames()
            
            # 对齐深度帧到颜色帧
            aligned_frames = align.process(frames)
            
            # 获取对齐后的颜色帧
            color_frame = aligned_frames.get_color_frame()
            if not color_frame:
                continue
            
            # 将颜色帧转换为OpenCV格式
            ##frame = np.asanyarray(color_frame.get_data())
            frame = cv2.imread("C:/Users/Agora/robohome/face_detect/image/xun_zhou.jpeg")
            

            # 获取实际帧尺寸并缩放
            frame_height, frame_width = frame.shape[:2]
            scaled_width = int(frame_width * scale)
            scaled_height = int(frame_height * scale)
            
            # 设置检测器输入尺寸
            detector.setInputSize((scaled_width, scaled_height))
            
            # 调整帧大小
            resized_frame = cv2.resize(frame, (scaled_width, scaled_height))
            
            # 人脸检测
            tm.start()
            _, faces = detector.detect(resized_frame)
            tm.stop()
            fps = tm.getFPS()
            
            tar_name = []
            if faces is not None:
                for i in range(faces.shape[0]):
                    tar_name.append("unknown")
                    print("检测到人脸")
                    
                    # 人脸对齐和特征提取
                    aligned_face = face_recognizer.alignCrop(resized_frame, faces[i])
                    feature = face_recognizer.feature(aligned_face)
                    
                    # 从XML加载特征
                    try:
                        fs = cv2.FileStorage("face_detect/feature/vocabulary.xml", cv2.FILE_STORAGE_READ)
                        
                        # 与数据库中的每个特征比较
                        for name in names:
                            mat_vocabulary = fs.getNode(name).mat()
                            if mat_vocabulary is None:
                                continue
                                
                            # 计算相似度
                            cos_score = face_recognizer.match(
                                feature, mat_vocabulary, 
                                cv2.FaceRecognizerSF_FR_COSINE)
                            
                            l2_score = face_recognizer.match(
                                feature, mat_vocabulary,
                                cv2.FaceRecognizerSF_FR_NORM_L2)
                            
                            print(f"{name} {cos_score} {l2_score}")
                            
                            # 检查是否匹配
                            if cos_score > cosine_similar_thresh and l2_score < l2norm_similar_thresh:
                                tar_name[i] = name
                                break
                    except Exception as e:
                        print(f"特征匹配错误: {e}")
            
            # 可视化结果
            result = resized_frame.copy()
            visualize(result, frame_count, faces, fps)
            write_names(result, faces, tar_name)
            
            # 显示结果
            cv2.imshow("Live Face Recognition (RealSense)", result)
            
            # 按键处理
            key = cv2.waitKey(30)
            if key == 27:  # ESC键退出
                break
            if key == 32:
                cv2.imwrite("face_detect/image/new.png",result)
            frame_count += 1
    
    finally:
        # 确保管道被正确停止
        pipeline.stop()
        cv2.destroyAllWindows()

# 摄像头人脸检测（使用RealSense）
def camera_detect_people():
    # 初始化RealSense相机
    pipeline, align = initialize_realsense()
    
    # 初始化人脸检测器
    detector = cv2.FaceDetectorYN_create(
        "face_detect/model/face_detection_yunet_2023mar.onnx", "", 
        (320, 320), 0.9, 0.3, 5000)
    
    frame_count = 0
    tm = cv2.TickMeter()
    
    try:
        while True:
            # 等待RealSense帧
            frames = pipeline.wait_for_frames()
            
            # 对齐深度帧到颜色帧
            aligned_frames = align.process(frames)
            
            # 获取对齐后的颜色帧
            color_frame = aligned_frames.get_color_frame()
            if not color_frame:
                continue
            
            # 将颜色帧转换为OpenCV格式
            frame = np.asanyarray(color_frame.get_data())

            # 获取实际帧尺寸
            frame_height, frame_width = frame.shape[:2]
            frame_width = int(frame_width * scale)
            frame_height = int(frame_height * scale)
            
            # 设置检测器输入尺寸
            detector.setInputSize((frame_width, frame_height))
            
            # 调整帧大小
            frame = cv2.resize(frame, (frame_width, frame_height))
            
            # 人脸检测
            tm.start()
            _, faces = detector.detect(frame)
            tm.stop()
            fps = tm.getFPS()
            
            result = frame.copy()
            visualize(result, frame_count, faces, fps)
            cv2.imshow("Face Detection (RealSense)", result)
            
            key = cv2.waitKey(30)
            if key == 32:  # 空格键保存
                names = ["Wangzhuochao"]
                save_face_data(result, faces, names)
            elif key == 27:  # ESC键退出
                break
    
    finally:
        # 确保管道被正确停止
        pipeline.stop()
        cv2.destroyAllWindows()

# 图像文件人脸检测（保持不变）
def image_detect_people(filepath, names):
    image = cv2.imread(filepath)
    if image is None:
        print("图片加载失败")
        return
    
    frame_width = int(image.shape[1] * scale)
    frame_height = int(image.shape[0] * scale)
    image = cv2.resize(image, (frame_width, frame_height))
    
    detector = cv2.FaceDetectorYN_create(
        "face_detect/model/face_detection_yunet_2023mar.onnx", "", 
        (320, 320), 0.9, 0.3, 5000)
    detector.setInputSize((frame_width, frame_height))
    
    tm = cv2.TickMeter()
    tm.start()
    _, faces = detector.detect(image)
    tm.stop()
    
    if faces is None or faces.shape[0] < 1:
        print("无法在图像中找到人脸")
        return
    
    print(f"检测到 {faces.shape[0]} 张人脸")
    visualize(image, -1, faces, tm.getFPS())
    
    while True:
        cv2.imshow("Image Face Detection", image)
        key = cv2.waitKey(25)
        
        if key == 32:  # 空格键保存
            save_face_data(image, faces, names)
        elif key == 27:  # ESC键退出
            break
    
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # 主程序入口
    #camera_detect_people()
    recognize_people()
    
    # 图像检测示例
    # names = ["YanZu"]
    # image_detect_people("./image/yanzu_wu1.png", names)
