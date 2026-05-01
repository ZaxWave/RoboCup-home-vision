import pyrealsense2 as rs
import cv2
import time

def main():
    # 用户设定物体名称
    object_name = input("请输入物体名称（将作为图片文件名前缀）: ").strip()
    if not object_name:
        object_name = "object"
        print(f"未输入名称，使用默认名称 '{object_name}'")

    # 配置 RealSense 管道
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)  # 彩色流
    pipeline.start(config)

    # 获取对齐对象（可选，彩色与深度对齐，这里只需要彩色）
    # 创建窗口
    cv2.namedWindow('RealSense Camera', cv2.WINDOW_AUTOSIZE)

    capturing = False           # 拍摄状态
    seq = 1                     # 图片序号
    last_capture_time = 0       # 上次拍摄的时间戳
    capture_interval = 1.0      # 拍摄间隔（秒）

    print("\n操作说明：")
    print("  空格键 -> 开始/停止连续拍摄（每秒一张）")
    print("  ESC键  -> 退出程序")
    print("\n等待按键...")

    try:
        while True:
            # 等待新的一帧（超时 5 秒）
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            # 转换为 numpy 数组并显示
            color_image = np.asanyarray(color_frame.get_data())
            cv2.imshow('RealSense Camera', color_image)

            # 按键检测
            key = cv2.waitKey(1) & 0xFF
            if key == 27:               # ESC 退出
                print("退出程序")
                break
            elif key == 32:             # 空格键切换拍摄状态
                capturing = not capturing
                if capturing:
                    # 开始拍摄，重置时间戳以避免立即拍摄
                    last_capture_time = time.time()
                    print("开始连续拍摄（每秒一张）")
                else:
                    print("停止拍摄")

            # 如果处于拍摄状态且间隔时间已到
            if capturing:
                now = time.time()
                if now - last_capture_time >= capture_interval:
                    # 生成文件名 "物体名称序号" （例如 apple1.jpg）
                    filename = f"{object_name}{seq}.jpg"
                    cv2.imwrite(filename, color_image)
                    print(f"已保存: {filename}")
                    seq += 1
                    last_capture_time = now

    finally:
        # 释放资源
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    import numpy as np   # 放在这里避免干扰
    main()