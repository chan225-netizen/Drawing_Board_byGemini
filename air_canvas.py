import cv2
import mediapipe as mp
import math
import numpy as np
import tkinter as tk
from tkinter import filedialog
import colorsys
import time

# --- 预处理：创建初始窗口 ---
cv2.namedWindow("Air Canvas AI", cv2.WINDOW_NORMAL)
# 创建一个纯黑色初始化背景
loading_screen = np.zeros((720, 1280, 3), dtype=np.uint8)
cv2.putText(loading_screen, "Initializing AI Canvas...", (400, 360), 
            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
cv2.imshow("Air Canvas AI", loading_screen)
cv2.waitKey(1) # 强制刷新显示

# ================= 1. 初始化 =================
# 将模型初始化放在显示加载动画之后
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8, min_tracking_confidence=0.8)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 60)

canvas = np.zeros((720, 1280, 3), dtype=np.uint8)
brush_thickness = 10

is_fullscreen = False
is_recording = False
use_gray_filter = False
video_writer = None

px, py = 0, 0
smooth_x, smooth_y = 0.0, 0.0
smoothing_factor = 0.65  
global_hue = 0.0  

def get_save_path():
    root = tk.Tk()
    root.withdraw()
    root.wm_attributes('-topmost', 1)
    file_path = filedialog.asksaveasfilename(defaultextension=".mp4", filetypes=[("MP4 Video", "*.mp4")])
    root.destroy()
    return file_path

# ================= 2. 主循环 =================
# 初始化完成，进入主循环
while True:
    success, frame = cap.read()
    if not success: 
        # 如果摄像头尚未准备好，继续显示加载文字
        cv2.imshow("Air Canvas AI", loading_screen)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        continue
        
    frame = cv2.flip(frame, 1)
    h, w, c = frame.shape
    
    global_hue = (global_hue + 2) % 360 
    r, g, b = colorsys.hsv_to_rgb(global_hue / 360.0, 1.0, 1.0)
    current_color = (int(b * 255), int(g * 255), int(r * 255))
    
    if use_gray_filter:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        final_frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    else:
        final_frame = frame.copy()
        
    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    if results.multi_hand_landmarks:
        hand_landmarks_list = results.multi_hand_landmarks
        
        if len(hand_landmarks_list) == 2:
            px, py = 0, 0 
            points = []
            for hand_lms in hand_landmarks_list:
                points.extend([(int(hand_lms.landmark[4].x * w), int(hand_lms.landmark[4].y * h)),
                               (int(hand_lms.landmark[8].x * w), int(hand_lms.landmark[8].y * h))])
            
            x_coords, y_coords = [p[0] for p in points], [p[1] for p in points]
            min_x, min_y = max(0, min(x_coords)-20), max(0, min(y_coords)-20)
            max_x, max_y = min(w, max(x_coords)+20), min(h, max(y_coords)+20)
            
            final_frame[min_y:max_y, min_x:max_x] = frame[min_y:max_y, min_x:max_x]
            cv2.rectangle(final_frame, (min_x, min_y), (max_x, max_y), current_color, 2, cv2.LINE_AA)
            
        elif len(hand_landmarks_list) == 1:
            hand_lms = hand_landmarks_list[0]
            lm = hand_lms.landmark
            pinch_dist = math.hypot(lm[8].x - lm[4].x, lm[8].y - lm[4].y)
            palm_size = math.hypot(lm[9].x - lm[0].x, lm[9].y - lm[0].y)
            is_pinching = (pinch_dist / (palm_size + 1e-6)) < 0.25
            raw_cx = int((lm[4].x + lm[8].x) / 2 * w)
            raw_cy = int((lm[4].y + lm[8].y) / 2 * h)
            
            if is_pinching:
                if px == 0 and py == 0:
                    px, py = raw_cx, raw_cy
                    smooth_x, smooth_y = raw_cx, raw_cy
                else:
                    smooth_x = smoothing_factor * raw_cx + (1 - smoothing_factor) * smooth_x
                    smooth_y = smoothing_factor * raw_cy + (1 - smoothing_factor) * smooth_y
                    cx_draw, cy_draw = int(smooth_x), int(smooth_y)
                    cv2.line(canvas, (px, py), (cx_draw, cy_draw), current_color, brush_thickness)
                    cv2.circle(canvas, (cx_draw, cy_draw), brush_thickness // 2, current_color, cv2.FILLED)
                    px, py = cx_draw, cy_draw
                cv2.circle(final_frame, (int(smooth_x), int(smooth_y)), 6, current_color, cv2.FILLED, cv2.LINE_AA)
                cv2.circle(final_frame, (int(smooth_x), int(smooth_y)), 10, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                px, py = 0, 0
                cv2.circle(final_frame, (raw_cx, raw_cy), 8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.circle(final_frame, (raw_cx, raw_cy), 2, current_color, cv2.FILLED, cv2.LINE_AA)

    # 图层融合
    if canvas.shape[:2] != final_frame.shape[:2]:
        canvas = cv2.resize(canvas, (final_frame.shape[1], final_frame.shape[0]))
    canvas_gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(canvas_gray, 1, 255, cv2.THRESH_BINARY)
    mask_inv = cv2.bitwise_not(mask)
    final_frame_bg = cv2.bitwise_and(final_frame, final_frame, mask=mask_inv)
    final_frame = cv2.add(final_frame_bg, canvas)

    if is_recording and video_writer:
        video_writer.write(final_frame)
        cv2.circle(final_frame, (50, 50), 10, (0, 0, 255), -1, cv2.LINE_AA)

    cv2.putText(final_frame, "W: Full | L: Filter | V: Record | C: Clear | Q: Quit", (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 2, cv2.LINE_AA)
    
    cv2.imshow("Air Canvas AI", final_frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'): break
    elif key == ord('c'): canvas = np.zeros((h, w, 3), dtype=np.uint8)
    elif key == ord('l'): use_gray_filter = not use_gray_filter
    elif key == ord('w'):
        is_fullscreen = not is_fullscreen
        cv2.setWindowProperty("Air Canvas AI", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN if is_fullscreen else cv2.WINDOW_NORMAL)
    elif key == ord('v'):
        if not is_recording:
            path = get_save_path()
            if path:
                fps = cap.get(cv2.CAP_PROP_FPS)
                if fps == 0 or math.isnan(fps): fps = 30.0 
                real_h, real_w = final_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video_writer = cv2.VideoWriter(path, fourcc, fps, (real_w, real_h))
                if video_writer.isOpened(): is_recording = True
        else:
            is_recording = False
            if video_writer: video_writer.release()

cap.release()
if video_writer: video_writer.release()
cv2.destroyAllWindows()