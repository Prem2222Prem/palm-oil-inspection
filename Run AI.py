import cv2
import sys
import os
import time  
from datetime import datetime
from ultralytics import YOLO

# ---------------------------------------------------------
# 1. ตั้งค่าไฟล์โมเดล
# ---------------------------------------------------------
model_path = 'C:/Users/Prem/Desktop/palm inspection AI.pt'
save_folder = 'detected_images'

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

# ---------------------------------------------------------
# 2. โหลดโมเดล
# ---------------------------------------------------------
try:
    print(f"🔄 กำลังโหลดโมเดลจาก: {model_path} ...")
    model = YOLO(model_path)
    print("✅ โหลดโมเดลสำเร็จ!")
except Exception as e:
    print(f"\n❌ โหลดโมเดลไม่สำเร็จ! สาเหตุ: {e}")
    sys.exit()

# ---------------------------------------------------------
# 3. เปิดกล้อง
# ---------------------------------------------------------
print("📷 กำลังเปิดกล้อง...")
# 💡 หมายเหตุ: หากเปิดกล้องแล้วยังมี Error ดึงภาพไม่ได้ ให้แก้เป็น cv2.VideoCapture(1, cv2.CAP_DSHOW)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ เปิดกล้องไม่ติด!")
    sys.exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# ตัวแปรสำหรับคำนวณ FPS
prev_time = 0

# ---------------------------------------------------------
# 4. เริ่มทำงาน (Main Loop)
# ---------------------------------------------------------
try:
    print("🚀 ระบบเริ่มทำงาน... [กด 's' บันทึก / 'q' ออก]")
    
    while True:
        success, frame = cap.read()
        if not success:
            print("⚠️ อ่านภาพจากกล้องไม่ได้")
            break

        # --- ส่วนการคำนวณ FPS ---
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # ---------------------------------------------------------
        # 🛠️ จุดที่แก้ไข: ปรับจูนพารามิเตอร์การทำนาย (Predict)
        # ---------------------------------------------------------
        results = model.predict(
            frame, 
            conf=0.65,          # 1. เพิ่มค่าความเชื่อมั่นเป็น 65% เพื่อลดการทายมั่ว (ปรับลด-เพิ่มได้ตามหน้างานจริง) 
            agnostic_nms=True,  # 2. ป้องกันกล่อง 2 คลาส (ดี/เสีย) ซ้อนทับกันบนผลปาล์มลูกเดียวกัน
            device=0, 
            verbose=False
        )

        # วาดผลลัพธ์การตรวจจับ
        annotated_frame = results[0].plot()

        # --- เขียนค่า FPS ลงบนภาพ ---
        cv2.putText(annotated_frame, f"FPS: {int(fps)}", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # แสดงผล
        cv2.imshow("YOLO Palm Inspection", annotated_frame)

        # ตรวจจับการกดปุ่ม
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("s"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{save_folder}/palm_{timestamp}.jpg"
            cv2.imwrite(filename, annotated_frame)
            print(f"📸 บันทึกภาพแล้ว: {filename} (FPS: {int(fps)})")

except Exception as e:
    print(f"❌ เกิดข้อผิดพลาดขณะรัน: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
    print("✅ ปิดระบบเรียบร้อย")