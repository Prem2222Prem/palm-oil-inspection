import torch

print(f"PyTorch Version: {torch.__version__}")

if torch.cuda.is_available():
    print(f"✅ สำเร็จ! GPU พร้อมใช้งาน: {torch.cuda.get_device_name(0)}")
else:
    print("❌ ยังไม่เจอ GPU")
    print("เช็คว่า: 1. คุณมีเวอชัน NVIDIA Driver ล่าสุดหรือยัง")
    print("        2. คอมพิวเตอร์ของคุณมีปุ่มการ์ดจอ NVIDIA ใช่ไหม")