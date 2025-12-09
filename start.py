"""
start.py - Edge AI Video Analytics System Quick Launcher (Güncel argümanlarla)

Bu script, projenin temel adımlarını sırasıyla çalıştırır:
1. Ortam kontrolü
2. Eğitim (isteğe bağlı)
3. Model optimizasyonu (ONNX/TensorRT)
4. API sunucusunu başlatma
5. İzleme ve test
"""
import os
import subprocess
import sys

# 1. Ortam kontrolü
print("\n[1] Ortam kontrolü...")
print(f"Python: {sys.version}")
try:
    import torch, ultralytics, onnx, onnxruntime, tensorrt, pycuda
    print("[OK] Tüm ana paketler yüklü.")
except ImportError as e:
    print(f"[ERROR] Eksik paket: {e.name}")
    sys.exit(1)

# 2. Eğitim (isteğe bağlı)
train_script = "training/train.py"
if os.path.exists(train_script):
    print("\n[2] Model eğitimi başlatılıyor...")
    print("(Eğitim adımını atlamak için Ctrl+C ile geçebilirsiniz)")
    try:
        subprocess.run([
            sys.executable, train_script,
            "--config", "training/dataset.yaml",
            "--model", "yolov8n.pt",
            "--epochs", "1",
            "--batch", "8",
            "--imgsz", "640",
            "--device", "0"
        ], check=True)
    except KeyboardInterrupt:
        print("[INFO] Eğitim adımı atlandı.")
else:
    print("[INFO] training/train.py bulunamadı, eğitim adımı atlanıyor.")

# 3. Model optimizasyonu
print("\n[3] Model optimizasyonu (ONNX/TensorRT)...")
if os.path.exists("optimization/export_to_onnx.py"):
    subprocess.run([sys.executable, "optimization/export_to_onnx.py", "--model", "runs/detect/train/weights/best.pt"], check=False)
if os.path.exists("optimization/calibrate_int8.py"):
    subprocess.run([sys.executable, "optimization/calibrate_int8.py", "--data-dir", "data/train/images", "--output", "models/calibration.cache"], check=False)
if os.path.exists("optimization/build_trt_engine.py"):
    subprocess.run([sys.executable, "optimization/build_trt_engine.py", "--onnx", "models/model.onnx", "--precision", "int8", "--calibration-cache", "models/calibration.cache"], check=False)

# 4. API sunucusunu başlatma
print("\n[4] FastAPI sunucusu başlatılıyor...")
if os.path.exists("api/server.py"):
    subprocess.Popen([sys.executable, "api/server.py"])
    print("[INFO] API sunucusu arka planda başlatıldı.")
else:
    print("[ERROR] api/server.py bulunamadı.")

# 5. İzleme ve test
print("\n[5] İzleme ve test...")
if os.path.exists("monitoring/dashboard.py"):
    subprocess.Popen([sys.executable, "monitoring/dashboard.py"])
    print("[INFO] Performans dashboard arka planda başlatıldı.")
else:
    print("[INFO] monitoring/dashboard.py bulunamadı.")

if os.path.exists("tests/test_inference.py"):
    print("[INFO] Testler çalıştırılıyor...")
    subprocess.run(["pytest", "tests/test_inference.py"], check=False)
else:
    print("[INFO] Test dosyası bulunamadı.")

print("\n[OK] Tüm ana adımlar tamamlandı. Sistemi kullanmaya başlayabilirsiniz!")
