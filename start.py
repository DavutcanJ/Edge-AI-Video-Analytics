"""
start.py - Edge AI Video Analytics System Advanced Launcher

Geliştirilmiş özellikler:
1. Ortam kontrolü (required paketler)
2. Mevcut modelleri kontrol et (ONNX/TensorRT varsa adımları atla)
3. İnteraktif menü:
   - Eğitim (isteğe bağlı, Ctrl+C ile çık)
   - ONNX export (varsa atla)
   - TensorRT build (varsa atla)
   - API sunucusu başlat (ayrı process)
4. API'yi ayrı yapıda çalıştır (TensorRT önce yüklenmesin)
"""
import os
import subprocess
import sys
from pathlib import Path
import time


def check_environment():
    """Ortam kontrolü."""
    print("\n[1] Ortam kontrolü...")
    print(f"Python: {sys.version.split()[0]}")
    
    required_packages = ['torch', 'ultralytics', 'onnx', 'onnxruntime', 'tensorrt', 'pycuda', 'fastapi', 'uvicorn']
    missing = []
    
    for pkg in required_packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"[ERROR] Eksik paketler: {', '.join(missing)}")
        return False
    
    print("[OK] Tüm gerekli paketler yüklü.")
    return True


def check_models():
    """Mevcut modelleri kontrol et."""
    print("\n[2] Mevcut modeller kontrol ediliyor...")
    
    pt_exists = Path("models/latest.pt").exists()
    onnx_exists = Path("models/latest.onnx").exists()
    trt_exists = Path("models/model_fp16.engine").exists()
    
    print(f"  PyTorch model (.pt): {'✓' if pt_exists else '✗'}")
    print(f"  ONNX model (.onnx):  {'✓' if onnx_exists else '✗'}")
    print(f"  TensorRT engine:     {'✓' if trt_exists else '✗'}")
    
    return pt_exists, onnx_exists, trt_exists


def ask_user(question: str) -> bool:
    """Kullanıcıya evet/hayır sor."""
    while True:
        response = input(f"\n{question} (y/n): ").lower().strip()
        if response in ('y', 'yes'):
            return True
        elif response in ('n', 'no'):
            return False
        else:
            print("Lütfen 'y' veya 'n' girin.")


def run_training():
    """Model eğitimi çalıştır."""
    train_script = "training/train.py"
    if not Path(train_script).exists():
        print("[INFO] training/train.py bulunamadı.")
        return
    
    print("\n[3a] Model eğitimi başlatılıyor...")
    print("(Eğitim adımını atlamak için Ctrl+C ile çıkabilirsiniz)\n")
    
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
        print("[OK] Eğitim tamamlandı.")
    except KeyboardInterrupt:
        print("\n[INFO] Eğitim adımı atlandı (Ctrl+C).")
    except Exception as e:
        print(f"[ERROR] Eğitim başarısız: {e}")


def export_to_onnx():
    """ONNX'e export et."""
    if Path("models/latest.onnx").exists():
        print("[INFO] ONNX modeli zaten var, atlanıyor.")
        return
    
    if not Path("optimization/export_to_onnx.py").exists():
        print("[INFO] export_to_onnx.py bulunamadı.")
        return
    
    print("\n[3b] ONNX'e export ediliyor...")
    try:
        subprocess.run([
            sys.executable, "optimization/export_to_onnx.py",
            "--model", "models/latest.pt"
        ], check=True)
        print("[OK] ONNX export tamamlandı.")
    except Exception as e:
        print(f"[ERROR] ONNX export başarısız: {e}")


def build_tensorrt_engine():
    """TensorRT engine oluştur."""
    if Path("models/model_fp16.engine").exists():
        print("[INFO] TensorRT engine zaten var, atlanıyor.")
        return
    
    if not Path("optimization/build_trt_engine.py").exists():
        print("[INFO] build_trt_engine.py bulunamadı.")
        return
    
    if not Path("models/latest.onnx").exists():
        print("[ERROR] ONNX modeli yok, TensorRT build yapılamıyor.")
        return
    
    print("\n[3c] TensorRT engine oluşturuluyor (biraz zaman alabilir)...")
    try:
        subprocess.run([
            sys.executable, "optimization/build_trt_engine.py",
            "--onnx", "models/latest.onnx",
            "--precision", "fp16",
            "--batch", "8",
            "--workspace", "4"
        ], check=True)
        print("[OK] TensorRT engine oluşturuldu.")
    except Exception as e:
        print(f"[ERROR] TensorRT build başarısız: {e}")


def start_api_server():
    """API sunucusunu ayrı process'te başlat."""
    if not Path("api/server.py").exists():
        print("[ERROR] api/server.py bulunamadı.")
        return None
    
    print("\n[4] API sunucusu başlatılıyor...")
    print("(http://localhost:8000 üzerinde çalışacak)")
    
    try:
        process = subprocess.Popen(
            [sys.executable, "api/server.py"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Sunucunun başlatılmasını bekle
        time.sleep(2)
        
        if process.poll() is None:  # Process hala çalışıyor
            print("[OK] API sunucusu başlatıldı (PID: {})".format(process.pid))
            print("     Endpoints: /detect, /detect/visualize, /health, /metrics")
            return process
        else:
            stdout, stderr = process.communicate()
            print(f"[ERROR] API başlatılamadı:")
            print(f"  stdout: {stdout}")
            print(f"  stderr: {stderr}")
            return None
    except Exception as e:
        print(f"[ERROR] API başlatma başarısız: {e}")
        return None


def start_gui():
    """GUI uygulamasını başlat (isteğe bağlı)."""
    if not Path("interface/app.py").exists():
        print("[INFO] GUI uygulaması bulunamadı.")
        return None
    
    if not ask_user("CustomTkinter GUI'yi başlatmak istiyor musunuz?"):
        return None
    
    print("\n[5] GUI uygulaması başlatılıyor...")
    try:
        subprocess.Popen([sys.executable, "interface/app.py"])
        print("[OK] GUI başlatıldı.")
        return True
    except Exception as e:
        print(f"[ERROR] GUI başlatılamadı: {e}")
        return None


def run_tests():
    """Test suite'i çalıştır."""
    if not ask_user("Testleri çalıştırmak istiyor musunuz?"):
        return
    
    if not Path("tests/test_inference.py").exists():
        print("[INFO] Test dosyası bulunamadı.")
        return
    
    print("\n[6] Testler çalıştırılıyor...")
    try:
        subprocess.run(["pytest", "tests/test_inference.py", "-v"], check=False)
    except Exception as e:
        print(f"[ERROR] Testler başarısız: {e}")


def main():
    """Ana akış."""
    print("=" * 70)
    print("  Edge AI Video Analytics System - Advanced Launcher")
    print("=" * 70)
    
    # 1. Ortam kontrolü
    if not check_environment():
        sys.exit(1)
    
    # 2. Mevcut modelleri kontrol et
    pt_exists, onnx_exists, trt_exists = check_models()
    
    # 3. İnteraktif menü
    print("\n" + "=" * 70)
    print("  İşlemler")
    print("=" * 70)
    
    if not pt_exists:
        if ask_user("Modeli eğitmek istiyor musunuz?"):
            run_training()
    else:
        print("[INFO] PyTorch model var, eğitim atlanıyor.")
    
    if not onnx_exists:
        if ask_user("ONNX'e export etmek istiyor musunuz?"):
            export_to_onnx()
    else:
        print("[INFO] ONNX modeli var, export atlanıyor.")
    
    if not trt_exists:
        if ask_user("TensorRT engine oluşturmak istiyor musunuz?"):
            build_tensorrt_engine()
    else:
        print("[INFO] TensorRT engine var, build atlanıyor.")
    
    # 4. API sunucusu başlat
    api_process = None
    if ask_user("API sunucusunu başlatmak istiyor musunuz?"):
        api_process = start_api_server()
    
    # 5. GUI başlat (isteğe bağlı)
    start_gui()
    
    # 6. Testleri çalıştır (isteğe bağlı)
    run_tests()
    
    # Sonuç
    print("\n" + "=" * 70)
    if api_process and api_process.poll() is None:
        print("  ✓ Sistem çalışıyor!")
        print("  API: http://localhost:8000")
        print("  Docs: http://localhost:8000/docs")
        print("\n  Sistemden çıkmak için Ctrl+C basın...")
        print("=" * 70)
        
        try:
            api_process.wait()
        except KeyboardInterrupt:
            print("\n\n[INFO] API sunucusu kapatılıyor...")
            api_process.terminate()
            api_process.wait(timeout=5)
            print("[OK] Sistem kapatıldı.")
    else:
        print("  [WARNING] API sunucusu başlatılamadı veya kapatıldı.")
        print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        sys.exit(1)
