"""
start.py - Edge AI Video Analytics System Launcher

Basit launcher - Uygulamayı başlatır.
Tüm yönetim işlemleri GUI üzerinden yapılır.
"""
import os
import subprocess
import sys
from pathlib import Path


def check_environment():
    """Basit ortam kontrolü."""
    print("\n[INFO] Ortam kontrol ediliyor...")
    print(f"  Python: {sys.version.split()[0]}")
    
    # Temel paketleri kontrol et
    required = ['torch', 'ultralytics', 'cv2', 'customtkinter']
    missing = []
    
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    
    if missing:
        print(f"[WARNING] Eksik paketler: {', '.join(missing)}")
        print("          Bazı özellikler çalışmayabilir.")
        return False
    
    print("  ✓ Temel paketler hazır")
    
    # GPU kontrolü
    try:
        import torch
        if torch.cuda.is_available():
            print(f"  ✓ GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("  ⚠ GPU bulunamadı - CPU modunda çalışacak")
    except:
        pass
    
    return True


def start_gui():
    """GUI uygulamasını başlat."""
    gui_script = Path("run_gui.py")
    
    if not gui_script.exists():
        print("[ERROR] run_gui.py bulunamadı!")
        return False
    
    print("\n[INFO] GUI başlatılıyor...")
    print("  Tüm yönetim işlemleri GUI üzerinden yapılabilir:")
    print("  • API Server yönetimi")
    print("  • Model eğitimi")
    print("  • ONNX/TensorRT export")
    print("  • Performance testleri")
    print("  • Image detection & Webcam tracking")
    print("  • Monitoring & Metrics")
    
    try:
        subprocess.run([sys.executable, str(gui_script)], check=True)
        return True
    except KeyboardInterrupt:
        print("\n[INFO] Kullanıcı tarafından durduruldu.")
        return True
    except Exception as e:
        print(f"[ERROR] GUI başlatılamadı: {e}")
        return False


def start_api_server():
    """API sunucusunu başlat (sadece API modu)."""
    api_script = Path("api/server.py")
    
    if not api_script.exists():
        print("[ERROR] api/server.py bulunamadı!")
        return False
    
    # Port'u environment'tan al, yoksa default 8000
    api_port = os.getenv("API_PORT", "8000")
    
    print("\n[INFO] API Server başlatılıyor...")
    print(f"  API Endpoints: http://localhost:{api_port}")
    print(f"  Docs: http://localhost:{api_port}/docs")
    print("  (Durdurmak için Ctrl+C)")
    
    try:
        subprocess.run([sys.executable, str(api_script)], check=True)
        return True
    except KeyboardInterrupt:
        print("\n[INFO] API server durduruldu.")
        return True
    except Exception as e:
        print(f"[ERROR] API server başlatılamadı: {e}")
        return False


def main():
    """Ana launcher."""
    print("=" * 70)
    print("  🚀 Edge AI Video Analytics System")
    print("=" * 70)
    
    # Ortam kontrolü
    if not check_environment():
        print("\n[WARNING] Bazı paketler eksik, devam ediliyor...")
    
    # Basit menü
    print("\n" + "=" * 70)
    print("  Başlatma Seçenekleri")
    print("=" * 70)
    print("\n  1) 🖥️  GUI Uygulaması (Önerilen)")
    print("     → Tüm özellikler GUI'den yönetilebilir")
    print("     → API, Training, Export, Test, Monitoring")
    print("     → Image Detection & Webcam Tracking")
    print("\n  2) 🌐 Sadece API Server")
    print("     → Backend API'yi başlatır")
    print("     → GUI olmadan kullanım için")
    print("\n  3) 🔧 Her İkisi (GUI + API)")
    print("     → API ve GUI'yi birlikte başlatır")
    print("\n  0) ❌ Çıkış")
    print("=" * 70)
    
    choice = input("\n  Seçiminiz (0-3) [1]: ").strip() or "1"
    
    if choice == "0":
        print("\n[INFO] Çıkılıyor...")
        return
    
    elif choice == "1":
        # Sadece GUI
        start_gui()
    
    elif choice == "2":
        # Sadece API
        start_api_server()
    
    elif choice == "3":
        # Her ikisi
        print("\n[INFO] API ve GUI başlatılıyor...")
        
        # Port'u environment'tan al
        api_port = os.getenv("API_PORT", "8000")
        
        # API'yi arka planda başlat
        api_script = Path("api/server.py")
        if api_script.exists():
            try:
                api_process = subprocess.Popen(
                    [sys.executable, str(api_script)],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
                print(f"  ✓ API server başlatıldı (arka planda - Port: {api_port})")
                print("  ✓ GUI açılıyor...")
                
                # GUI'yi başlat
                start_gui()
                
                # GUI kapandığında API'yi de kapat
                if api_process.poll() is None:
                    print("\n[INFO] API server kapatılıyor...")
                    api_process.terminate()
                    api_process.wait(timeout=5)
                    print("  ✓ API server kapatıldı")
            except Exception as e:
                print(f"[ERROR] Başlatma hatası: {e}")
        else:
            print("[ERROR] api/server.py bulunamadı!")
            start_gui()
    
    else:
        print("\n[ERROR] Geçersiz seçim!")
        return


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n[INFO] Uygulama kapatılıyor...")
    except Exception as e:
        print(f"\n[ERROR] Fatal error: {e}")
        sys.exit(1)
