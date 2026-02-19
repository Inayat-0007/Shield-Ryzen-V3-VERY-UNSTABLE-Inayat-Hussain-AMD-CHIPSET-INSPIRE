
import cv2
import yaml
import os
import sys
import time
from v3_int8_engine import ShieldEngine

def list_cameras():
    print("[INIT] Scanning for connected cameras (Index 0-3)...")
    available = []
    # Check first 4 indices (DSHOW is faster but still takes a moment)
    for i in range(4):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                available.append(i)
            cap.release()
    return available

def run_live():
    # Load config
    config_path = "config.yaml"
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    # Select Camera
    cameras = list_cameras()
    if not cameras:
        print("[ERROR] No cameras found! Check connection.")
        return

    print(f"\n[CAM] Available Camera Indices: {cameras}")
    selected_cam = 0
    if len(cameras) > 1:
        start_t = time.time()
        # Non-blocking input is hard in standard python without curses/external libs.
        # We will block and wait for user.
        try:
            choice = input(f">> Select Camera ID to use {cameras} (Press Enter for Default {cameras[0]}): ")
            if choice.strip() == "":
                selected_cam = cameras[0]
            else:
                selected_cam = int(choice)
        except ValueError:
            print("[WARN] Invalid input. Using default.")
            selected_cam = cameras[0]
    elif cameras:
        selected_cam = cameras[0]

    print(f"[INIT] Using Camera: {selected_cam}\n")
    config["camera_id"] = selected_cam
    # Use the optimized INT8 model
    config["model_path"] = "shield_ryzen_int8.onnx"
    
    try:
        print("[SHIELD] Initializing Shield-Ryzen V2 Engine...")
        engine = ShieldEngine(config)
        print("[OK] Engine Ready. Opening Webcam Window...")
        print(">> Press 'q' to quit.")
        
        # Open window
        cv2.namedWindow("Shield-Ryzen V2 Live", cv2.WINDOW_NORMAL)
        
        frame_count = 0
        while True:
            result = engine.process_frame()
            
            if result.frame is not None:
                # Show frame with HUD
                cv2.imshow("Shield-Ryzen V2 Live", result.frame)
            
            frame_count += 1
            # Console Log (Throttled or removed for performance)
            if result.face_results:
                f = result.face_results[0]
                print(f"[{frame_count:04d}] STATE: {f.state:<10} | NEURAL: {f.neural_confidence:.2f} | FPS: {result.fps:.1f} | EAR: {f.ear_value:.2f} | TEX: {f.texture_score:.2f} | OCC: {f.occlusion_score:.2f}")
            elif frame_count % 30 == 0:
                print(f"[{frame_count:04d}] NO FACE DETECTED (FPS: {result.fps:.1f})")
            
            # Check for 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"[ERROR] Error during live run: {e}")
    finally:
        if 'engine' in locals():
            engine.release()
        cv2.destroyAllWindows()
        print("[EXIT] Live Demo Closed.")

if __name__ == "__main__":
    run_live()
