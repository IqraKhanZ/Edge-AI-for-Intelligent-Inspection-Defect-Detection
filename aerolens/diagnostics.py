import os
import sys
import platform

def run_diagnostics():
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "telemetry")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "hardware_check.log")
    
    diagnostic_info = []
    diagnostic_info.append("=== AeroLens Sentinel System Diagnostics ===")
    diagnostic_info.append(f"OS: {platform.system()} {platform.release()} ({platform.machine()})")
    diagnostic_info.append(f"Python Version: {sys.version}")
    
    # Check ONNX Runtime & Execution Providers
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        diagnostic_info.append("ONNX Runtime: Installed")
        diagnostic_info.append(f"Available Providers: {', '.join(providers)}")
        if 'CUDAExecutionProvider' in providers:
            diagnostic_info.append("  - CUDA Support: AVAILABLE")
        if 'TensorrtExecutionProvider' in providers:
            diagnostic_info.append("  - TensorRT Support: AVAILABLE")
    except ImportError:
        diagnostic_info.append("ONNX Runtime: NOT Installed")
        providers = []
        
    # Check OpenVINO
    try:
        from openvino.runtime import Core
        ie = Core()
        devices = ie.available_devices
        diagnostic_info.append("OpenVINO: Installed")
        diagnostic_info.append(f"Available Devices: {', '.join(devices)}")
    except ImportError:
        diagnostic_info.append("OpenVINO: NOT Installed")
        
    # Check OpenCV
    try:
        import cv2
        diagnostic_info.append(f"OpenCV: Installed (Version: {cv2.__version__})")
        build_info = cv2.getBuildInformation()
        if "CUDA" in build_info and "YES" in build_info.split("CUDA")[1].split("\n")[0]:
            diagnostic_info.append("  - OpenCV CUDA backend: SUPPORTED")
        else:
            diagnostic_info.append("  - OpenCV CUDA backend: UNSUPPORTED")
    except ImportError:
        diagnostic_info.append("OpenCV: NOT Installed")

    # Check PyTorch
    try:
        import torch
        diagnostic_info.append(f"PyTorch: Installed (Version: {torch.__version__})")
        if torch.cuda.is_available():
            diagnostic_info.append(f"  - CUDA Available: YES (Device: {torch.cuda.get_device_name(0)})")
        else:
            diagnostic_info.append("  - CUDA Available: NO")
    except ImportError:
        diagnostic_info.append("PyTorch: NOT Installed")

    # Write log
    log_content = "\n".join(diagnostic_info)
    with open(log_path, "w") as f:
        f.write(log_content)
    
    print(log_content)
    print(f"\nDiagnostics log written to: {log_path}")

if __name__ == "__main__":
    run_diagnostics()
