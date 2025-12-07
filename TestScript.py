import time
import torch
import psutil
import subprocess

def get_cpu_temp():
    """Return CPU temperature in Celsius (Pi-specific)."""
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        return float(out.replace("temp=", "").replace("'C", ""))
    except:
        return None

def benchmark(model, input_shape=(1, 3, 224, 224), runs=100, device="cpu"):
    print("===================================")
    print("      Raspberry Pi Benchmark")
    print("===================================")

    model.eval().to(device)

    dummy = torch.randn(*input_shape).to(device)

    # Warm-up
    print("Warming up...")
    for _ in range(10):
        _ = model(dummy)

    torch.cuda.synchronize() if device == "cuda" else None

    # Timing
    print("Running benchmark...")
    times = []
    for _ in range(runs):
        start = time.time()
        _ = model(dummy)
        torch.cuda.synchronize() if device == "cuda" else None
        end = time.time()
        times.append(end - start)

    avg = sum(times) / len(times)
    throughput = 1.0 / avg

    print("\n========== RESULTS ==========")
    print(f"Runs:              {runs}")
    print(f"Avg Latency:       {avg*1000:.2f} ms")
    print(f"Throughput:        {throughput:.2f} inferences/sec")
    
    # System information
    print("\n------ System Info ------")
    print(f"CPU Usage:         {psutil.cpu_percent()}%")
    print(f"RAM Usage:         {psutil.virtual_memory().percent}%")

    temp = get_cpu_temp()
    if temp:
        print(f"CPU Temp:          {temp}°C")
    else:
        print("CPU Temp:          (Unavailable)")

    print("============================")


if __name__ == "__main__":
    # EXAMPLE:
    # Replace with your own model or import script
    # Example: loading a quantized model
    model = torch.hub.load("pytorch/vision:v0.10.0", "mobilenet_v2", pretrained=False)

    # If quantized:
    # model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

    benchmark(model, input_shape=(1, 3, 224, 224), runs=50)
