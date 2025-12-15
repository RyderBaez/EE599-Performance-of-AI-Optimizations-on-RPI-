import time
import os
import csv
import argparse
import numpy as np
import psutil
import subprocess
import onnxruntime as ort
import matplotlib.pyplot as plt


def get_cpu_temp():
    """Return CPU temperature in Celsius (Pi-specific)."""
    try:
        out = subprocess.check_output(["vcgencmd", "measure_temp"]).decode()
        return float(out.replace("temp=", "").replace("'C", ""))
    except Exception:
        return None


def run_stage(sess, input_name, dummy, runs, sleep_s=0.0):
    """
    Run a timed inference stage and collect:
    - latency list
    - temp trace (if available)
    - CPU utilization trace
    Returns averages + maxes + traces.
    """
    times = []
    temp_trace = []
    cpu_trace = []

    psutil.cpu_percent(interval=None)

    for _ in range(runs):
        start = time.perf_counter()
        sess.run(None, {input_name: dummy})
        end = time.perf_counter()
        times.append(end - start)

        temp_trace.append(get_cpu_temp())
        cpu_trace.append(psutil.cpu_percent(interval=None))

        if sleep_s > 0:
            time.sleep(sleep_s)

    avg_latency_ms = float(np.mean(times) * 1000.0)
    fps = float(1000.0 / avg_latency_ms) if avg_latency_ms > 0 else float("inf")

    temps_valid = [t for t in temp_trace if t is not None]
    temp_avg = float(np.mean(temps_valid)) if temps_valid else None
    temp_max = float(np.max(temps_valid)) if temps_valid else None

    cpu_avg = float(np.mean(cpu_trace)) if cpu_trace else None
    cpu_max = float(np.max(cpu_trace)) if cpu_trace else None

    return avg_latency_ms, fps, temp_avg, temp_max, temp_trace, cpu_avg, cpu_max, cpu_trace


def save_csv(path, results, fieldnames):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)
    print(f"Saved results to {path}")


def bar_plot(filename, title, ylabel, models, values):
    plt.figure()
    plt.bar(models, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")


def overlay_plot(filename, title, xlabel, ylabel, models, traces_dict):
    """Generic overlay plot for traces (e.g., temperature, CPU%)."""
    plt.figure()
    any_data = False
    for m in models:
        trace = traces_dict.get(m, [])
        if trace:
            y = [v if v is not None else np.nan for v in trace]
            if any(np.isfinite(yv) for yv in y):
                plt.plot(y, label=m)
                any_data = True

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    if any_data:
        plt.legend()
    else:
        plt.text(
            0.5, 0.5,
            "Data unavailable",
            ha="center", va="center",
            transform=plt.gca().transAxes,
        )
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved {filename}")


def main(models_dir, runs_short, runs_long, warmup, threads, sleep_s):
    so = ort.SessionOptions()
    if threads is not None:
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1

    onnx_files = sorted([f for f in os.listdir(models_dir) if f.endswith(".onnx")])
    if not onnx_files:
        raise RuntimeError(f"No .onnx files found in: {models_dir}")

    print("===================================")
    print("   Raspberry Pi ONNX Benchmark All")
    print("===================================")
    print(f"Models dir: {models_dir}")
    print(f"Warmup: {warmup} | Runs(50): {runs_short} | Runs(200): {runs_long} | Threads: {threads} | Sleep: {sleep_s}s")
    print("===================================\n")

    results_50 = []
    results_200 = []

    temp_traces_50 = {}
    temp_traces_200 = {}

    cpu_traces_50 = {}
    cpu_traces_200 = {}

    for fname in onnx_files:
        path = os.path.join(models_dir, fname)
        print(f"Benchmarking: {fname}")

        sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)

        for _ in range(warmup):
            sess.run(None, {input_name: dummy})

        lat50, fps50, tavg50, tmax50, traceT50, cpuavg50, cpumax50, traceC50 = run_stage(
            sess, input_name, dummy, runs_short, sleep_s
        )
        
        lat200, fps200, tavg200, tmax200, traceT200, cpuavg200, cpumax200, traceC200 = run_stage(
            sess, input_name, dummy, runs_long, sleep_s
        )

        ram_now = psutil.virtual_memory().percent  

        results_50.append({
            "model": fname,
            "latency_ms": lat50,
            "fps": fps50,
            "cpu_avg_percent": cpuavg50,
            "cpu_max_percent": cpumax50,
            "ram_percent": ram_now,
            "temp_avg_c": tavg50,
            "temp_max_c": tmax50,
        })
        results_200.append({
            "model": fname,
            "latency_ms": lat200,
            "fps": fps200,
            "cpu_avg_percent": cpuavg200,
            "cpu_max_percent": cpumax200,
            "ram_percent": ram_now,
            "temp_avg_c": tavg200,
            "temp_max_c": tmax200,
        })

        temp_traces_50[fname] = traceT50
        temp_traces_200[fname] = traceT200
        cpu_traces_50[fname] = traceC50
        cpu_traces_200[fname] = traceC200

        def fmt(x):
            return "NA" if x is None else f"{x:.2f}"

        print(f"  [50 ] Latency: {lat50:.2f} ms | FPS: {fps50:.2f} | CPU(avg/max): {fmt(cpuavg50)}/{fmt(cpumax50)}% | Temp(avg/max): {fmt(tavg50)}/{fmt(tmax50)} °C")
        print(f"  [200] Latency: {lat200:.2f} ms | FPS: {fps200:.2f} | CPU(avg/max): {fmt(cpuavg200)}/{fmt(cpumax200)}% | Temp(avg/max): {fmt(tavg200)}/{fmt(tmax200)} °C\n")

    fieldnames = [
        "model", "latency_ms", "fps",
        "cpu_avg_percent", "cpu_max_percent",
        "ram_percent",
        "temp_avg_c", "temp_max_c"
    ]

    save_csv("benchmark_results_50.csv", results_50, fieldnames)
    save_csv("benchmark_results_200.csv", results_200, fieldnames)

    models = [r["model"] for r in results_50]
    lat50_vals = [r["latency_ms"] for r in results_50]
    fps50_vals = [r["fps"] for r in results_50]
    cpuavg50_vals = [r["cpu_avg_percent"] if r["cpu_avg_percent"] is not None else 0.0 for r in results_50]

    bar_plot("latency_plot_50.png", "ONNX Runtime Inference Latency (50-run avg)", "Latency (ms)", models, lat50_vals)
    bar_plot("fps_plot_50.png", "ONNX Runtime Throughput (50-run avg)", "Throughput (FPS)", models, fps50_vals)
    bar_plot("cpu_avg_plot_50.png", "CPU Utilization (50-run avg)", "CPU Utilization (%)", models, cpuavg50_vals)

    overlay_plot("temperature_plot_50.png", "Temperature During Inference (50 runs)", "Inference iteration", "CPU Temperature (°C)", models, temp_traces_50)
    overlay_plot("cpu_trace_plot_50.png", "CPU Utilization Trace (50 runs)", "Inference iteration", "CPU Utilization (%)", models, cpu_traces_50)

    models200 = [r["model"] for r in results_200]
    lat200_vals = [r["latency_ms"] for r in results_200]
    fps200_vals = [r["fps"] for r in results_200]
    cpuavg200_vals = [r["cpu_avg_percent"] if r["cpu_avg_percent"] is not None else 0.0 for r in results_200]

    bar_plot("latency_plot_200.png", "ONNX Runtime Inference Latency (200-run avg)", "Latency (ms)", models200, lat200_vals)
    bar_plot("fps_plot_200.png", "ONNX Runtime Throughput (200-run avg)", "Throughput (FPS)", models200, fps200_vals)
    bar_plot("cpu_avg_plot_200.png", "CPU Utilization (200-run avg)", "CPU Utilization (%)", models200, cpuavg200_vals)

    overlay_plot("temperature_plot_200.png", "Temperature During Inference (200 runs)", "Inference iteration", "CPU Temperature (°C)", models200, temp_traces_200)
    overlay_plot("cpu_trace_plot_200.png", "CPU Utilization Trace (200 runs)", "Inference iteration", "CPU Utilization (%)", models200, cpu_traces_200)

    print("\nDone. Generated CSVs and plots for both 50 and 200 runs (including CPU utilization).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", default="Models/Baseline", help="Directory containing ONNX models")
    ap.add_argument("--runs_short", type=int, default=50, help="Short benchmark runs (default 50)")
    ap.add_argument("--runs_long", type=int, default=200, help="Long benchmark runs (default 200)")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--threads", type=int, default=None, help="Set intra-op threads (optional)")
    ap.add_argument("--sleep", type=float, default=0.0, help="Sleep seconds between runs (optional)")
    args = ap.parse_args()

    main(args.models_dir, args.runs_short, args.runs_long, args.warmup, args.threads, args.sleep)
