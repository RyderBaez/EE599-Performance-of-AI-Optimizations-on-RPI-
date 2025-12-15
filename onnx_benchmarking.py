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
    """Run a timed inference stage and collect latency + temperature trace."""
    times = []
    temp_trace = []

    for _ in range(runs):
        start = time.perf_counter()
        sess.run(None, {input_name: dummy})
        end = time.perf_counter()
        times.append(end - start)

        t = get_cpu_temp()
        temp_trace.append(t)

        if sleep_s > 0:
            time.sleep(sleep_s)

    avg_latency_ms = float(np.mean(times) * 1000.0)
    fps = float(1000.0 / avg_latency_ms) if avg_latency_ms > 0 else float("inf")

    temps_valid = [t for t in temp_trace if t is not None]
    temp_avg = float(np.mean(temps_valid)) if temps_valid else None
    temp_max = float(np.max(temps_valid)) if temps_valid else None

    return avg_latency_ms, fps, temp_avg, temp_max, temp_trace


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


def temp_overlay_plot(filename, title, models, temp_traces):
    plt.figure()
    any_temp = False
    for m in models:
        trace = temp_traces.get(m, [])
        if trace and any(t is not None for t in trace):
            y = [t if t is not None else np.nan for t in trace]
            plt.plot(y, label=m)
            any_temp = True

    plt.xlabel("Inference iteration")
    plt.ylabel("CPU Temperature (°C)")
    plt.title(title)
    if any_temp:
        plt.legend()
    else:
        plt.text(
            0.5, 0.5,
            "CPU temperature unavailable (vcgencmd not found)",
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

    for fname in onnx_files:
        path = os.path.join(models_dir, fname)
        print(f"Benchmarking: {fname}")

        sess = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
        input_name = sess.get_inputs()[0].name
        dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)

        for _ in range(warmup):
            sess.run(None, {input_name: dummy})

        # Stage 50
        lat50, fps50, tavg50, tmax50, trace50 = run_stage(sess, input_name, dummy, runs_short, sleep_s)
        # Stage 200
        lat200, fps200, tavg200, tmax200, trace200 = run_stage(sess, input_name, dummy, runs_long, sleep_s)

        cpu_now = psutil.cpu_percent()
        ram_now = psutil.virtual_memory().percent

        results_50.append({
            "model": fname,
            "latency_ms": lat50,
            "fps": fps50,
            "cpu_percent": cpu_now,
            "ram_percent": ram_now,
            "temp_avg_c": tavg50,
            "temp_max_c": tmax50,
        })
        results_200.append({
            "model": fname,
            "latency_ms": lat200,
            "fps": fps200,
            "cpu_percent": cpu_now,
            "ram_percent": ram_now,
            "temp_avg_c": tavg200,
            "temp_max_c": tmax200,
        })

        temp_traces_50[fname] = trace50
        temp_traces_200[fname] = trace200

        def fmt(x):
            return "NA" if x is None else f"{x:.2f}"

        print(f"  [50 ] Avg Latency: {lat50:.2f} ms | FPS: {fps50:.2f} | Temp(avg/max): {fmt(tavg50)}/{fmt(tmax50)} °C")
        print(f"  [200] Avg Latency: {lat200:.2f} ms | FPS: {fps200:.2f} | Temp(avg/max): {fmt(tavg200)}/{fmt(tmax200)} °C\n")

    fieldnames = ["model", "latency_ms", "fps", "cpu_percent", "ram_percent", "temp_avg_c", "temp_max_c"]

    # Save CSVs
    save_csv("benchmark_results_50.csv", results_50, fieldnames)
    save_csv("benchmark_results_200.csv", results_200, fieldnames)

    # Plot bars (50)
    models = [r["model"] for r in results_50]
    lat50_vals = [r["latency_ms"] for r in results_50]
    fps50_vals = [r["fps"] for r in results_50]
    bar_plot("latency_plot_50.png", "ONNX Runtime Inference Latency (50-run avg)", "Latency (ms)", models, lat50_vals)
    bar_plot("fps_plot_50.png", "ONNX Runtime Throughput (50-run avg)", "Throughput (FPS)", models, fps50_vals)
    temp_overlay_plot("temperature_plot_50.png", "Temperature During Inference (50 runs)", models, temp_traces_50)

    # Plot bars (200)
    models200 = [r["model"] for r in results_200]
    lat200_vals = [r["latency_ms"] for r in results_200]
    fps200_vals = [r["fps"] for r in results_200]
    bar_plot("latency_plot_200.png", "ONNX Runtime Inference Latency (200-run avg)", "Latency (ms)", models200, lat200_vals)
    bar_plot("fps_plot_200.png", "ONNX Runtime Throughput (200-run avg)", "Throughput (FPS)", models200, fps200_vals)
    temp_overlay_plot("temperature_plot_200.png", "Temperature During Inference (200 runs)", models200, temp_traces_200)

    print("\nDone. Generated CSVs and plots for both 50 and 200 runs.")


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
