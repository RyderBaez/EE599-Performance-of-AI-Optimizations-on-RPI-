import argparse
import os
import numpy as np

import onnx
import onnxruntime as ort
from onnxruntime.quantization import (
    quantize_static,
    CalibrationDataReader,
    QuantType,
    QuantFormat,
)

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def get_onnx_input_name(onnx_path: str) -> str:
    """Best-effort: pick the first real graph input (not an initializer)."""
    model = onnx.load(onnx_path)
    graph = model.graph

    initializer_names = set(init.name for init in graph.initializer)
    for inp in graph.input:
        if inp.name not in initializer_names:
            return inp.name
          
    return graph.input[0].name


class CIFAR10DataReader(CalibrationDataReader):
    def __init__(self, input_name: str, batch_size: int = 1, num_batches: int = 200, data_root: str = "data"):
        self.input_name = input_name
        self.batch_size = batch_size
        self.num_batches = num_batches

        tfm = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ])

        ds = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tfm)
        self.loader = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=0)

        self._iter = iter(self.loader)
        self._count = 0

    def get_next(self):
        if self._count >= self.num_batches:
            return None

        try:
            x, _ = next(self._iter)
        except StopIteration:
            self._iter = iter(self.loader)
            x, _ = next(self._iter)

        x_np = x.numpy().astype(np.float32)
        self._count += 1
        return {self.input_name: x_np}


def sanity_run_onnx(onnx_path: str, input_name: str):
    """Quick check that ORT can load and run one inference."""
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    dummy = np.random.randn(1, 3, 224, 224).astype(np.float32)
    _ = sess.run(None, {input_name: dummy})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="onnx_in", required=True, help="Path to FP32 ONNX model (e.g., baseline_fp32.onnx)")
    ap.add_argument("--out", dest="onnx_out", default=None, help="Output path for INT8 ONNX (default: <in>_int8.onnx)")
    ap.add_argument("--calib-batches", type=int, default=200, help="Number of calibration batches (default 200)")
    ap.add_argument("--batch-size", type=int, default=1, help="Calibration batch size (default 1)")
    ap.add_argument("--data-root", type=str, default="data", help="Dataset root (default: data)")
    args = ap.parse_args()

    onnx_in = args.onnx_in
    if not os.path.exists(onnx_in):
        raise FileNotFoundError(f"Input ONNX not found: {onnx_in}")

    onnx_out = args.onnx_out
    if onnx_out is None:
        base, ext = os.path.splitext(onnx_in)
        onnx_out = base + "_int8" + ext

    input_name = get_onnx_input_name(onnx_in)
    print(f"[INFO] Input model: {onnx_in}")
    print(f"[INFO] Output model: {onnx_out}")
    print(f"[INFO] Detected input name: {input_name}")

    print("[INFO] Sanity-checking FP32 ONNX inference...")
    sanity_run_onnx(onnx_in, input_name)
    print("[OK] FP32 model runs in ONNX Runtime.")

    dr = CIFAR10DataReader(
        input_name=input_name,
        batch_size=args.batch_size,
        num_batches=args.calib_batches,
        data_root=args.data_root,
    )

    print("[INFO] Quantizing (static INT8 PTQ) ...")
    quantize_static(
        model_input=onnx_in,
        model_output=onnx_out,
        calibration_data_reader=dr,
        quant_format=QuantFormat.QOperator,  
        activation_type=QuantType.QUInt8,
        weight_type=QuantType.QInt8,
    )

    print("[INFO] Sanity-checking INT8 ONNX inference...")
    sanity_run_onnx(onnx_out, input_name)
    print("[OK] INT8 model runs. Done.")


if __name__ == "__main__":
    main()
