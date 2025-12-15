import os
import argparse
import numpy as np
import onnxruntime as ort
from torchvision import datasets, transforms


def eval_onnx_accuracy(model_path):
    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    test_ds = datasets.CIFAR10(root="data", train=False, download=True, transform=tfm)

    sess = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    correct = 0
    total = 0

    for x, y in test_ds:
        x_np = x.unsqueeze(0).numpy().astype(np.float32)  

        out = sess.run(None, {input_name: x_np})[0]       
        pred = int(np.argmax(out, axis=1)[0])

        correct += int(pred == y)
        total += 1

    return correct / total


def main(models_dir, model_list):
    models = model_list
    if models is None:
        models = [f for f in sorted(os.listdir(models_dir)) if f.endswith(".onnx")]

    if not models:
        raise RuntimeError(f"No ONNX models found. models_dir={models_dir}")

    print("Model,Accuracy")
    for m in models:
        path = m if os.path.isabs(m) else os.path.join(models_dir, m)
        acc = eval_onnx_accuracy(path)
        print(f"{os.path.basename(path)},{acc:.4f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", default="Models/Baseline", help="Directory containing ONNX models")
    ap.add_argument("--models", nargs="*", default=None, help="Optional explicit list of model filenames/paths")
    args = ap.parse_args()

    main(args.models_dir, args.models)
