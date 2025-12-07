import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import os

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "rpi_models"
os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------------
# DATA
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=transform)
test_dataset  = datasets.CIFAR10(root="data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader  = DataLoader(test_dataset, batch_size=64, shuffle=False)

# ----------------------------
# HELPER FUNCTIONS
# ----------------------------
def evaluate(model):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            out = model(x)
            pred = out.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total

def export_to_onnx(model, name, input_shape=(1,3,224,224)):
    model.eval()
    os.makedirs(SAVE_DIR, exist_ok=True)
    dummy_input = torch.randn(*input_shape)
    onnx_path = os.path.join(SAVE_DIR, name + ".onnx")
    torch.onnx.export(
        model.to("cpu"),
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0:'batch_size'}, 'output': {0:'batch_size'}}
    )
    print(f"[ONNX] Saved: {onnx_path}")

# ----------------------------
# TRAIN LOOP
# ----------------------------
def train_model(model, epochs=5):
    model = model.to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        running_loss = 0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        acc = evaluate(model)
        print(f"Epoch {epoch+1}: loss={running_loss/len(train_loader):.4f}, test_acc={acc:.4f}")

    return model

# ----------------------------
# MODEL VARIANTS
# ----------------------------
def prune_model(model, amount=0.3):
    for m in model.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            prune.l1_unstructured(m, "weight", amount=amount)
            prune.remove(m, "weight")
    return model

def quantize_model(model):
    return torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

def train_distilled(student, teacher, epochs=5):
    teacher.eval()
    optimizer = optim.Adam(student.parameters(), lr=1e-3)
    alpha, T = 0.5, 4.0
    for epoch in range(epochs):
        student.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            s_out = student(x)
            with torch.no_grad():
                t_out = teacher(x)
            loss = alpha*F.cross_entropy(s_out, y) + (1-alpha)*F.kl_div(
                F.log_softmax(s_out/T, dim=1),
                F.softmax(t_out/T, dim=1),
                reduction="batchmean"
            )*(T*T)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        acc = evaluate(student)
        print(f"[Distilled] Epoch {epoch+1}: test_acc={acc:.4f}")
    return student

# ----------------------------
# MAIN TRAINING & EXPORT
# ----------------------------
if __name__ == "__main__":
    # Baseline FP32
    baseline = train_model(models.mobilenet_v2(num_classes=10))
    torch.save(baseline.state_dict(), os.path.join(SAVE_DIR, "baseline_fp32.pt"))
    export_to_onnx(baseline, "baseline_fp32")  # MANDATORY step for RPi

    # Quantized
    quant = quantize_model(baseline)
    torch.save(quant.state_dict(), os.path.join(SAVE_DIR, "quantized_int8.pt"))
    export_to_onnx(quant, "quantized_int8")

    # Pruned
    pruned = prune_model(models.mobilenet_v2(num_classes=10))
    pruned = train_model(pruned)
    torch.save(pruned.state_dict(), os.path.join(SAVE_DIR, "pruned_30.pt"))
    export_to_onnx(pruned, "pruned_30")

    # Distilled
    student = models.mobilenet_v2(num_classes=10)
    distilled = train_distilled(student, baseline)
    torch.save(distilled.state_dict(), os.path.join(SAVE_DIR, "distilled.pt"))
    export_to_onnx(distilled, "distilled")

    print("All models trained, saved, and exported to ONNX for Raspberry Pi!")
