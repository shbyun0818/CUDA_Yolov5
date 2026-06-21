from pathlib import Path
import torch

weights = Path(__file__).resolve().parent.parent / "model" / "yolov5s.pt"
onnx_path = str(Path(weights).with_suffix(".onnx"))

# PyTorch 2.6 changed torch.load() to weights_only=True by default.
# YOLOv5 checkpoints require loading the full checkpoint object.
# Use this only for checkpoint files from a trusted source.
model = torch.load(weights, map_location="cpu", weights_only=False)
model = model["model"].float().eval()

dummy = torch.zeros(1, 3, 640, 640)

torch.onnx.export(
    model,
    dummy,
    onnx_path,
    opset_version=18,
    input_names=["images"],
    output_names=["output"],
    dynamic_axes={
        "images": {0: "batch"},
        "output": {0: "batch"},
    },
)

print(f"saved: {onnx_path}")
