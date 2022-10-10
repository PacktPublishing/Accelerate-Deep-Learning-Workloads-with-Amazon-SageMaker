import torch
import torch_tensorrt
import os

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

MODEL_NAME = "resnet50"
MODEL_VERSION = "1"

device = "cuda" if torch.cuda.is_available() else "cpu"

# load model
model = (
    torch.hub.load("pytorch/vision:v0.10.0", MODEL_NAME, pretrained=True)
    .eval()
    .to(device)
)

# Compile with Torch TensorRT;
trt_model = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input((1, 3, 224, 224))],
    enabled_precisions={torch.half},  # Run with FP32
)

# Save the model
model_dir = os.path.join(os.getcwd(), "3_src", MODEL_NAME, MODEL_VERSION)
os.makedirs(model_dir, exist_ok=True)
print(model_dir)
torch.jit.save(trt_model, os.path.join(model_dir, "model.pt"))
