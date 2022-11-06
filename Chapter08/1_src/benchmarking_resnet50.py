# This is adoption of TensorRT notebook: https://pytorch.org/TensorRT/_notebooks/Resnet50-example.html


import json
import time

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch_tensorrt
from PIL import Image
from torchvision import transforms

torch.hub._validate_not_a_forked_repo = lambda a, b, c: True

resnet50_model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)
resnet50_model.eval()

print("Resnet50 backbone")
print(resnet50_model)


for i in range(4):
    img_path = "./data/img%d.JPG" % i
    img = Image.open(img_path)
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(img)

# loading labels
with open("./data/imagenet_class_index.json") as json_file:
    d = json.load(json_file)


cudnn.benchmark = True


def rn50_preprocess():
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return preprocess


# decode the results into ([predicted class, description], probability)
def predict(img_path, model):
    img = Image.open(img_path)
    preprocess = rn50_preprocess()
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(
        0
    )  # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        model.to("cuda")

    with torch.no_grad():
        output = model(input_batch)
        # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
        sm_output = torch.nn.functional.softmax(output[0], dim=0)

    ind = torch.argmax(sm_output)
    return (
        d[str(ind.item())],
        sm_output[ind],
    )  # ([predicted class, description], probability)


def benchmark(
    model, input_shape=(1024, 1, 224, 224), dtype="fp32", nwarmup=50, nruns=10000
):
    input_data = torch.randn(input_shape)
    input_data = input_data.to("cuda")
    if dtype == "fp16":
        input_data = input_data.half()

    print("Warm up ...")
    with torch.no_grad():
        for _ in range(nwarmup):
            features = model(input_data)
    torch.cuda.synchronize()
    print("Start timing ...")
    timings = []
    with torch.no_grad():
        for i in range(1, nruns + 1):
            start_time = time.time()
            features = model(input_data)
            torch.cuda.synchronize()
            end_time = time.time()
            timings.append(end_time - start_time)
            if i % 10 == 0:
                print(
                    "Iteration %d/%d, ave batch time %.2f ms"
                    % (i, nruns, np.mean(timings) * 1000)
                )

    print("Input shape:", input_data.size())
    print("Output features size:", features.size())
    print("Average batch time: %.2f ms" % (np.mean(timings) * 1000))


model = resnet50_model.eval().to("cuda")
print("Model benchmark without Torch-TensorRT")
benchmark(model, input_shape=(128, 3, 224, 224), nruns=100)


# The compiled module will have precision as specified by "op_precision".
# Here, it will have FP32 precision.
trt_model_fp32 = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input((128, 3, 224, 224), dtype=torch.float32)],
    enabled_precisions=torch.float32,  # Run with FP32
    workspace_size=1 << 22,
)

print("Compiled FP32 model benchmark")
benchmark(trt_model_fp32, input_shape=(128, 3, 224, 224), nruns=100)

# The compiled module will have precision as specified by "op_precision".
# Here, it will have FP16 precision.
trt_model_fp16 = torch_tensorrt.compile(
    model,
    inputs=[torch_tensorrt.Input((128, 3, 224, 224), dtype=torch.half)],
    enabled_precisions={torch.half},  # Run with FP32
    workspace_size=1 << 22,
)

print("Compiled FP16 model benchmark")
benchmark(trt_model_fp16, input_shape=(128, 3, 224, 224), dtype="fp16", nruns=100)
