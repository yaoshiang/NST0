import torch
import torchvision

import onnx
from model_nst0 import NST0

# dummy_input = torch.randn(1, 3, 224, 224)
dummy_input = {
    'image': torch.randint(255, (1, 3, 224, 224), dtype=torch.float32) / 255.,
    'style': torch.randint(255, (1, 3, 224, 224), dtype=torch.float32) / 255.
}

# model = torchvision.models.alexnet(pretrained=True)
model = NST0()

input_names = ["image", "style"]
output_names = ["output"]

torch.onnx.export(
    model,
    dummy_input,
    # "alexnet.onnx",
    "model_nst0.onnx",
    verbose=False,
    input_names=input_names,
    output_names=output_names,
    dynamic_axes={
        'image': [1, 2, 3],
        'style': [1, 2, 3],
        'output': [1, 2, 3],
    },
)

model = onnx.load("model_nst0.onnx")
for input in model.graph.input:
  print(input.name, input.type)
