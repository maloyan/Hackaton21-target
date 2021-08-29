import json
import sys
import torch.onnx

with open(sys.argv[1], "r") as f:
    config = json.load(f)

device = "cpu"


model = torch.load(
    f"{config['checkpoints']}/{config['image_size']}_{config['model']}.pt",
    map_location=torch.device(device)
)
model.eval()

image = torch.ones((1, 3, config["image_size"], config["image_size"]))

torch.onnx.export(
    model,
    (image),
    f"{config['checkpoints']}/{config['image_size']}_{config['model']}.onnx",
    input_names=["image"],
    output_names=["output"],
    dynamic_axes={"image": {0: "batch_size"}, "output": {0: "batch_size"}},
    opset_version=11
)
