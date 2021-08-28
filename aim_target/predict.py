import json
import os
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from aim_target.utils import compute_accuracy

with open(sys.argv[1], "r") as f:
    config = json.load(f)

model = torch.load(
    f"{config['checkpoints']}/{config['image_size']}_{config['model']}.pt"
)
model.to("cuda")
model.eval()

meta_info = pd.read_csv(config["data_csv"])
meta_info = meta_info[meta_info["split"] == "test"]
test_data = [os.path.join(config["data_path"], i) for i in meta_info.path.values]
test_target = meta_info.target.values

fin_outputs = []
with torch.no_grad():
    for i in tqdm(test_data, total=len(test_data)):
        image = cv2.imread(i)
        image = cv2.resize(image, (config["image_size"], config["image_size"]))
        image = torch.tensor(np.moveaxis(image, -1, 0), dtype=torch.float).unsqueeze(0)
        image = image.to(config["device"])

        outputs = model(image)
        outputs = outputs.squeeze(1)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        fin_outputs.extend(outputs.tolist())

compute_accuracy(test_target, fin_outputs)
# meta_info.target = fin_outputs
# meta_info.path = meta_info.path.apply(lambda x: x.split("/")[1])
# meta_info[["target", "path"]].to_csv("submission.csv", index=None)
