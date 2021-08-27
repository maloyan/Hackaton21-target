import json
import sys

import cv2
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from tqdm import tqdm

with open(sys.argv[1], "r") as f:
    config = json.load(f)

model = torch.load(
    f"{config['checkpoints']}/{config['image_size']}_{config['model']}.pt"
)
model.to("cuda")
model.eval()

meta_info = pd.read_csv(config["data_csv"])
test_data = meta_info[meta_info["split"] == "test"].path.values
test_target = meta_info[meta_info["split"] == "test"].target.values

fin_outputs = []
with torch.no_grad():
    for i in tqdm(test_data, total=len(test_data)):
        image = cv2.imread(i)
        image = torch.tensor(np.moveaxis(image, -1, 0), dtype=torch.float).unsqueeze(0)
        image = image.to(config["device"])

        outputs = model(image)
        outputs = outputs.squeeze(1)
        outputs = np.argmax(outputs.detach().cpu().numpy(), axis=1)
        fin_outputs.extend(outputs.tolist())

print(accuracy_score(test_target, fin_outputs))
