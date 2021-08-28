import sys

import cv2
import gradio as gr
import numpy as np
import torch
import onnxruntime as ort
# Constants

DEVICE = "cpu"
IMG_SIZE = 128
PORT = 8989
TARGET_DICT = {
    0: "human",
    1: "target_human",
    2: "target_laser",
    3: "target_gun",
    4: "target_tank"
}
# Colors


CRIMSON = (220, 20, 60)
CADMIUMORANGE = (255, 97, 3)
GOLD = (255, 215, 0)
FORESTGREEN = (34, 139, 34)
CYAN = (0, 205, 205)
DODGERBLUE = (16, 78, 139)
DARKORCHID = (104, 34, 139)

class_to_color = [
    CRIMSON,
    CADMIUMORANGE,
    GOLD,
    FORESTGREEN,
    CYAN,
    DODGERBLUE,
    DARKORCHID,
]


# Load model


# model = torch.load(sys.argv[1])
# model.to(DEVICE)
# model.eval()

model = ort.InferenceSession(sys.argv[1])

def handler(file_obj):
    orig_image = cv2.imread(file_obj.name)
    image = cv2.resize(orig_image, (IMG_SIZE, IMG_SIZE))
    image = np.array([np.moveaxis(image, -1, 0)], dtype=np.float32)
    #image = torch.tensor(np.moveaxis(orig_image, -1, 0), dtype=torch.float).unsqueeze(0)
    #image = image.to(DEVICE)

    #output = model(image)
    output = model.run(None, {'image': image})
    output = np.argmax(output[0])
    #output = np.argmax(output.detach().cpu().numpy(), axis=1)

    return [orig_image, TARGET_DICT[output]]


# UI


iface = gr.Interface(
    handler,
    inputs=gr.inputs.File(label="Исходное изображение"),
    outputs=[
        gr.outputs.Image(label="Загруженное изображение"),
        gr.outputs.Textbox(label="Результат работы модели"),
    ],
    layout="vertical",
    title="Распознавание вида мишени",
    article="""### Инструкция по применению

1. Для выбора изображения нажмите на форму вверху страницы или перетащите файл на неё.

2. Далее, нажмите кнопку "Submit" чтобы начать обработку изображения, либо кнопку "Clear" чтобы выбрать другой файл.

3. После обработки загруженного файла будут отображено изображение: оригинал и результат работы модели.

""",
    theme="compact",
    server_port=PORT,
    server_name="0.0.0.0",
    live=False,
    allow_flagging=False,
    allow_screenshot=False
)
iface.launch(share=True)
