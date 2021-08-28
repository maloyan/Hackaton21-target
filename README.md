## Хакатон Digital League AI Challenge

#### Настройка окружения
```
pip install -e .
pip install -r requirements
```
#### Запуск обучения
```
python aim_target/train.py configs/config.json 
```

#### Конвертация в onnx
```
python aim_target/torch2onnx.py configs/config.json 
```