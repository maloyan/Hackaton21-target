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

#### Запуск фронта
```
python aim_target/webapp.py checkpoints/128_resnet50.onnx
```

#### Собрать докер
```
docker build . -t army
docker run -p 8989:8989 --ipc=host -d --rm army:latest
```