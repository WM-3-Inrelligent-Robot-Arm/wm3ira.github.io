from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n-obb.pt")  # load a pretrained model

# Train the model
results = model.train(data="train_info.yaml", epochs=100, imgsz=640, device="cpu")