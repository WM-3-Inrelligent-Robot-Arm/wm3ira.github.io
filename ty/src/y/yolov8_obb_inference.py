from ultralytics import YOLO

# Load a model
model = YOLO("/home/eireland/ty/src/y/best.pt")  # load a custom model

# Predict with the model
results = model("/home/eireland/ty/src/y/data_for_test/processed_image.jpg", save=True)  # predict on an image