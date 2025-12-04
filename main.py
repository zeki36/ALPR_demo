from ultralytics import YOLO

model = YOLO("yolov8n.pt")

results = model.train(
    data="data.yaml",
    epochs = 50,
    device='mps', #mac icin
    name="plate_model",
    workers=4
)

print("Train yapıldı, Test yapılıyor")
test = model.val()

model.export(format="coreml")

