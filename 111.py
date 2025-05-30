from ultralytics import YOLO

model = YOLO('runs/detect/camus_yolov12_multi_class3/weights/best.pt')
results = model('D:/CAMUS_YOLO1/images/train/patient0001_2CH_ED_0.png', conf=0.1)
print(results[0].boxes.cls, results[0].boxes.conf)
