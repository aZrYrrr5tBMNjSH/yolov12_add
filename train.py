from ultralytics import YOLO


def main():
    # 1. 初始化模型（修改 yolov12.yaml 中 nc=3）
    model = YOLO('yolov12.yaml')

    # 2. 打印模型结构，确认 Detect 层的类别数是否是3
    print(model.model)

    # 3. 训练模型，确保传入正确的 data.yaml，类别数为3
    model.train(
        data='D:/CAMUS_YOLO1/data.yaml',
        epochs=50,
        batch=16,
        imgsz=640,
        name='camus_yolov12_multi_class34'
    )

    # 4. 训练结束后，可以用一个测试图片推理，降低置信度阈值，观察多类别结果
    results = model.predict(source='D:/CAMUS_YOLO1/images/test/test_image.png', conf=0.05, iou=0.1)
    results.print()  # 打印推理结果，看看是否检测到多类别


if __name__ == "__main__":
    main()
