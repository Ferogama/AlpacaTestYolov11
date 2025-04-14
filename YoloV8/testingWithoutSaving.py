import cv2
from ultralytics import YOLO

# Путь к файоу для детекции
video_path = r'C:\Users\fero\Desktop\дрон\test\video\f3.webm'
# Путь к модели
model_path = r'D:\pycharm\projects\YoloV8\runs\detect\train\weights\best.pt'

# Загрузка модели
model = YOLO(model_path)
threshold = 0.5  # Порог уверенности

# Открываем видео
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print("Ошибка открытия видеофайла")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break  # Выход, если видео закончилось

    # Детекция
    results = model(frame)[0]

    # Отрисовка bounding boxes
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
            # Рисуем прямоугольник
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Подпись с классом и уверенностью
            label = f"{results.names[int(class_id)].upper()} {score:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

    # Показываем результат
    cv2.imshow('Alpaca Detection', frame)

    # Выход по нажатию 'q' или ESC
    if cv2.waitKey(1) & 0xFF in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()