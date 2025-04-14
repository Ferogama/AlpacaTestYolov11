import os
from ultralytics import YOLO
import cv2

# Пути к файлам
VIDEOS_DIR = r'C:\Users\fero\Desktop\дрон\alpacaTest\video'
video_path = r'C:\Users\fero\Desktop\дрон\alpacaTest\video\f3.webm'
video_path_out = os.path.join(VIDEOS_DIR, 'f3_detected.mp4')  # Результат сохранится рядом с исходным видео
model_path = r'D:\pycharm\projects\YoloV8\runs\detect\train\weights\best.pt'

# Открываем видео
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("Ошибка чтения видеофайла")
    exit()

# Получаем параметры видео
H, W, _ = frame.shape
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Создаем VideoWriter для сохранения результата
out = cv2.VideoWriter(video_path_out,
                      cv2.VideoWriter_fourcc(*'mp4v'),
                      fps,
                      (W, H))

# Загружаем модель
model = YOLO(model_path)  # загружаем кастомную модель
threshold = 0.5  # Порог уверенности для детекции

while ret:
    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            # Рисуем прямоугольник
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            # Добавляем текст с классом и уверенностью
            label = f"{results.names[int(class_id)].upper()} {score:.2f}"
            cv2.putText(frame, label, (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    # Сохраняем кадр с детекциями
    out.write(frame)

    # Показываем процесс обработки (опционально)
    cv2.imshow('Alpaca Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Читаем следующий кадр
    ret, frame = cap.read()

# Освобождаем ресурсы
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Обработка завершен, результат сохранен в : {video_path_out}")