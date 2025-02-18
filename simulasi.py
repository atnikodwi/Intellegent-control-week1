import cv2
import numpy as np
from ultralytics import YOLO

def load_model():
    """Memuat model YOLOv8"""
    model = YOLO("yolov8s.pt")  # Menggunakan YOLOv8 (versi small)
    return model

def detect_objects(model, frame):
    """Mendeteksi objek pada frame video"""
    results = model(frame)
    return results

def draw_boxes(frame, results):
    """Menggambar bounding box di frame"""
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])  # Koordinat bbox
            conf = box.conf[0].item()  # Confidence score
            class_id = int(box.cls[0])  # ID kelas
            label = f"{result.names[class_id]}: {conf:.2f}"

            # Gambar bounding box dan label
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    return frame

def main():
    model = load_model()
    cap = cv2.VideoCapture(0)  # Menggunakan webcam

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = detect_objects(model, frame)
        frame = draw_boxes(frame, results)

        # Tampilkan hasil deteksi
        cv2.imshow("YOLOv8 Object Detection", frame)

        # Tekan 'q' untuk keluar
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
