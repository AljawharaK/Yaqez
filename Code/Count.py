import cv2
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort

# تحميل YOLOv5m (نسخة متوازنة بين السرعة والدقة)
model = torch.hub.load('ultralytics/yolov5', 'yolov5m', pretrained=True)
model.classes = [0]  # فقط الأشخاص

# تهيئة Deep SORT
tracker = DeepSort(max_age=30)

# فتح الكاميرا (أو ضع مسار فيديو بدلاً من 0)
cap = cv2.VideoCapture(0)

# رفع دقة الكاميرا (اختياري)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # كشف الأشخاص باستخدام YOLOv5m
    results = model(frame, size=1024)  # تصغير الحجم لـ 1024 لتحسين السرعة
    detections = results.xyxy[0].cpu().numpy()

    # تحضير البيانات لـ Deep SORT
    deep_sort_inputs = []
    for *box, conf, cls in detections:
        x1, y1, x2, y2 = map(int, box)
        bbox = [x1, y1, x2 - x1, y2 - y1]  # x, y, width, height
        deep_sort_inputs.append((bbox, conf, 'person'))

    # تتبع الأشخاص
    tracks = tracker.update_tracks(deep_sort_inputs, frame=frame)

    ids = set()
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = track.track_id
        l, t, w, h = track.to_ltrb()
        ids.add(track_id)

        # رسم البوكس وكتابة ID
        cv2.rectangle(frame, (int(l), int(t)), (int(l + w), int(t + h)), (0, 255, 0), 2)
        cv2.putText(frame, f'ID: {track_id}', (int(l), int(t) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # عرض عدد الأشخاص على الشاشة
    cv2.putText(frame, f'People count: {len(ids)}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

    cv2.imshow("عداد الأشخاص - YOLOv5m + Deep SORT", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
