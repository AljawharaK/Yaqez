from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import cv2
import threading
import numpy as np
from inference import InferencePipeline

app = FastAPI()

frame_with_boxes = None  # لتخزين آخر فريم بعد التعديل عليه
def my_sink(result, video_frame):
    global frame_with_boxes
    frame = video_frame.image.copy()
    predictions = result.get("predictions", [])

    for pred in predictions:
        try:
            if isinstance(pred, (list, tuple)) and len(pred) >= 6:
                box = pred[0]  # numpy array [x0, y0, x1, y1]
                conf = float(pred[2])  # confidence
                class_id = pred[3]
                class_name = pred[5].get("class_name", "unknown")

                # التأكد من أن الإحداثيات أعداد صحيحة
                x0, y0, x1, y1 = map(int, box)

                # رسم الصندوق
                cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 0), 2)
                cv2.putText(
                    frame,
                    f"{class_name} {round(conf, 2)}",
                    (x0, y0 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            else:
                print(f"❌ Skipping unsupported prediction format: {pred}")
        except Exception as e:
            print(f"❌ Error processing prediction {pred}: {e}")
            continue

    frame_with_boxes = frame



# تهيئة الـ pipeline
pipeline = InferencePipeline.init_with_workflow(
    api_key="q4izGFwbfMIkW9Eyjjtc",
    workspace_name="ai-nwwvh",
    workflow_id="custom-workflow-5",
    video_reference=0,  # الكاميرا الافتراضية
    max_fps=10,
    on_prediction=my_sink
)


# تشغيل الـ pipeline في thread منفصل
pipeline_thread = threading.Thread(target=pipeline.start, daemon=True)
pipeline_thread.start()


# بث الفيديو بالفريمات المعدلة
def generate_frames():
    global frame_with_boxes
    while True:
        if frame_with_boxes is not None:
            success, buffer = cv2.imencode('.jpg', frame_with_boxes)
            if not success:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


@app.get("/")
def read_root():
    return {"message": "🎥 Live video with bounding boxes is running ✅"}


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")