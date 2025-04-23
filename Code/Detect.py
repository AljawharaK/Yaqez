import cv2
import os
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from PIL import Image
from deepface import DeepFace
import numpy as np
import time

# تحميل النموذج من HuggingFace
model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
model = YOLO(model_path)

# تحميل الوجه المستهدف
target_image_path = "Target.png"
target_img_data = DeepFace.extract_faces(img_path=target_image_path, detector_backend="retinaface", enforce_detection=False)
if not target_img_data:
    print(" لم يتم العثور على وجه في الصورة المستهدفة.")
    exit()

target_img = target_img_data[0]["face"]

# مجلد لحفظ الفريمات
os.makedirs("detections", exist_ok=True)

# دالة معالجة الفيديو
def process_video(video_path, camera_id):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f" تعذر فتح الفيديو {video_path}")
        return

    frame_number = 0
    timestamps_found = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_number += 1
        if frame_number > 15:
            break

        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        output = model(pil_frame)
        results = Detections.from_ultralytics(output[0])

        if results.xyxy is not None and len(results.xyxy) > 0:
            for i in range(len(results.xyxy)):
                x1, y1, x2, y2 = results.xyxy[i]
                confidence = results.confidence[i]

                face = frame[int(y1):int(y2), int(x1):int(x2)]
                if face.size == 0:
                    continue
                face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

                try:
                    verification = DeepFace.verify(face_rgb, target_img, model_name='VGG-Face', enforce_detection=False)
                    if verification["verified"]:
                        timestamp = time.strftime('%H-%M-%S', time.gmtime(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000))
                        if timestamp not in timestamps_found:
                            timestamps_found.append(timestamp)

                            # رسم المربع على الوجه
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cv2.putText(frame, f"{confidence:.2f}", (int(x1), int(y1)-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

                            # حفظ الفريم
                            save_path = f"detections/camera{camera_id}_frame{frame_number}_{timestamp}.jpg"
                            cv2.imwrite(save_path, frame)
                            print(f"\n تم حفظ الفريم: {save_path}")
                            print(f" الشخص وُجد في الكاميرا {camera_id} في الوقت {timestamp}")
                            print(f" الإحداثيات: ({x1:.0f}, {y1:.0f}) -> ({x2:.0f}, {y2:.0f})")
                            print(f" الثقة: {confidence:.2f}")
                except Exception as e:
                    print(" خطأ أثناء التحقق:", e)

        cv2.imshow(f"Camera {camera_id}", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    return timestamps_found

# تشغيل على الفيديوهات
video_path_camera_1 = 'Camera1.mp4'
video_path_camera_2 = 'Camera2.mp4'

print(" البحث في الكاميرا 1...")
results_1 = process_video(video_path_camera_1, camera_id=1)
if results_1:
    for time_found in results_1:
        print(f" الشخص وُجد في الكاميرا 1 عند {time_found}")
else:
    print(" لم يتم العثور على الشخص في الكاميرا 1.")

print("📹 البحث في الكاميرا 2...")
results_2 = process_video(video_path_camera_2, camera_id=2)
if results_2:
    for time_found in results_2:
        print(f" الشخص وُجد في الكاميرا 2 عند {time_found}")
else:
    print(" لم يتم العثور على الشخص في الكاميرا 2.")
