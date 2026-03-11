import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("../models/yolov8s.pt") # lightweight model for real-time

# Start webcam
cap = cv2.VideoCapture(0)

PHONE_CLASS_ID = 67
CONF_THRESH = 0.7

if not cap.isOpened():
    print("\nNo cam found\n")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    results = model(frame, conf=CONF_THRESH)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            
            if cls_id == PHONE_CLASS_ID and conf >= CONF_THRESH:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(frame, "Phone Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
    cv2.imshow("Phone Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()