import cv2
from ultralytics import YOLO

# Load YOLOv8 model
model = YOLO("yolo11s.pt") # lightweight model for real-time

# Start webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 384)

cv2.namedWindow("Phone Detection", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Phone Detection", 640, 384)

PHONE_CLASS_ID = 67
CONF_THRESH = 0.5

if not cap.isOpened():
    print("\nNo cam found\n")
    exit()

frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame_count += 1
    if frame_count % 2 != 0:
        continue
    
    resize = cv2.resize(frame, (640, 384))

    # Run object detection
    results = model(resize, conf=CONF_THRESH, imgsz=384, verbose=False)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])

            
            if cls_id == PHONE_CLASS_ID and conf >= CONF_THRESH:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                # Draw bounding box
                cv2.rectangle(resize, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(resize, "Phone Detected", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
            
    cv2.imshow("Phone Detection", resize)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()