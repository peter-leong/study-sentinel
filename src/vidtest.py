import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("\nNo cam found\n")
    exit()

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("\nNothing returned\n")
        exit()
    
    cv2.imshow("frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break
    
cap.release()
cv2.destroyAllWindows()