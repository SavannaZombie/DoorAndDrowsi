import cv2
import os
root_dir = os.getcwd()

face_set_dir = os.path.join(root_dir, 'face_set')

cam = cv2.VideoCapture(0)



cv2.namedWindow("test")

img_counter = 0

while True:
    ret, frame = cam.read()
    frame = cv2.resize(frame,(200, 150),fx=0,fy=0)
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("test", frame)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        # SPACE pressed
        name = input('Enter your name: ')
        img_name = os.path.join(face_set_dir, f"{name}.png")
        cv2.imwrite(img_name, frame)
        print(f"{img_name} written!")
        img_counter += 1

cam.release()
cv2.destroyAllWindows()
