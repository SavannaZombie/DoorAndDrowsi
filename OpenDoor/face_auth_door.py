print('Import lib...')
import face_recognition
import cv2
import os
import beepy
import datetime as dt
import numpy as np
import time
from pathlib import Path
from gtts import gTTS 

print('Load config...')
root_dir = Path(os.getcwd())
face_set_dir = root_dir / 'face_set'
face_bin_dir = root_dir / 'face_bin'

sound_dir = root_dir / 'sound'
face_name_sound_dir = sound_dir / 'face_name'
doorSound = str(sound_dir / 'open.mp3')
closeSound = str(sound_dir / 'close.mp3')

lastTime = dt.datetime.now()
currentTime = dt.datetime.now()
video_capture = cv2.VideoCapture(0)

print('Load Faces...')
known_face_encodings = []
known_face_names = []

list_face_name = {i.stem: i for i in list(face_name_sound_dir.iterdir())}

for img_path in list(face_set_dir.iterdir()):
    person_image = face_recognition.load_image_file(img_path)
    person_face_encoding = face_recognition.face_encodings(person_image)[0]
    known_face_encodings.append(person_face_encoding)
    known_face_names.append(img_path.stem)

    if img_path.stem not in list_face_name.keys():
        var = gTTS('Welcome ' + img_path.stem, lang='en') 
        var.save(str(face_name_sound_dir/ (img_path.stem + '.mp3')))


face_locations = []
face_encodings = []
face_names = []
list_face_sound = {i.stem: str(i) for i in list(face_name_sound_dir.iterdir())}
process_this_frame = True

def play_sound(name):
    beepy.beep(sound=5)
    os.system('mpg123 ' + doorSound)
    os.system('mpg123 ' + list_face_sound[name])
    time.sleep(3)
    os.system('mpg123 ' + closeSound)

print('Starting...')
i = 0
while True:
    _, frame = video_capture.read()

    frame = cv2.resize(frame,(200, 150),fx=0,fy=0)

    # small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    # small_frame = cv2.resize(frame, (200, 150),fx=0,fy=0, interpolation = cv2.INTER_CUBIC)

    rgb_small_frame = frame[:, :, ::-1]

    if i  % 10 == 0:

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        face_names = []
        for face_encoding in face_encodings:

            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
            name = "Unknown"


            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
                if (currentTime - lastTime).seconds > 8:
                    lastTime = dt.datetime.now()
                    play_sound(name)
                    # break

            face_names.append(name)
        currentTime = dt.datetime.now()
        print(currentTime.strftime('%Y-%m-%d %H:%M:%S'), ' : ', lastTime.strftime('%Y-%m-%d %H:%M:%S'), '\r', flush=True, end='')
    i += 1

    for (top, right, bottom, left), name in zip(face_locations, face_names):

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # เขียนตัวหนังสือที่แสดงชื่อลงที่กรอบ
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

    # แสดงรูปภาพผลลัพธ์
    cv2.imshow('Video', frame)

    # กด 'q' เพื่อปิด!
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
        
video_capture.release()
cv2.destroyAllWindows()
