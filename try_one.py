import cv2
import math
import numpy as np
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Inisialisasi pycaw untuk kontrol volume
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))
volRange = volume.GetVolumeRange()
minVol, maxVol, volStep = volRange[0], volRange[1], 0.05

# Inisialisasi volume berdasarkan sistem
currentVol = volume.GetMasterVolumeLevelScalar()
volPer = int(currentVol * 100)
volBar = np.interp(volPer, [0, 100], [400, 150])

# Inisialisasi webcam
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

# Variabel kontrol
pinch_threshold = 40
smoothing_factor = 5
delta_history = []

while cap.isOpened():
    success, img = cap.read()
    if not success:
        continue
    
    img = cv2.flip(img, 1)
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            # Dapatkan landmark jari
            index_tip = handLms.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            thumb_tip = handLms.landmark[mp_hands.HandLandmark.THUMB_TIP]
            
            # Konversi ke koordinat pixel
            h, w, _ = img.shape
            x1, y1 = int(index_tip.x * w), int(index_tip.y * h)
            x2, y2 = int(thumb_tip.x * w), int(thumb_tip.y * h)
            
            # Gambar garis dan lingkaran
            cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
            
            # Hitung jarak antar jari
            length = math.hypot(x2-x1, y2-y1)
            
            if length < pinch_threshold:
                cv2.putText(img, "PINCH MODE", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Hitung posisi vertikal rata-rata
                avg_y = (y1 + y2) // 2
                
                if 'prev_y' not in locals():
                    prev_y = avg_y
                
                # Hitung delta dengan smoothing
                delta_y = prev_y - avg_y
                delta_history.append(delta_y)
                
                if len(delta_history) > smoothing_factor:
                    delta_history.pop(0)
                
                smoothed_delta = sum(delta_history) / len(delta_history)
                
                # Konversi ke perubahan volume
                volChange = np.interp(smoothed_delta, [-50, 50], [-volStep, volStep])
                volPer += volChange * 100
                volPer = max(0, min(100, volPer))
                
                # Set volume aktual
                vol = np.interp(volPer, [0, 100], [minVol, maxVol])
                volume.SetMasterVolumeLevel(vol, None)
                
                # Update volume bar
                volBar = np.interp(volPer, [0, 100], [400, 150])
                
                prev_y = avg_y
            else:
                # Reset tracking saat tidak pinch
                if 'prev_y' in locals():
                    del prev_y
                delta_history.clear()

    # Tampilan volume
    cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
    cv2.rectangle(img, (50, int(volBar)), (85, 400), (0, 255, 0), cv2.FILLED)
    cv2.putText(img, f'{int(volPer)}%', (40, 450),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Volume Control", img)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()