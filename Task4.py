import cv2
import mediapipe as mp
import pyautogui
import time
 
def count_fingers(landmark):
    cnt = 0
    
    thresh = (landmark[0].y * 100 - landmark[5].y * 100) / 2
    
    if (landmark[5].y * 100 - landmark[8].y * 100) > thresh:
        cnt += 1
    if (landmark[9].y * 100 - landmark[12].y * 100) > thresh:
        cnt += 1
    if (landmark[13].y * 100 - landmark[16].y * 100) > thresh:
        cnt += 1
    if (landmark[17].y * 100 - landmark[20].y * 100) > thresh:
        cnt += 1
    if (landmark[5].x * 100 - landmark[4].x * 100) > 5:
        cnt += 1
    
    return cnt

cap = cv2.VideoCapture(0)

drawing = mp.solutions.drawing_utils
hands = mp.solutions.hands
hands_obj = hands.Hands(max_num_hands=1)

start_init = False
prev = -1

while True:
    end_time = time.time()
    _, frm = cap.read()
    frm = cv2.flip(frm, 1)
    
    res = hands_obj.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))
    
    if res.multi_hand_landmarks:
        hand_keypoints = res.multi_hand_landmarks[0].landmark
        cnt = count_fingers(hand_keypoints)
        
        if not(prev == cnt):
            if not start_init:
                start_time = time.time()
                start_init = True
            elif (end_time - start_time) > 0.2:
                if cnt == 1:
                    pyautogui.press("Right")
                elif cnt == 2:
                    pyautogui.press("Left")
                elif cnt == 3:
                    pyautogui.press("Up")
                elif cnt == 4:
                    pyautogui.press("Down")
                elif cnt == 5:
                    pyautogui.press("Space")
                
                prev = cnt
                start_init = False
        
        drawing.draw_landmarks(frm, res.multi_hand_landmarks[0], hands.HAND_CONNECTIONS)
    
    cv2.imshow("window", frm)
    
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break
