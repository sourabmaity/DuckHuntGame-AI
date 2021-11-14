import cv2
from grabscreen import grab_screen
import torch
import pyautogui
from PIL import ImageGrab
import numpy as np
import time

prev_frame_time = 0
new_frame_time = 0

# Model
model = torch.hub.load('E:/Hand_Finger_detection/yolov5', 'custom', path='D:/last.pt', source='local')  # local repo
while True:
    # img = np.array(ImageGrab.grab(bbox=(0, 0, 1920, 700)))
    img = grab_screen(region=(0, 0, 1920, 700))
    # new_frame_time = time.time()
    img = cv2.resize(img, (640, 640))
    # cv2.imshow("re", img)
    results = model(img, size=640)
    b1 = results.pandas().xyxy[0]
    try:

        x, y, xm, ym = int(b1['xmin'][0]), int(b1['ymin'][0]), int(b1['xmax'][0]), int(b1['ymax'][0])
        pyautogui.click((x + (xm - x) // 2) * 3, ((y + (ym - y) / 2) * 35) // 32)
        # cv2.rectangle(img, (x, y), (xm, ym), [0, 255, 0], 2)
    except:
        pass
    results.render()  # box selected object
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # cv2.circle(img, (x + (xm - x) // 2, y + (ym - y) // 2), 2, [0, 0, 255], 1)
    # fps = 1 / (new_frame_time - prev_frame_time)
    # prev_frame_time = new_frame_time

    # converting the fps into integer
    # fps = int(fps)

    # converting the fps to string so that we can display it on frame
    # by using putText function
    # fps = str(fps)

    # putting the FPS count on the frame
    # cv2.putText(img, fps, (7, 70), cv2.FONT_HERSHEY_SIMPLEX,2, (100, 255, 0), 3, cv2.LINE_AA)
    cv2.imshow("result", img)
    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()

# for testing on a single image
'''import torch
from PIL import ImageGrab
import cv2
# Model
model = torch.hub.load('E:/Hand_Finger_detection/yolov5', 'custom', path='D:/last.pt', source='local')  # local repo

# Image
img = cv2.imread("E:/DuckHuntGame/images/train/3.png")  # take a screenshot
img=cv2.resize(img,(640,640))
img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
cv2.imshow("re",img)
# Inference
results = model(img, size=640)
results.render()
#cv2.rectangle(img,(i[0],i[1]),(i[2],i[3]),[0,0,255],1)
b1=results.pandas().xyxy[0]
x,y,xm,ym=int(b1['xmin'][0]),int(b1['ymin'][0]),int(b1['xmax'][0]),int(b1['ymax'][0])
cv2.circle(img,(x+(xm-x)//2,y+(ym-y)//2),2,[0,0,255],1)
cv2.rectangle(img,(x,y),(xm,ym),[0,255,0],2)
print(results.pandas().xyxy[0]['xmin'][0])
img=cv2.cvtColor(img,cv2.COLOR_RGB2BGR)
cv2.imshow("re2",img)

cv2.waitKey()
cv2.destroyAllWindows()
print("hi")'''

# to check CUDA is properly working or not
'''import torch
torch.zeros(1).cuda()'''
