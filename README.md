<h1 style="font-size:10vw"> DuckHuntGame-<img src="https://github.com/SOURAB-BAPPA/DuckHuntGame-AI/blob/main/ai.gif" width=40 height=40 />  <img src="https://user-images.githubusercontent.com/73097560/115834477-dbab4500-a447-11eb-908a-139a6edaec5c.gif">
<img src="https://github.com/SOURAB-BAPPA/DuckHuntGame-AI/blob/main/DuckHunt.png" >
   
[Video](https://photos.app.goo.gl/ensQAaN13FhL1gGY6)

# Train Custom Model:
1. Gather some [Duck Hunt Game](https://duckhuntjs.com/index.html) images (min 100 img)
2. [Label that images](https://www.makesense.ai/)
3. Export the label data and store in labels folder
4. Open yolo v5
<div>
   <a href="https://github.com/ultralytics/yolov5/actions"><img src="https://github.com/ultralytics/yolov5/workflows/CI%20CPU%20testing/badge.svg" alt="CI CPU testing"></a>
   <a href="https://zenodo.org/badge/latestdoi/264818686"><img src="https://zenodo.org/badge/264818686.svg" alt="YOLOv5 Citation"></a>
   <a href="https://hub.docker.com/r/ultralytics/yolov5"><img src="https://img.shields.io/docker/pulls/ultralytics/yolov5?logo=docker" alt="Docker Pulls"></a>
   <br>
   <a href="https://colab.research.google.com/github/ultralytics/yolov5/blob/master/tutorial.ipynb"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
   <a href="https://www.kaggle.com/ultralytics/yolov5"><img src="https://kaggle.com/static/images/open-in-kaggle.svg" alt="Open In Kaggle"></a>
   <a href="https://join.slack.com/t/ultralytics/shared_invite/zt-w29ei8bp-jczz7QYUmDtgo6r6KcMIAg"><img src="https://img.shields.io/badge/Slack-Join_Forum-blue.svg?logo=slack" alt="Join Forum"></a>
</div>

5. Run setup
6. Upload DuckHuntGame.zip
7. Unzip DuckHuntGame folder ```!unzip -q ../DuckHuntGame.zip -d ../```
8. Upload [custom_data.yaml](https://github.com/SOURAB-BAPPA/DuckHuntGame-AI/blob/main/custom_data.yaml) in yolov5->data section
9. Train YOLOv5s on custom_data for min 90 epochs
10. Download runs/train/exp2/weights/last.pt or runs/train/exp2/weights/best.pt

 
Then, your directory structure should look something like this

```
DuckHuntGame/
└─ images/
   ├─ train/
   └─ val/
└─ labels/
   ├─ train/
   └─ val/
```

<img alt="Coding Gif" src="https://github.com/SOURAB-BAPPA/DuckHuntGame-AI/blob/main/duck-hunt.gif" height="200" align="right"/>
<br/>

## <div align="center">Quick Start Examples</div> 

<details open>
<summary>Install</summary>

[**Python>=3.6.0**](https://www.python.org/) is required with all
[requirements.txt](https://github.com/ultralytics/yolov5/blob/master/requirements.txt) installed including
[**PyTorch>=1.7**](https://pytorch.org/get-started/locally/):
<!-- $ sudo apt update && apt install -y libgl1-mesa-glx libsm6 libxext6 libxrender-dev -->

```bash
$ git clone https://github.com/ultralytics/yolov5
$ cd yolov5
$ pip install -r requirements.txt
$ pip install PyAutoGUI
$ git clone https://github.com/SOURAB-BAPPA/DuckHuntGame-AI/blob/main/last.pt
```

</details>
<details open>
<summary>Run</summary>

```bash
import cv2
from grabscreen import grab_screen
import torch
import pyautogui
from PIL import ImageGrab
import numpy as np
import time

# Model
model = torch.hub.load('E:/Hand_Finger_detection/yolov5', 'custom', path='D:/last.pt', source='local')  # local repo  
while True:
    img = grab_screen(region=(0, 0, 1920, 700))
    img = cv2.resize(img, (640, 640))
    cv2.imshow("Screenshot", img)
    results = model(img, size=640)
    b1 = results.pandas().xyxy[0]
    try:
        x, y, xm, ym = int(b1['xmin'][0]), int(b1['ymin'][0]), int(b1['xmax'][0]), int(b1['ymax'][0])
        pyautogui.click((x + (xm - x) // 2) * 3, ((y + (ym - y) / 2) * 35) // 32)
    except:
        pass
    results.render()  # box selected object
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imshow("Result", img)
    key = cv2.waitKey(1)
    if key == 27:
        break
cv2.destroyAllWindows()
```

</details>

# <div align="center">Issues</div>

<details close>
<summary>CUDA error</summary>
  
1. [Select based on your system](https://pytorch.org/)
2. Update graphics driver
3. This command must run successfully if solved CUDA problem
  ```
  import torch
  torch.zeros(1).cuda()
  ```
 [Not Solved](https://stackoverflow.com/questions/60987997/why-torch-cuda-is-available-returns-false-even-after-installing-pytorch-with/61034368#61034368)
 
</details>
<details close>
<summary>Training problem</summary>
  
[Support video](https://www.youtube.com/watch?v=GRtgLlwxpc4)
 
</details>
<details close>
<summary>Load YOLOv5</summary>
  
[Load YOLOv5 from PyTorch Hub](https://github.com/ultralytics/yolov5/issues/36)
 
</details>

For program bugs and issues or any types of suggestions [mail](mailto:maitysourab@gmail.com) me.

<div align="center">
   <a href = "https://www.linkedin.com/in/sourab-maity-4551061b8/"><img src="https://img.icons8.com/cute-clipart/45/000000/linkedin.png"/></a>
   <img width="3%" />
   <a href = "https://twitter.com/maity_sourab"><img src="https://img.icons8.com/cotton/45/000000/twitter.png"/></a>
    <img width="3%" />
    <a href="https://github.com/SOURAB-BAPPA">
        <img src="https://img.icons8.com/nolan/64/github.png" width="5%"/>
    </a>
    <img width="3%" />
    <a href="https://stackoverflow.com/users/13909768/sourab-maity">
        <img src="https://img.icons8.com/color/48/000000/stackoverflow.png"/>
    </a>
</div>
