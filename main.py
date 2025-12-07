from ultralytics import YOLO
import os
import glob
import random
import shutil
from pathlib import Path
import cv2

# img = cv2.imread("archive-3/images/test/7.jpg")
# h ,w ,c = img.shape
# print(h ,w,c)

OUTPUT_IMAGES_TRAIN = "archive-3/images/train"
OUTPUT_IMAGES_TEST = "archive-3/images/test"
print(os.getcwd()) #nerde çalışıy
#---------------------------------------
def get_box_coordinates(center_x, center_y, width, height, img_w, img_h):
    
    x_min = int((center_x - (width / 2)) * img_w)
    y_min = int((center_y - (height / 2)) * img_h)
    x_max = int((center_x + (width / 2)) * img_w)
    y_max = int((center_y + (height / 2)) * img_h)

    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(img_w, x_max)
    y_max = min(img_h, y_max)

    return x_min, y_min, x_max, y_max
#---------------------------------------
model1 = YOLO('yolov8n.pt')
model2 = YOLO('plate_detector.pt')

train_images = []
for s in os.listdir(OUTPUT_IMAGES_TRAIN):
    if s.endswith(('.jpg')):
        train_images.append(s)
for trn_img in train_images:
    img_dosya_yolu = os.path.join(OUTPUT_IMAGES_TRAIN,train_images)
    image = cv2.imread(img_dosya_yolu)
    h, w, c = image.shape
    results1 = model1(image)

    for r in results1:
        boxes = r.boxes #tüm bounding boc cıktıları
        for box in boxes:
                    


























'''
 Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk




    from ultralytics import YOLO

# Load a pretrained YOLO11n model
model = YOLO("yolo11n.pt")

# Run inference on the source
results = model(source=0, stream=True)  # generator of Results objects



cpu()	Method	Move the object to CPU memory.
numpy()	Method	Convert the object to a numpy array.
cuda()	Method	Move the object to CUDA memory.
to()	Method	Move the object to the specified device.
xyxy	Property (torch.Tensor)	Return the boxes in xyxy format.
conf	Property (torch.Tensor)	Return the confidence values of the boxes.
cls	Property (torch.Tensor)	Return the class values of the boxes.
id	Property (torch.Tensor)	Return the track IDs of the boxes (if available).
xywh	Property (torch.Tensor)	Return the boxes in xywh format.
xyxyn	Property (torch.Tensor)	Return the boxes in xyxy format normalized by original image size.
xywhn	Property (torch.Tensor)	Return the boxes in xywh format normalized by original image size.
'''
