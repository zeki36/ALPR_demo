from ultralytics import YOLO
import os
import cv2


# img = cv2.imread("archive-3/images/test/7.jpg")
# h ,w ,c = img.shape
# print(h ,w,c)

OUTPUT_IMAGES_TRAIN = "archive-3/images/train"
OUTPUT_IMAGES_TEST = "archive-3/images/test"
print(os.getcwd()) #nerde çalışıy 
#---------------------------------------

#---------------------------------------
#model2 = YOLO('plate_detector.pt') #kendi modelimiz
model1 = YOLO('yolov8n.pt')


vehicles = [1,2 , 3,5, 6 ,7]
test_images = []
for s in os.listdir(OUTPUT_IMAGES_TEST):
    if s.endswith(('.jpg')):
        test_images.append(s)
for tst_img in test_images:
    img_dosya_yolu = os.path.join(OUTPUT_IMAGES_TEST,tst_img)
    image = cv2.imread(img_dosya_yolu)
    h, w, c = image.shape
    results1 = model1(image)

    for r in results1:
        boxes = r.boxes #tüm bounding boc cıktıları
        for box in boxes:
            #print(box.cls) tensor([2.])
            class_id = int(box.cls[0])
            #print(box.xyxy) tensor([[  59.3524,   97.9828, 1042.2819, 1032.7407]])
            if class_id in vehicles:
                x1 = int(box.xyxy[0][0])
                y1 = int(box.xyxy[0][1])
                x2 = int(box.xyxy[0][2])
                y2 = int(box.xyxy[0][3])
                #cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
                cv2.rectangle(image,(x1 ,y1),(x2,y2), (255,0,0),9)
                cv2.imshow("s",image)
                if cv2.waitKey(0) & 0xFF == ord('q'):
                    break                
