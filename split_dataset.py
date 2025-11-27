import os
import glob
import random
import shutil
from pathlib import Path

print(os.getcwd()) #nerde çalışıyom

RAW_IMAGES_DIR = "archive-3/images"
RAW_LABELS_DIR = "archive-3/label"

OUTPUT_IMAGES_TRAIN = "archive-3/images/train"
OUTPUT_IMAGES_TEST = "archive-3/images/test"
OUTPUT_LABELS_TRAIN = "archive-3/label/train"
OUTPUT_LABELS_TEST = "archive-3/label/test"

image_paths = []
image_paths.extend(glob.glob(os.path.join(RAW_IMAGES_DIR, "*.jpg")))
#print(image_paths)

random.shuffle(image_paths)
total = len(image_paths)
train = int(total * (0.8))
train_images= image_paths[:train]
test_images = image_paths[train:]
#print(len(train_images))
#print(len(test_images))

for tr_images in train_images:
    base = os.path.basename(tr_images)           
    go = os.path.join(OUTPUT_IMAGES_TRAIN, base)
    labels = Path(base).with_suffix(".txt").name
    label_base = os.path.join(RAW_LABELS_DIR, labels)
    label_go = os.path.join(OUTPUT_LABELS_TRAIN , labels)

    shutil.move(label_base , label_go)
    shutil.move(tr_images, go)

for tst_images in test_images:
    base = os.path.basename(tst_images)        
    go = os.path.join(OUTPUT_IMAGES_TEST, base)
    labels = Path(base).with_suffix(".txt").name
    label_base = os.path.join(RAW_LABELS_DIR, labels)
    label_go = os.path.join(OUTPUT_LABELS_TEST , labels)

    shutil.move(label_base , label_go)
    shutil.move(tst_images, go)