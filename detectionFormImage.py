from ultralytics import YOLO
import yaml
import cv2

model = YOLO("yolov8s.pt")

model.predict("testImg.jpg", save=True, save_txt=True)

file_name = "../usr/local/lib/python3.8/dist-packages/ultralytics/yolo/data/datasets/coco8.yaml"
with open(file_name, "r") as stream:
    names = yaml.safe_load(stream)["names"]

names

lis = open("\runs\detect\predict2\labels\testImg.txt", "r").readlines()

lis

for l in lis:
    ind = int(l.split()[0])
    print(ind, names[ind])
