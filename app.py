import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from lib.utils import non_max_suppression, intersection_over_union
from lib.YOLOV3 import YOLOv3 as YOLO
import torchvision.transforms as transforms
import time
import cv2
import mss
from PIL import Image
import pyautogui
from lib.utils import non_max_suppression, intersection_over_union, load_checkpoint, cells_to_bboxes
from lib.utils import intersection_over_union as iou
from lib.YOLOV3 import YOLOv3 as YOLO
from albumentations.pytorch import ToTensorV2
import albumentations as A
from lib import config as C
import cv2
from torch.utils.data import DataLoader
from PIL import Image, ImageFile, ImageDraw, ImageFont
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from lib.utils import (
    cells_to_bboxes,
    iou_width_height as iou,
    non_max_suppression as nms,
    plot_image
)
from lib.app_utils import get_eval_boxes

def annotate_sign(image, keys, bboxes):
    height, width, _ = image.shape
    ids = keys
    for count, box in enumerate(bboxes):
        color = (100 + count * int(155 / 4), 0, 100 + count * int(155 / 4))
        print(box)
        name = ids[ids["ID"] == int(box[0])]["Name"].tolist()[0]
        # x, y, w, h = box[3], box[4], box[5], box[6]
        # left, right, top, bottom = [(x - w / 2) * width, (x + w / 2) * width,
        #                             (y - h / 2) * height, (y + h / 2) * height]
        _, _, left, right, top, bottom = box
        cv2.rectangle(image, (int(left), int(top)), (int(right), int(bottom)), color, 2)
        cv2.putText(img, name + " " +str(box[2]), (int(left) + 5, int(top) - 10), cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)

def convert(boxes, width, height):
    ratio_height = height * C.IMAGE_SIZE / width
    convert_height = (416 - ratio_height) / 2
    bboxes = []
    for count, box in enumerate(boxes):
        x = box[3] * C.IMAGE_SIZE
        y = box[4] * C.IMAGE_SIZE
        y -= convert_height
        x, y = x / 416 * width, y / ratio_height * height
        w = box[5] * width
        h = box[6] * width
        left, right, top, bottom = [(x - w / 2), (x + w / 2),
                                    (y - h / 2), (y + h / 2)]
        bboxes.append([box[1],box[2],left, right, top, bottom])
    return bboxes

def check_sign_detection(image, model, transform, keys):
    try:
        #image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_pil = np.array(Image.fromarray(image).convert("RGB"))
        #image_pil = np.array(image_pil.resize((C.IMAGE_SIZE, C.IMAGE_SIZE)))
        augmentations = transform(image = image_pil)
        image_aug = augmentations["image"]
        model.eval()
        pred_boxes = get_eval_boxes(image_aug, model, C.ANCHORS, iou_threshold = C.NMS_IOU_THRESH, threshold=0.7)
        height, width, _ = image.shape
        bboxes = convert(pred_boxes, width, height)
        image.flags.writeable = True
        annotate_sign(image, keys, bboxes)

    except Exception as e:
        print(e)
        image.flags.writeable = True
        left_location = int(50 * image.shape[0] / 480)
        font_scale = max(image.shape) / 800
        right_loc_1 = int(100 * image.shape[1] / 640)
        cv2.putText(image, "Looking for Target", (left_location, right_loc_1), cv2.FONT_HERSHEY_COMPLEX, font_scale,(255, 50, 0), 2, lineType=cv2.LINE_AA)

    video_display(image)

def video_display(image):
    # height, width = image.shape[:2]
    #tmp = cv2.resize(image, (800, 500), interpolation = cv2.INTER_CUBIC)
    cv2.imshow('YOLO', image)
    vid_writer.write(image)



if __name__ == "__main__":
    output_source = "lib/datasets/test_vids/tmp.mp4"
    start_time = time.time()
    this_width = 800
    this_height = 500
    monitor = {'top': 200, 'left': 200, 'width': this_width, 'height': this_height}

    keys = pd.read_csv("lib/datasets/ids.csv")

    # test_transforms = A.Compose(
    #     [
    #         A.LongestMaxSize(max_size=C.IMAGE_SIZE),
    #         A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
    #         ToTensorV2(),
    #     ],
    # )

    test_transforms = A.Compose(
        [
            A.LongestMaxSize(max_size=C.IMAGE_SIZE),
            A.PadIfNeeded(
                min_height=int(C.IMAGE_SIZE),
                min_width=int(C.IMAGE_SIZE),
                border_mode=cv2.BORDER_CONSTANT,
            ),
            A.Normalize(mean=[0, 0, 0], std=[1, 1, 1], max_pixel_value=255, ),
            ToTensorV2(),
        ],
    )

    S=[C.IMAGE_SIZE // 32, C.IMAGE_SIZE // 16, C.IMAGE_SIZE // 8]
    anchors = C.ANCHORS
    model = YOLO(num_classes=C.NUM_CLASSES).to(C.DEVICE)
    check = "lib/models/checkpoint_test1.pth.tar"
    optimizer = optim.Adam(
        model.parameters(), lr=C.LEARNING_RATE, weight_decay=C.WEIGHT_DECAY
    )
    load_checkpoint(
        check, model, optimizer, C.LEARNING_RATE
    )

    vid_writer = cv2.VideoWriter(output_source, cv2.VideoWriter_fourcc('m', 'p', '4', 'v'), 10,
                                 (this_width, this_height))
    with mss.mss() as sct:
        while 'Screen capturing':
            img = np.array(sct.grab(monitor))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
            if True:
                check_sign_detection(img, model, test_transforms, keys)
            else:
                print("Ignoring empty camera frame.")
                vid_writer.release()
                # If loading a video, use 'break' instead of 'continue'.
                continue

            if cv2.waitKey(5) & 0xFF == 27:
                vid_writer.release()
                break

    cv2.destroyAllWindows()