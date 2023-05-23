import torch
from my_iou import intersection_union

def nms(bboxes, threshold, iou_threshold, box_format = "corners"):
    assert(type(bboxes) == list)
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    box_nms = []

    while(bboxes):
        chosen_box = bboxes.pop(0)

        bboxes = [box for box in bboxes
                  if box[0] != chosen_box[0]    # decide by class
                  or intersection_union(   # decided by iou
                      torch.tensor(chosen_box[2:]),
                      torch.tensor(box[2:]),
                      box_format=box_format) < iou_threshold]

        box_nms.append(chosen_box)

    return box_nms
