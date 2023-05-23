import torch

#  输入是只含有坐标的tensor
def intersection_union(boxes_pred, boxes_label, box_format):

    if box_format == "midpoint":
        box1_x1 = boxes_pred[..., 0:1] - boxes_pred[..., 2:3] / 2
        box1_y1 = boxes_pred[..., 1:2] - boxes_pred[..., 3:4] / 2
        box1_x2 = boxes_pred[..., 0:1] + boxes_pred[..., 2:3] / 2
        box1_y2 = boxes_pred[..., 1:2] + boxes_pred[..., 3:4] / 2
        box2_x1 = boxes_label[..., 0:1] - boxes_label[..., 2:3] / 2
        box2_y1 = boxes_label[..., 1:2] - boxes_label[..., 3:4] / 2
        box2_x2 = boxes_label[..., 0:1] + boxes_label[..., 2:3] / 2
        box2_y2 = boxes_label[..., 1:2] + boxes_label[..., 3:4] / 2

    if box_format == 'corners':
        box1_x1 = boxes_label[..., 0:1]
        box1_y1 = boxes_label[..., 1:2]
        box1_x2 = boxes_label[..., 2:3]
        box1_y2 = boxes_label[..., 3:4]
        box2_x1 = boxes_pred[..., 0:1]
        box2_y1 = boxes_pred[..., 1:2]
        box2_x2 = boxes_pred[..., 2:3]
        box2_y2 = boxes_pred[..., 3:4]


        x_1 = torch.max(box1_x1, box2_x1)
        x_2 = torch.min(box1_x2, box2_x2)
        y_1 = torch.max(box1_y1, box2_y1)
        y_2 = torch.min(box1_y2, box2_y2)

        width = x_2 - x_1
        height = y_2 - y_1

        intersection =  width.clamp(0) * height.clamp(0)
        area_1 = abs((box1_x2-box1_x1) * (box1_y2 - box1_y1))
        area_2 = abs((box2_x2-box2_x1) * (box2_y2 - box2_y1))
        iou = intersection /(area_1 + area_2 - intersection + 1e-6)


        return iou
