import torch
from collections import Counter
from my_iou import intersection_union


def mean_avg_precision(
    pred_boxes, gth_boxes, iou_threshold, num_classes, box_format="midpoint", 
):
    average_precision = []

    epsilon = 1e-6

    for c in range(num_classes):
        ground_truth = []
        detections = []

        ground_truth = [
            box
            for box in ground_truth
            if lambda x : x[1] == c
        ]

        detections = [
            box
            for box in pred_boxes
            if lambda x : x[1] == c
        ]

        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))
        true_gth_num = len(ground_truth)

        if true_gth_num == 0:
            continue

        gth_num = Counter([gt[0] for gt in gth_boxes])
        gth_tensor = []
        for key, val in gth_num:
            gth_tensor.append(torch.zeros(val))

        detections.sort(key=lambda x: x[2], reverse=True)

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [gth for gth in ground_truth
                                if gth[0] == detection[0]]

            best_iou = 0
            best_index = 0

            for idx, gth in enumerate(ground_truth):
                iou = intersection_union(
                    torch.tensor(detection[3:]),
                    torch.tensor(gth[3:]),
                    box_format=box_format
                    )

                if iou > best_iou:
                    best_iou = iou
                    best_index = idx
                else:
                    best_iou = best_iou
                    best_index = best_index

                if gth_num[detection[0]][best_index] == 0:
                    if best_iou > iou_threshold:
                        TP[detection_idx] = 1
                        gth_num[detection[0]][best_index] = 1
                    else:
                        FP[detection_idx] = 1
                else:
                    FP[detection_idx] = 1
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP,dim=0)
        recalls = TP_cumsum / (true_gth_num + 1e-6)
        precisions = TP_cumsum / (TP_cumsum + FP_cumsum)
        recalls = torch.cat((torch.tensor([1]), recalls))
        precisions = torch.cat((torch.tensor([1]), precisions))
        average_precision.append(torch.trapz(precisions,recalls))

    return sum(average_precision) / len(average_precision)
