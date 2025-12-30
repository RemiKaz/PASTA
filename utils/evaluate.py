import numpy as np
import torch
from tqdm import tqdm


# Define the evaluation fct
def evaluate_(model, dataloader):
    """Evaluates a model on a given dataloader.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): The dataloader to use for evaluation.

    Returns:
        float: The average accuracy of the model on the dataloader.
    """
    model.eval()
    n_sucess = 0

    if model.name == "CBM-backbone-resnet":
        perf_label_test = 0
        # Iterate over the dataloader
        for item in tqdm(dataloader):
            with torch.no_grad():
                preds = model(item["input"])
            target = item["output"].cpu().detach().numpy()
            pred = preds.cpu().detach().numpy()
            # compute the predictions
            pred = pred > 0
            # update the performance
            perf_label_test = perf_label_test + (target == pred).sum(axis=0)
        model.train()
        acc_parts = perf_label_test / len(dataloader)
        return sum(acc_parts) / len(acc_parts)

    # Iterate over the dataloader
    for item in dataloader:
        with torch.no_grad():
            preds = model(item["input"])
        target = item["output"].cpu().detach().numpy()
        id_infer = torch.argmax(preds, dim=1).cpu().detach().numpy()
        n_sucess += sum(id_infer == target)

    model.train()

    return n_sucess / len(dataloader)


def evaluate_fasterrcnn(model, data_loader, iou_thresholds="Default"):
    """Calculate the mean Average Precision (mAP) of a model on a dataset."""
    model.eval()
    all_precisions = []
    all_recalls = []

    if iou_thresholds == "Default":
        iou_thresholds = np.arange(0.5, 1.0, 0.05)

    """no = 0"""

    with torch.no_grad():
        for item in tqdm(data_loader):
            """# Pass the first item 
            if no < 15:
                no+=1
            
            else :
                exit()"""

            outputs = model(item["input"])

            if isinstance(item["output"], dict):
                true_boxes = item["output"]["bbox"][0].cpu().numpy()
                true_labels = item["output"]["labels"][0].cpu().numpy()
                pred_boxes = outputs[0]["boxes"].cpu().numpy()
                pred_labels = outputs[0]["labels"].cpu().numpy()
                scores = outputs[0]["scores"].cpu().numpy()

            else:
                true_boxes = item["output"][0]["bbox"].cpu().numpy()
                true_labels = item["output"][0]["labels"].cpu().numpy()
                pred_boxes = outputs[0]["boxes"].cpu().numpy()
                pred_labels = outputs[0]["labels"].cpu().numpy()
                scores = outputs[0]["scores"].cpu().numpy()

            # Keep only bboxes with score > 0.5

            score_treshold = 0.01
            pred_boxes = pred_boxes[scores > score_treshold]
            pred_labels = pred_labels[scores > score_treshold]
            scores = scores[scores > score_treshold]

            """print(true_boxes, true_labels, pred_boxes, pred_labels, scores)

            img_np = item['input'][0].cpu().numpy().transpose((1,2,0))
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            unnormalized_image = std * img_np + mean
            input_image_pil = Image.fromarray(np.uint8(unnormalized_image*255))
            
            utils.plot_image_with_bboxes(input_image_pil, pred_boxes,name_file='test_bboxes.png')
            utils.plot_image_with_bboxes(input_image_pil, true_boxes,name_file='test_bboxes_gt.png')
            exit()"""

            for iou_threshold in iou_thresholds:
                """print(true_boxes,true_labels)
                print(pred_boxes,pred_labels)"""

                precision, recall = calculate_precision_recall(
                    true_boxes, true_labels, pred_boxes, pred_labels, iou_threshold=iou_threshold
                )
                all_precisions.append(precision)
                all_recalls.append(recall)
                """print(np.mean(all_precisions) if all_precisions else 0)
                exit()"""

    mean_ap = np.mean(all_precisions) if all_precisions else 0
    model.train()
    return mean_ap


def calculate_precision_recall(true_boxes, true_labels, pred_boxes, pred_labels, iou_threshold=0.5):
    """Calculate precision and recall for a given IoU threshold, considering labels."""
    tp = 0
    fp = 0
    fn = 0

    matched_pred_boxes = set()
    matched_true_boxes = set()

    for pred_box, pred_label in zip(pred_boxes, pred_labels, strict=False):
        best_iou = 0
        best_true_box_idx = -1

        for idx, (true_box, true_label) in enumerate(zip(true_boxes, true_labels, strict=False)):
            if idx not in matched_true_boxes and pred_label == true_label:
                iou = calculate_iou(pred_box, true_box)
                if iou > best_iou:
                    best_iou = iou
                    best_true_box_idx = idx

        if best_iou >= iou_threshold:
            tp += 1
            matched_pred_boxes.add(tuple(pred_box))
            matched_true_boxes.add(best_true_box_idx)
        else:
            fp += 1

    fn = len(true_boxes) - len(matched_true_boxes)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0

    return precision, recall


def calculate_iou(box1, box2):
    """Calculate the Intersection over Union (IoU) of two bounding boxes."""
    x1, y1, x2, y2 = box1
    x1_gt, y1_gt, x2_gt, y2_gt = box2

    # Determine the coordinates of the intersection rectangle
    xi1 = max(x1, x1_gt)
    yi1 = max(y1, y1_gt)
    xi2 = min(x2, x2_gt)
    yi2 = min(y2, y2_gt)
    inter_width = max(0, xi2 - xi1)
    inter_height = max(0, yi2 - yi1)
    inter_area = inter_width * inter_height

    # Calculate the area of both bounding boxes
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x2_gt - x1_gt) * (y2_gt - y1_gt)

    # Calculate the intersection over union
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area
