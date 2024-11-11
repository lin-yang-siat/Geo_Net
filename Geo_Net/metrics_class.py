from scipy.spatial import distance_matrix
import numpy as np
from medpy import metric


def calculate_metrics(predict_image, gt_image, evaluate):
    predict_image = np.array(predict_image.cpu(), dtype=bool)
    gt_image = np.array(gt_image.cpu(), dtype=bool)

    # True Positive（TP）
    tp = np.sum(np.logical_and(predict_image, gt_image))

    # True Negative（TN）
    tn = np.sum(np.logical_and(np.logical_not(predict_image), np.logical_not(gt_image)))

    # False Positive（FP）
    fp = np.sum(np.logical_and(predict_image, np.logical_not(gt_image)))

    # False Negative（FN）
    fn = np.sum(np.logical_and(np.logical_not(predict_image), gt_image))

    # IOU（Intersection over Union）
    iou = tp / (tp + fn + fp + 1e-7)

    # Dice Coefficient
    dice_coefficient = 2 * tp / (2 * tp + fn + fp + 1e-7)

    # Accuracy
    accuracy = (tp + tn) / (tp + fp + tn + fn + 1e-7)

    # precision
    precision = tp / (tp + fp + 1e-7)

    # recall
    recall = tp / (tp + fn + 1e-7)

    # Sensitivity
    sensitivity = tp / (tp + fn + 1e-7)

    # F1-score
    f1 = 2 * (precision * recall) / (precision + recall + 1e-7)

    # Specificity
    specificity = tn / (tn + fp + 1e-7)

    predicted_volume = np.sum(predict_image)
    true_volume = np.sum(gt_image)

    # Relative Absolute Volume Difference
    ravd = abs(predicted_volume - true_volume) / (0.5 * (predicted_volume + true_volume + 1e-7))

    if evaluate == "iou":
        return iou*100

    if evaluate == "dice":
        return dice_coefficient*100

    if evaluate == "accuracy":
        return accuracy*100

    if evaluate == "precision":
        return precision*100

    if evaluate == "recall":
        return recall*100

    if evaluate == "sensitivity":
        return sensitivity*100

    if evaluate == "f1":
        return f1*100

    if evaluate == "specificity":
        return specificity*100

    elif evaluate == "ravd":
        return ravd * 100

def calculate_distance(predict_image, gt_image, evaluate):
    predict_image = np.array(predict_image, dtype=bool)
    gt_image = np.array(gt_image, dtype=bool)

    if evaluate == "hd":
        return metric.hd95(predict_image, gt_image)

    if evaluate == "assd":
        return metric.assd(predict_image, gt_image)

