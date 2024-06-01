import sys

def precision(confusion_matrix):
    """
    Computes the precision metric.
    :param confusion_matrix: the confusion matrix
    :return:
    """
    precision_v = None
    if confusion_matrix is not None and (len(confusion_matrix) == 2 and len(confusion_matrix[0]) == 2):
        try:
            precision_v = confusion_matrix[0][0] / (confusion_matrix[0][1] + confusion_matrix[0][0])
        except ZeroDivisionError:
            precision_v = confusion_matrix[0][0] / (confusion_matrix[0][1] + confusion_matrix[0][0] + sys.float_info.epsilon)
    return precision_v


def accuracy(confusion_matrix):
    """
    Computes the accuracy metric.
    :param confusion_matrix: the confusion matrix
    :return:
    """
    accuracy_v = None
    if confusion_matrix is not None and (len(confusion_matrix) == 2 and len(confusion_matrix[0]) == 2):
        try:
            accuracy_v = (
                    (confusion_matrix[0][0] + confusion_matrix[1][1]) /
                    (confusion_matrix[0][1] + confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[1][0])
            )
        except ZeroDivisionError:
            accuracy_v = (
                    (confusion_matrix[0][0] + confusion_matrix[1][1]) /
                    (confusion_matrix[0][1] + confusion_matrix[0][0] + confusion_matrix[1][1] + confusion_matrix[1][0] + sys.float_info.epsilon)
            )
    return accuracy_v


def f1_score(confusion_matrix):
    """
    Computes the f1 score metric.
    :param confusion_matrix:
    :return: f1 - the f1 score
    """
    precision_val = precision(confusion_matrix)
    recall_val = recall(confusion_matrix)
    try:
        f1 = 2 * ((precision_val * recall_val) / (precision_val + recall_val))
    except ZeroDivisionError:
        f1 = 2 * ((precision_val * recall_val) / (precision_val + recall_val + sys.float_info.epsilon))
    return f1


def recall(confusion_matrix):
    """
    Computes the recall metric.
    :param confusion_matrix: the confusion matrix
    :return: recall_v - the recall value
    """
    recall_v = None
    if confusion_matrix is not None and (len(confusion_matrix) == 2 and len(confusion_matrix[0]) == 2):
        try:
            recall_v = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0])
        except ZeroDivisionError:
            recall_v = confusion_matrix[0][0] / (confusion_matrix[0][0] + confusion_matrix[1][0] + sys.float_info.epsilon) 
    return recall_v


def compute_confusion_matrix(true_labels, predicted_labels, labels):
    """
    Computes the confusion matrix.
    :param true_labels: true labels
    :param predicted_labels: predicted labels
    :param labels:
    :return:
    """
    num_classes = len(labels)
    confusion_matrix = [[0] * num_classes for _ in range(num_classes)]

    for true, predicted in zip(true_labels, predicted_labels):
        true_index = labels.index(true)
        predicted_index = labels.index(predicted)
        confusion_matrix[true_index][predicted_index] += 1

    return confusion_matrix

def count_elements(lst):
    """
    Counts the number of elements in a list.
    :param lst: list of elements to count.
    :return: count dict
    """
    count_dict = {}
    for element in lst:
        if element in count_dict:
            count_dict[element] += 1
        else:
            count_dict[element] = 1
    return count_dict