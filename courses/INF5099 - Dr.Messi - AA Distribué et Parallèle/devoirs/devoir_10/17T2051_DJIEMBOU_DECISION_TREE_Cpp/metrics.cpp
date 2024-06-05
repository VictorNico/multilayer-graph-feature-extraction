//
// Created by DJIEMBOU TIENCTHEU VICTOR NICO on 25/05/2024.
//

#include "headers/metrics.h"

using namespace std;

double precision(const std::vector<std::vector<int>>& confusion_matrix) {
    double precision_v = 0.0;
    if (!confusion_matrix.empty() && confusion_matrix.size() == 2 && confusion_matrix[0].size() == 2) {
        if (confusion_matrix[0][0] >= 0 && confusion_matrix[0][1] >= 0 && confusion_matrix[1][0] >= 0 &&
            confusion_matrix[1][1] >= 0) {
            precision_v =
                    static_cast<double>(confusion_matrix[0][0]) / (confusion_matrix[0][1] + confusion_matrix[0][0]);
        }
    }
    return precision_v;
}

double accuracy(const std::vector<std::vector<int>>& confusion_matrix) {
    double accuracy_v = 0.0;
    if (!confusion_matrix.empty() && confusion_matrix.size() == 2 && confusion_matrix[0].size() == 2) {
        if (confusion_matrix[0][0] >= 0 && confusion_matrix[0][1] >= 0 && confusion_matrix[1][0] >= 0 &&
             confusion_matrix[1][1] >= 0) {
            accuracy_v = static_cast<double>(confusion_matrix[0][0] + confusion_matrix[1][1]) /
                         (confusion_matrix[0][0] + confusion_matrix[0][1] + confusion_matrix[1][0] +
                          confusion_matrix[1][1]);
        }
    }
    return accuracy_v;
}

double recall(const std::vector<std::vector<int>>& confusion_matrix) {
    double recall_v = 0.0;
    if (!confusion_matrix.empty() && confusion_matrix.size() == 2 && confusion_matrix[0].size() == 2) {
        if (confusion_matrix[0][0] >= 0 && confusion_matrix[0][1] >= 0 && confusion_matrix[1][0] >= 0 &&
            confusion_matrix[1][1] >= 0) {
            recall_v = static_cast<double>(confusion_matrix[0][0]) / (confusion_matrix[0][0] + confusion_matrix[1][0]);
        }
    }
    return recall_v;
}

double f1_score(const std::vector<std::vector<int>>& confusion_matrix) {
    double precision_val = precision(confusion_matrix);
    double recall_val = recall(confusion_matrix);
    double f1 = 0.0;
    if (precision_val > 0 || recall_val > 0)
            f1 = 2 * ((precision_val * recall_val) / (precision_val + recall_val));
    return f1;
}


std::vector<std::vector<int>> compute_confusion_matrix(const std::vector<int>& true_labels,
                                                       const std::vector<int>& predicted_labels,
                                                       const std::vector<int>& labels) {
    /**
     * Computes the confusion matrix.
     *
     * @param true_labels The true labels.
     * @param predicted_labels The predicted labels.
     * @param labels The list of unique labels.
     * @return The confusion matrix.
     */
    int num_classes = labels.size();
    std::vector<std::vector<int>> confusion_matrix(num_classes, std::vector<int>(num_classes, 0));

    // Create a dictionary to map labels to their index
    std::unordered_map<int, int> label_to_index;
    for (int i = 0; i < num_classes; i++) {
        label_to_index[labels[i]] = i;
    }

    for (int i = 0; i < true_labels.size(); i++) {
        int true_index = label_to_index[true_labels[i]];
        int predicted_index = label_to_index[predicted_labels[i]];
        confusion_matrix[true_index][predicted_index]++;
    }

    return confusion_matrix;
}

std::vector<std::vector<int>> compute_confusion_matrix(
        const std::vector<float>& true_labels,
        const std::vector<float>& predicted_labels,
        const std::vector<float>& labels) {
    /**
     * Computes the confusion matrix.
     *
     * @param true_labels The true labels.
     * @param predicted_labels The predicted labels.
     * @param labels The list of unique labels.
     * @return The confusion matrix.
     */
    int num_classes = static_cast<int>(labels.size());
    std::vector<std::vector<int>> confusion_matrix(num_classes, std::vector<int>(num_classes, 0));

    // Create a dictionary to map labels to their index
    std::unordered_map<float, int> label_to_index;
    for (int i = 0; i < num_classes; i++) {
        label_to_index[labels[i]] = i;
    }

    if (true_labels.size() != predicted_labels.size()) {
        throw std::runtime_error("True labels and predicted labels must have the same size.");
    }

    for (size_t i = 0; i < true_labels.size(); i++) {
        int true_index = label_to_index[true_labels[i]];
        int predicted_index = label_to_index[predicted_labels[i]];
        confusion_matrix[true_index][predicted_index]++;
    }

    return confusion_matrix;
}

std::unordered_map<std::string, int> count_elements(const std::vector<float>& lst) {
    /**
     * Counts the number of elements in a list.
     *
     * @param lst The list of elements to count.
     * @return The count dictionary.
     */
    std::unordered_map<std::string, int> count_dict;
    for (const float& element : lst) {
        std::string elem_str = std::to_string(element);
        if (count_dict.find(elem_str) != count_dict.end()) {
            count_dict[elem_str]++;
        } else {
            count_dict[elem_str] = 1;
        }
    }
    return count_dict;
}

