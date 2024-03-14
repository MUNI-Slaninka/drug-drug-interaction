from matplotlib.pyplot import *
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
import copy
import csv
import random
from sklearn.metrics import roc_curve
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score


def load_csv(filename, data_type):  # load csv, ignore the first row,type=int, data read as int， else float
    matrix_data = []
    with open(filename, 'r') as csvfile:
        csvreader = csv.reader(csvfile, delimiter=',')
        next(csvreader)
        for row_vector in csvreader:
            if data_type == 'int':
                matrix_data.append(list(map(int, row_vector[1:])))
            else:
                matrix_data.append(list(map(float, row_vector[1:])))
    return np.matrix(matrix_data)


def model_evaluation(real_matrix, predict_matrix, test_position, feature_name):  # compute cross validation results
    real_labels = []
    predicted_probability = []

    for i in range(0, len(test_position)):
        real_labels.append(real_matrix[test_position[i][0], test_position[i][1]])
        predicted_probability.append(predict_matrix[test_position[i][0], test_position[i][1]])

    normalize = MinMaxScaler()
    predicted_probability = normalize.fit_transform(np.array(predicted_probability).reshape(-1, 1))
    real_labels = np.array(real_labels)
    predicted_probability = np.array(predicted_probability).reshape(-1)

    print(len(test_position))
    print(list(set(np.array(real_matrix).reshape(-1))))
    precision, recall, pr_thresholds = precision_recall_curve(real_labels, predicted_probability)
    aupr_score = auc(recall, precision)

    all_F_measure = np.zeros(len(pr_thresholds))
    for k in range(0, len(pr_thresholds)):
        if (precision[k] + precision[k]) > 0:
            all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
        else:
            all_F_measure[k] = 0
    max_index = all_F_measure.argmax()
    threshold = pr_thresholds[max_index]

    fpr, tpr, auc_thresholds = roc_curve(real_labels, predicted_probability)
    auc_score = auc(fpr, tpr)
    predicted_score = np.zeros(len(real_labels))
    predicted_score[predicted_probability > threshold] = 1

    f = f1_score(real_labels, predicted_score)
    accuracy = accuracy_score(real_labels, predicted_score)
    precision = precision_score(real_labels, predicted_score)
    recall = recall_score(real_labels, predicted_score)
    print('results for feature:' + feature_name)
    print(
        '************************AUC score:%.3f, AUPR score:%.3f, recall score:%.3f, precision score:%.3f, accuracy:%.3f, f-measure:%.3f************************' % (
        auc_score, aupr_score, recall, precision, accuracy, f))
    auc_score, aupr_score, precision, recall, accuracy, f = ("%.4f" % auc_score), ("%.4f" % aupr_score), (
                "%.4f" % precision), ("%.4f" % recall), ("%.4f" % accuracy), ("%.4f" % f)
    results = [auc_score, aupr_score, precision, recall, accuracy, f]
    return results


def holdout_by_link(drug_drug_matrix, ratio, seed):
    link_number = 0
    link_position = []
    non_links_position = []  # all non-link position
    for i in range(0, len(drug_drug_matrix)):
        for j in range(i + 1, len(drug_drug_matrix)):
            if drug_drug_matrix[i, j] == 1:
                link_number = link_number + 1
                link_position.append([i, j])
            else:
                non_links_position.append([i, j])

    link_position = np.array(link_position)
    random.seed(seed)
    index = np.arange(0, link_number)
    random.shuffle(index)
    train_index = index[(int(link_number * ratio) + 1):]
    test_index = index[0:int(link_number * ratio)]
    train_index.sort()
    test_index.sort()
    test_link_position = link_position[test_index]
    train_drug_drug_matrix = copy.deepcopy(drug_drug_matrix)

    for i in range(0, len(test_link_position)):
        train_drug_drug_matrix[test_link_position[i, 0], test_link_position[i, 1]] = 0
        train_drug_drug_matrix[test_link_position[i, 1], test_link_position[i, 0]] = 0
    testPosition = list(test_link_position) + list(non_links_position)

    return train_drug_drug_matrix, testPosition


def ensemble_scoring(real_matrix, multiple_matrix, test_position, weights, cf1, cf2):
    real_labels = []
    for i in range(0, len(test_position)):
        real_labels.append(real_matrix[test_position[i][0], test_position[i][1]])

    multiple_prediction = []
    for i in range(0, len(multiple_matrix)):
        predicted_probability = []
        predict_matrix = multiple_matrix[i]
        for j in range(0, len(test_position)):
            predicted_probability.append(predict_matrix[test_position[j][0], test_position[j][1]])
        normalize = MinMaxScaler()
        predicted_probability = normalize.fit_transform(np.array(predicted_probability).reshape(-1, 1))
        predicted_probability = np.array(predicted_probability).reshape(-1)
        multiple_prediction.append(predicted_probability)
    ensemble_prediction = np.zeros(len(real_labels))
    for i in range(0, len(multiple_matrix)):
        ensemble_prediction = ensemble_prediction + weights[i] * multiple_prediction[i]

    ensemble_prediction_cf1 = np.zeros(len(real_labels))
    ensemble_prediction_cf2 = np.zeros(len(real_labels))
    for i in range(0, len(test_position)):
        vector = []
        for j in range(0, len(multiple_matrix)):
            vector.append(multiple_matrix[j][test_position[i][0], test_position[i][1]])
        vector = np.array(vector).reshape(1, -1)

        aa = cf1.predict_proba(vector)
        print(aa)
        ensemble_prediction_cf1[i] = (cf1.predict_proba(vector))[0][1]
        ensemble_prediction_cf2[i] = (cf2.predict_proba(vector))[0][1]

    normalize = MinMaxScaler()
    ensemble_prediction = normalize.fit_transform(np.array(ensemble_prediction).reshape(-1, 1))

    result = calculate_metric_score(real_labels, (np.array(ensemble_prediction). reshape(-1)))
    result_cf1 = calculate_metric_score(real_labels, ensemble_prediction_cf1)
    result_cf2 = calculate_metric_score(real_labels, ensemble_prediction_cf2)

    return result, result_cf1, result_cf2


def calculate_metric_score(real_labels, predict_score):
    precision, recall, pr_thresholds = precision_recall_curve(real_labels, predict_score)
    aupr_score = auc(recall, precision)

    all_F_measure = np.zeros(len(pr_thresholds))
    for k in range(0, len(pr_thresholds)):
        if (precision[k] + precision[k]) > 0:
            all_F_measure[k] = 2 * precision[k] * recall[k] / (precision[k] + recall[k])
        else:
            all_F_measure[k] = 0
    max_index = all_F_measure.argmax()
    threshold = pr_thresholds[max_index]
    fpr, tpr, auc_thresholds = roc_curve(real_labels, predict_score)
    auc_score = auc(fpr, tpr)

    predicted_score = np.zeros(len(real_labels))
    predicted_score[predict_score > threshold] = 1

    f = f1_score(real_labels, predicted_score)
    accuracy = accuracy_score(real_labels, predicted_score)
    precision = precision_score(real_labels, predicted_score)
    recall = recall_score(real_labels, predicted_score)
    print('results for feature:' + 'weighted_scoring')
    print(
        '************************AUC score:%.3f, AUPR score:%.3f, recall score:%.3f, precision score:%.3f, accuracy:%.3f************************' % (
            auc_score, aupr_score, recall, precision, accuracy))
    auc_score, aupr_score, precision, recall, accuracy, f = ("%.4f" % auc_score), ("%.4f" % aupr_score), (
                "%.4f" % precision), ("%.4f" % recall), ("%.4f" % accuracy), ("%.4f" % f)
    results = [auc_score, aupr_score, precision, recall, accuracy, f]
    return results
