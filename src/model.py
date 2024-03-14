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


class Model:
    drug_drug_matrix = None
    train_drug_drug_matrix = None
    links = None
    links_n = None
    test_position = None
    fold_num = None

    def __init__(self, drug_drug_matrix):
        self.drug_drug_matrix = drug_drug_matrix

    @staticmethod
    def evaluate(real_matrix, predict_matrix, test_position, feature_name):  # compute cross validation results
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

    def train(self):
        pass

    def _load_csv(self, filename, data_type):  # load csv, ignore the first row,type=int, data read as int， else float
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

    def _find_links(self):
        link_number = 0
        link_position = []
        non_links_position = []  # all non-link position
        for i in range(0, len(self.drug_drug_matrix)):
            for j in range(i + 1, len(self.drug_drug_matrix)):
                if self.drug_drug_matrix[i, j] == 1:
                    link_number = link_number + 1
                    link_position.append([i, j])
                else:
                    non_links_position.append([i, j])

        self.links_n = link_number
        self.links = np.array(link_position)
        self.non_links = np.array(non_links_position)


    def _split_training_data(self, ratio, seed):
        link_number = 0
        link_position = []
        non_links_position = []  # all non-link position
        for i in range(0, len(self.drug_drug_matrix)):
            for j in range(i + 1, len(self.drug_drug_matrix)):
                if self.drug_drug_matrix[i, j] == 1:
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
        train_drug_drug_matrix = copy.deepcopy(self.drug_drug_matrix)

        for i in range(0, len(test_link_position)):
            train_drug_drug_matrix[test_link_position[i, 0], test_link_position[i, 1]] = 0
            train_drug_drug_matrix[test_link_position[i, 1], test_link_position[i, 0]] = 0
        test_position = list(test_link_position) + list(non_links_position)

        self.train_drug_drug_matrix = train_drug_drug_matrix
        self.test_position = test_position

        return train_drug_drug_matrix, test_position

    @staticmethod
    def _calculate_metric_score(real_labels, predict_score):
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

    @staticmethod
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

