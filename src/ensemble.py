
from sklearn.metrics import precision_recall_curve
from sklearn.preprocessing import MinMaxScaler
import copy
from numpy import linalg as LA
import array
import random
import numpy as np
from deap import base, algorithms, creator, tools
from sklearn.metrics import auc
from sklearn import linear_model
import networkx as nx
import math

from model import Model


class EnsembleModel(Model):
    chem_sim_similarity_matrix = None
    target_similarity_matrix = None
    transporter_similarity_matrix = None
    enzyme_similarity_matrix = None
    pathway_similarity_matrix = None
    indication_similarity_matrix = None
    label_similarity_matrix = None
    offlabel_similarity_matrix = None

    def __init__(self, train_drug_drug_matrix, chem_sim_similarity_matrix, target_similarity_matrix,
                 transporter_similarity_matrix, enzyme_similarity_matrix, pathway_similarity_matrix,
                 indication_similarity_matrix, label_similarity_matrix, offlabel_similarity_matrix):

        super().__init__(train_drug_drug_matrix)
        self.train_drug_drug_matrix = train_drug_drug_matrix
        self.chem_sim_similarity_matrix = chem_sim_similarity_matrix
        self.target_similarity_matrix = target_similarity_matrix
        self.transporter_similarity_matrix = transporter_similarity_matrix
        self.enzyme_similarity_matrix = enzyme_similarity_matrix
        self.pathway_similarity_matrix = pathway_similarity_matrix
        self.indication_similarity_matrix = indication_similarity_matrix
        self.label_similarity_matrix = label_similarity_matrix
        self.offlabel_similarity_matrix = offlabel_similarity_matrix

    def calculate(self, full_drug_drug_matrix, test_position):
        multiple_matrix = []
        multiple_result = []
        print('***************************Calculating*****************************')
        predict_matrix = self._neighbor_method(self.chem_sim_similarity_matrix)
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'chem_neighbor')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)

        predict_matrix = self._neighbor_method(self.target_similarity_matrix)
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'target_neighbor')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)

        predict_matrix = self._neighbor_method(self.transporter_similarity_matrix)
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'transporter_neighbor')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)

        predict_matrix = self._neighbor_method(self.enzyme_similarity_matrix)
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'enzyme_neighbor')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)

        predict_matrix = self._neighbor_method(self.pathway_similarity_matrix)
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'pathway_neighbor')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)

        predict_matrix = self._neighbor_method(self.indication_similarity_matrix)
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'indication_neighbor')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)

        predict_matrix = self._neighbor_method(self.label_similarity_matrix)
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'label_neighbor')
        multiple_result.append(results)
        results = multiple_matrix.append(predict_matrix)

        predict_matrix = self._neighbor_method(self.offlabel_similarity_matrix)
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'offlabel_neighbor')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)

        # print('*************************************************************************************************************************************')
        common_similarity_matrix, AA_similarity_matrix, RA_similarity_matrix, Katz_similarity_matrix, ACT_similarity_matrix, RWR_similarity_matrix = self._topology_similarity_matrix()
        predict_matrix = self._neighbor_method(common_similarity_matrix)
        # predict_matrix=common_similarity_matrix
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'common_similarity_neighbor')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)

        predict_matrix = self._neighbor_method(AA_similarity_matrix)
        # predict_matrix=AA_similarity_matrix
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'AA_similarity_neighbor')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)

        predict_matrix = self._neighbor_method(RA_similarity_matrix)
        # predict_matrix=RA_similarity_matrix
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'RA_similarity_neighbor')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)

        predict_matrix = self._neighbor_method(Katz_similarity_matrix)
        # predict_matrix=Katz_similarity_matrix
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'Katz_similarity_neighbor')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)

        predict_matrix = self._neighbor_method(ACT_similarity_matrix)
        # predict_matrix=ACT_similarity_matrix
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'ACT_similarity_neighbor')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)

        predict_matrix = self._neighbor_method(RWR_similarity_matrix)
        # predict_matrix=RWR_similarity_matrix
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'RWR_similarity_neighbor')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)

        # print('*************************************************************************************************************************************')
        predict_matrix = self._label_propagation(self.chem_sim_similarity_matrix)
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'chem_label')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)  # 12

        predict_matrix = self._label_propagation(self.target_similarity_matrix)
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'target_label')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)

        predict_matrix = self._label_propagation(self.transporter_similarity_matrix)
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'transporter_label')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)

        predict_matrix = self._label_propagation(self.enzyme_similarity_matrix)
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'enzyme_label')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)

        predict_matrix = self._label_propagation(self.pathway_similarity_matrix)
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'pathway_label')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)

        predict_matrix = self._label_propagation(self.indication_similarity_matrix)
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'indication_label')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)

        predict_matrix = self._label_propagation(self.label_similarity_matrix)
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'label_label')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)

        predict_matrix = self._label_propagation(self.offlabel_similarity_matrix)
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'offlabel_label')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)  # 14

        # print('*************************************************************************************************************************************')
        predict_matrix = self._label_propagation(common_similarity_matrix)
        # predict_matrix=common_similarity_matrix
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'common_similarity_label')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)

        predict_matrix = self._label_propagation(AA_similarity_matrix)
        # predict_matrix=AA_similarity_matrix
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'AA_similarity_label')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)

        predict_matrix = self._label_propagation(RA_similarity_matrix)
        # predict_matrix=RA_similarity_matrix
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'RA_similarity_label')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)

        predict_matrix = self._label_propagation(Katz_similarity_matrix)
        # predict_matrix=Katz_similarity_matrix
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'Katz_similarity_label')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)

        predict_matrix = self._label_propagation(ACT_similarity_matrix)
        # predict_matrix=ACT_similarity_matrix
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'ACT_similarity_label')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)

        predict_matrix = self._label_propagation(RWR_similarity_matrix)
        # predict_matrix=RWR_similarity_matrix
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'RWR_similarity_label')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)

        # print('*************************************************************************************************************************************')
        predict_matrix = self._disturb_matrix_method()
        results = super().evaluate(full_drug_drug_matrix, predict_matrix, test_position, 'disturb_matrix_label')
        multiple_result.append(results)
        multiple_matrix.append(predict_matrix)

        return multiple_matrix, multiple_result

    def train(self):
        full_drug_drug_matrix = copy.deepcopy(self.train_drug_drug_matrix)
        train_drug_drug_matrix, test_position = super()._split_training_data(0.2, 1)
        self.train_drug_drug_matrix = train_drug_drug_matrix
        [multiple_matrix, multiple_result] = self.calculate(copy.deepcopy(full_drug_drug_matrix), test_position)
        weights = self._get_paramter(copy.deepcopy(full_drug_drug_matrix), multiple_matrix, test_position)
        # weights=[]

        input_matrix = []
        output_matrix = []
        for i in range(0, len(test_position)):
            vector = []
            for j in range(0, len(multiple_matrix)):
                vector.append(multiple_matrix[j][test_position[i][0], test_position[i][1]])
            input_matrix.append(vector)
            output_matrix.append(full_drug_drug_matrix[test_position[i][0], test_position[i][1]])

        input_matrix = np.array(input_matrix)
        output_matrix = np.array(output_matrix)
        clf1 = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6, solver='liblinear')
        clf1.fit(input_matrix, output_matrix)

        clf2 = linear_model.LogisticRegression(C=1.0, penalty='l2', tol=1e-6)
        clf2.fit(input_matrix, output_matrix)
        print('*************************parameter determined*************************')
        # return weights
        self.train_drug_drug_matrix = full_drug_drug_matrix
        return weights, clf1, clf2

    @staticmethod
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

        result = Model.calculate_metric_score(real_labels, (np.array(ensemble_prediction).reshape(-1)))
        result_cf1 = Model.calculate_metric_score(real_labels, ensemble_prediction_cf1)
        result_cf2 = Model.calculate_metric_score(real_labels, ensemble_prediction_cf2)

        return result, result_cf1, result_cf2
    @staticmethod
    def fit_function(individual, parameter1, parameter2):
        real_labels = parameter1
        multiple_prediction = parameter2
        ensemble_prediction = np.zeros(len(real_labels))
        for i in range(0, len(multiple_prediction)):
            ensemble_prediction = ensemble_prediction + individual[i] * multiple_prediction[i]
        precision, recall, pr_thresholds = precision_recall_curve(real_labels, ensemble_prediction)
        aupr_score = auc(recall, precision)
        return (aupr_score),

    @staticmethod
    def _get_paramter(real_matrix, multiple_matrix, test_position):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", array.array, typecode='d', fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        # Attribute generator
        toolbox.register("attr_float", random.uniform, 0, 1)
        # Structure initializers
        variable_num = len(multiple_matrix)
        toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, variable_num)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        #################################################################################################
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
            multiple_prediction.append(np.array(predicted_probability).reshape(-1))

        #################################################################################################
        toolbox.register("evaluate", EnsembleModel.fit_function, parameter1=real_labels, parameter2=multiple_prediction)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=3)

        random.seed(0)
        pop = toolbox.population(n=100)
        hof = tools.HallOfFame(1)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2, ngen=50,
                                       stats=stats, halloffame=hof, verbose=True)
        pop.sort(key=lambda ind: ind.fitness, reverse=True)
        print(pop[0])
        return pop[0]

    def _topology_similarity_matrix(self):
        drug_drug_matrix = np.matrix(self.train_drug_drug_matrix)
        g = nx.from_numpy_array(drug_drug_matrix)
        drug_num = len(drug_drug_matrix)
        common_similarity_matrix = np.zeros(shape=(drug_num, drug_num))
        aa_similarity_matrix = np.zeros(shape=(drug_num, drug_num))
        ra_similarity_matrix = np.zeros(shape=(drug_num, drug_num))

        eigen_values, eigen_vectors = LA.eig(drug_drug_matrix)
        idx = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[idx[0]]

        beta = 0.5 * (1 / eigen_values)
        katz_similarity_matrix = LA.pinv(np.identity(drug_num) - beta * drug_drug_matrix) - np.identity(drug_num)
        for i in range(0, drug_num):
            for j in range(i + 1, drug_num):
                commonn_neighbor = list(nx.common_neighbors(g, i, j))
                common_similarity_matrix[i][j] = len(commonn_neighbor)
                aa_score = 0
                ra_score = 0
                for k in range(0, len(commonn_neighbor)):
                    aa_score = aa_score + 1 / math.log(len(list(g.neighbors(commonn_neighbor[k]))))
                    ra_score = ra_score + 1 / len(list(g.neighbors(commonn_neighbor[k])))
                aa_similarity_matrix[i][j] = aa_score
                ra_similarity_matrix[i][j] = ra_score

                common_similarity_matrix[j][i] = common_similarity_matrix[i][j]
                aa_similarity_matrix[j][i] = aa_similarity_matrix[i][j]
                ra_similarity_matrix[j][i] = ra_similarity_matrix[i][j]

        d = np.diag(((drug_drug_matrix.sum(axis=1)).getA1()))
        l = d - drug_drug_matrix
        ll = LA.pinv(l)
        ll = np.matrix(ll)
        act_similarity_matrix = np.zeros(shape=(drug_num, drug_num))
        for i in range(0, drug_num):
            for j in range(i + 1, drug_num):
                act_similarity_matrix[i][j] = 1 / (ll[i, i] + ll[j, j] - 2 * ll[i, j])
                act_similarity_matrix[j][i] = act_similarity_matrix[i][j]

        d = np.diag(((drug_drug_matrix.sum(axis=1)).getA1()))
        n = LA.pinv(d) * drug_drug_matrix
        alpha = 0.9
        rwr_similarity_matrix = (1 - alpha) * LA.pinv(np.identity(drug_num) - alpha * n)
        rwr_similarity_matrix = rwr_similarity_matrix + np.transpose(rwr_similarity_matrix)

        return np.matrix(common_similarity_matrix), np.matrix(aa_similarity_matrix), np.matrix(
            ra_similarity_matrix), np.matrix(katz_similarity_matrix), np.matrix(act_similarity_matrix), np.matrix(
            rwr_similarity_matrix)

    def _neighbor_method(self, similarity_matrix):
        return_matrix = np.matrix(self.train_drug_drug_matrix) * np.matrix(similarity_matrix)
        d = np.diag(((similarity_matrix.sum(axis=1)).getA1()))
        return_matrix = return_matrix * LA.pinv(d)
        return_matrix = return_matrix + np.transpose(return_matrix)
        return return_matrix

    def _label_propagation(self, similarity_matrix):
        alpha = 0.9
        similarity_matrix = np.matrix(similarity_matrix)
        train_drug_drug_matrix = np.matrix(self.train_drug_drug_matrix)
        d = np.diag(((similarity_matrix.sum(axis=1)).getA1()))
        n = LA.pinv(d) * similarity_matrix

        transform_matrix = (1 - alpha) * LA.pinv(np.identity(len(similarity_matrix)) - alpha * n)
        return_matrix = transform_matrix * train_drug_drug_matrix
        return_matrix = return_matrix + np.transpose(return_matrix)
        return return_matrix

    def _generate_distrub_matrix(self):
        a = np.matrix(self.train_drug_drug_matrix)
        [num, num] = a.shape
        upper_a = np.triu(a, k=0)
        [row_index, col_index] = np.where(upper_a == 1)

        ratio = 0.1  # disturb how many links are removed
        select_num = int(len(row_index) * ratio)
        index = np.arange(0, (upper_a.sum()).sum())
        # print(index.shape)

        random.seed(0)
        random.shuffle(index)
        # np.random.shuffle(index)
        select_index = index[0:select_num]
        delta_a = np.zeros(shape=(num, num))
        for i in range(0, select_num):
            delta_a[row_index[select_index[i]]][col_index[select_index[i]]] = 1
            delta_a[col_index[select_index[i]]][row_index[select_index[i]]] = 1

        return delta_a, row_index, col_index, select_num

    def _disturb_matrix_method(self):
        input_a = np.matrix(self.train_drug_drug_matrix)
        [num, num] = input_a.shape
        delta_a, row_index, col_index, select_num = self._generate_distrub_matrix()
        a = input_a - delta_a
        eigenvalues, eigenvectors = LA.eig(a)
        num_eigenvalues = len(eigenvalues)

        delta_eigenvalues = np.zeros(num_eigenvalues)
        for i in range(0, num_eigenvalues):
            delta_eigenvalues[i] = (np.transpose(eigenvectors[:, i]) * delta_a * eigenvectors[:, i]) / (
                    np.transpose(eigenvectors[:, i]) * eigenvectors[:, i])

        reconstructed_a = np.zeros(shape=(num, num))
        for i in range(0, num_eigenvalues):
            reconstructed_a = (reconstructed_a + (eigenvalues[i] + delta_eigenvalues[i]) *
                               eigenvectors[:, i] * np.transpose(eigenvectors[:, i]))

        reconstructed_a[np.where(input_a == 1)] = 1

        return_matrix = reconstructed_a + np.transpose(reconstructed_a)
        return return_matrix
