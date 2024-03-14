from matplotlib.pyplot import *
import copy
import random
from ensemble import ensemble_model
from utilities import load_csv, ensemble_scoring


def cross_validation(drug_drug_matrix, CV_num, seed):
    link_number = 0
    link_position = []
    nonLinksPosition = []  # all non-link position
    for i in range(0, len(drug_drug_matrix)):
        for j in range(i + 1, len(drug_drug_matrix)):
            if drug_drug_matrix[i, j] == 1:
                link_number = link_number + 1
                link_position.append([i, j])
            else:
                nonLinksPosition.append([i, j])

    link_position = np.array(link_position)
    random.seed(seed)
    index = np.arange(0, link_number)
    random.shuffle(index)

    fold_num = link_number // CV_num
    print(fold_num)

    for CV in range(0, CV_num):
        print('*********round:' + str(CV) + "**********\n")
        test_index = index[(CV * fold_num):((CV + 1) * fold_num)]
        test_index.sort()
        testLinkPosition = link_position[test_index]
        train_drug_drug_matrix = copy.deepcopy(drug_drug_matrix)
        for i in range(0, len(testLinkPosition)):
            train_drug_drug_matrix[testLinkPosition[i, 0], testLinkPosition[i, 1]] = 0
            train_drug_drug_matrix[testLinkPosition[i, 1], testLinkPosition[i, 0]] = 0
            testPosition = list(testLinkPosition) + list(nonLinksPosition)
        #  GA
        model = ensemble_model(copy.deepcopy(train_drug_drug_matrix),
                               load_csv('dataset/chem_Jacarrd_sim.csv', 'float'),
                               load_csv('dataset/target_Jacarrd_sim.csv', 'float'),
                               load_csv('dataset/transporter_Jacarrd_sim.csv', 'float'),
                               load_csv('dataset/enzyme_Jacarrd_sim.csv', 'float'),
                               load_csv('dataset/pathway_Jacarrd_sim.csv', 'float'),
                               load_csv('dataset/indication_Jacarrd_sim.csv', 'float'),
                               load_csv('dataset/sideeffect_Jacarrd_sim.csv', 'float'),
                               load_csv('dataset/offsideeffect_Jacarrd_sim.csv', 'float'))

        weights, cf1, cf2 = model.determine_ensemble_parameter()
        # cf1,cf2=internal_determine_parameter(copy.deepcopy(train_drug_drug_matrix))

        [multiple_predict_matrix, multiple_predict_results] = model.calculate(copy.deepcopy(drug_drug_matrix), testPosition)

        # logstic weight

        ensemble_results, ensemble_results_cf1, ensemble_results_cf2 = ensemble_scoring(copy.deepcopy(drug_drug_matrix),
                                                                                        multiple_predict_matrix,
                                                                                        testPosition, weights, cf1, cf2)
        for i in range(0, len(multiple_predict_results)):
            [auc_score, aupr_score, precision, recall, accuracy, f] = multiple_predict_results[i]
            file_results.write(
                auc_score + ' ' + aupr_score + ' ' + precision + ' ' + recall + ' ' + accuracy + ' ' + f + "\n")
            file_results.flush()

        [auc_score, aupr_score, precision, recall, accuracy, f] = ensemble_results
        file_results.write(
            auc_score + ' ' + aupr_score + ' ' + precision + ' ' + recall + ' ' + accuracy + ' ' + f + "\n")
        file_results.flush()

        [auc_score, aupr_score, precision, recall, accuracy, f] = ensemble_results_cf1
        file_results.write(
            auc_score + ' ' + aupr_score + ' ' + precision + ' ' + recall + ' ' + accuracy + ' ' + f + "\n")
        file_results.flush()

        [auc_score, aupr_score, precision, recall, accuracy, f] = ensemble_results_cf2
        file_results.write(
            auc_score + ' ' + aupr_score + ' ' + precision + ' ' + recall + ' ' + accuracy + ' ' + f + "\n")
        file_results.flush()

        weights_str = ''
        for i in range(0, len(weights)):
            weights_str = weights_str + ' ' + str(weights[i])
        file_weights.write(weights_str + "\n")
        file_results.flush()
        file_weights.flush()


if __name__ == '__main__':
    runtimes = 1
    cv_num = 3 # implement x runs of y-fold cross validation for base predictors and the ensmeble model

    drug_drug_matrix = load_csv('dataset/drug_drug_matrix.csv', 'int')
    file_results_str = "result/result_on_our_dataset_5CV"
    weights_results_str = "result/weights_on_our_dataset_5CV"
    for seed in range(0, runtimes):
        file_results_path = file_results_str + "_" + str(seed) + ".txt"
        weights_results_path = weights_results_str + "_" + str(seed) + ".txt"
        file_results = open(file_results_path, "w")
        file_weights = open(weights_results_path, "w")
        cross_validation(drug_drug_matrix, cv_num, seed)
