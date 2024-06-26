from matplotlib.pyplot import *
import csv


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


# Transforms data but keeps AB, BA duplicates
def transform_matrices(*matrices):
    stacked_matrices = np.stack(matrices, axis=0)
    num_matrices, rows, cols = stacked_matrices.shape

    # Reshape the stacked matrices to extract values
    reshaped_matrices = stacked_matrices.transpose(2, 1, 0).reshape(rows * cols, num_matrices)
    return reshaped_matrices


# Transforms data without duplicates
def transform_matrices_keep_lower(*matrices, stack=True):
    if stack:
        stacked_matrices = np.stack(matrices, axis=0)
        num_matrices, rows, cols = stacked_matrices.shape
    else:
        stacked_matrices = np.array(matrices)
        num_matrices, rows, cols = stacked_matrices.shape

    # Create a mask to identify the lower triangle of each input matrix
    lower_triangle_mask = np.tril(np.ones((rows, cols), dtype=bool))

    # Apply the mask
    masked_matrices = stacked_matrices[:, lower_triangle_mask]
    # Reshape the masked matrices to extract values
    reshaped_matrices = masked_matrices.transpose(1, 0).reshape(-1, num_matrices)
    return reshaped_matrices


def create_data(folder="dataset"):
    # Load all datasets
    chem = np.array(load_csv(folder + '/chem_Jacarrd_sim.csv', 'float'))
    target = np.array(load_csv(folder + '/target_Jacarrd_sim.csv', 'float'))
    transporter = np.array(load_csv(folder + '/transporter_Jacarrd_sim.csv', 'float'))
    enzyme = np.array(load_csv(folder + '/enzyme_Jacarrd_sim.csv', 'float'))
    pathway = np.array(load_csv(folder + '/pathway_Jacarrd_sim.csv', 'float'))
    indication = np.array(load_csv(folder + '/indication_Jacarrd_sim.csv', 'float'))
    side_effect = np.array(load_csv(folder + '/sideeffect_Jacarrd_sim.csv', 'float'))
    offside_effect = np.array(load_csv(folder + '/offsideeffect_Jacarrd_sim.csv', 'float'))

    X = transform_matrices_keep_lower(chem, target, transporter, enzyme, indication, side_effect, offside_effect)
    Y = transform_matrices_keep_lower(load_csv(folder + '/drug_drug_matrix.csv', 'float'), stack=False)
    return X, Y




