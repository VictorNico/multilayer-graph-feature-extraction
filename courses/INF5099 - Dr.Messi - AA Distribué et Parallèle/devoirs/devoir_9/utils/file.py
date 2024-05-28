import random
from collections import defaultdict
import time
def process_file(file_path, details_separator, header_flag):
    """

    :param file_path: path to file
    :param details_separator: which separator to use for details scission
    :param header_flag: does file content headers' informations
    :return: descriptions
    """
    # Read the text file
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Determine the start index based on the presence of a header line
    start_index = 1 if header_flag else 0

    # Retrieve the header line if it exists
    header = None
    if header_flag:
        header = [elt.strip() if elt != '' else 'Index' for elt in lines[0].split(details_separator)]

    # Process each line and extract the descriptions
    descriptions = {}
    for line in lines[start_index:]:
        if line:
            fields = line.split(details_separator)

            for i, field in enumerate(fields):
                # Trim the field and convert to the appropriate type
                field = field.strip()
                if header:
                    if sum([header[i] == key for key in descriptions.keys()]) == 0:
                        descriptions[header[i]] = []
                    field_name = header[i]
                    descriptions[field_name].append(cast_to_appropriate_type(field))
                else:
                    if sum([i == key for key in descriptions.keys()]) == 0:
                        descriptions[i] = []
                    descriptions[i].append(cast_to_appropriate_type(field))
    descriptions['Index'] = list(range(len(descriptions[list(descriptions.keys())[0]])))
    return descriptions

def cast_to_appropriate_type(data):
    if data.isdigit():
        return int(data)
    try:
        return float(data)
    except ValueError:
        if data.lower() == 'true':
            return True
        elif data.lower() == 'false':
            return False
        else:
            return data

def train_test_split(data, labels, test_size=0.2, random_state=None):
    """
    splits data into train and test sets
    :param data: original data
    :param labels: labels for train and test sets
    :param test_size: size of test set
    :param random_state: random state
    :return:
    """
    class_samples = defaultdict(list)
    for sample, label in zip(data['Index'], labels):
        class_samples[label].append(sample)
    # print(class_samples)
    test_data = {"Index": []}
    train_data = {"Index": []}
    y_test = []
    y_train = []

    if random_state is not None:
        random.seed(random_state)

    for classe, class_samples_list in class_samples.items():
        random.shuffle(class_samples_list)
        # print(class_samples_list, classe, len(class_samples_list))
        i = 0
        num_test_samples = int(test_size * len(class_samples_list))
        # print(num_test_samples)
        index_te = class_samples_list[:num_test_samples]
        index_tr = class_samples_list[num_test_samples:]
        # print(f"""
        # __________________
        # {index_tr} @@@
        # {index_te} @@@
        # """)
        for ele in list(data.keys()):
            if sum([(ele == k) for k in list(train_data.keys())]) == 0:
                train_data[ele] = []
                test_data[ele] = []
            if "Index" not in ele:
                test_data[ele].extend([data[ele][i] for i in index_te])
                train_data[ele].extend([data[ele][i] for i in index_tr])

        y_test.extend([classe] * num_test_samples)
        y_train.extend([classe] * (len(class_samples_list) - num_test_samples))
        train_data["Index"].extend(index_tr)
        test_data["Index"].extend(index_te)
    return train_data, test_data, y_train, y_test

def write_file(file_path, line):
    with open(file_path, 'a') as file:
        file.write(line)


def save_metrics_to_csv(filename, metric_names, metric_values):
    # Vérifier que le nombre de métriques correspond
    if len(metric_names) != len(metric_values):
        print("Nombre de noms de métriques et de valeurs de métriques différents.")
        return

    # Obtenir l'heure actuelle
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S')

    # Ouvrir le fichier en mode ajout
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as file:
        # Écrire l'en-tête s'il n'existe pas déjà
        if not file_exists:
            file.write(','.join(['Temps'] + metric_names) + '\n')

        # Écrire les données de métriques
        row = [timestamp] + [str(value) for value in metric_values]
        file.write(','.join(row) + '\n')

    print(f"Métriques enregistrées dans le fichier : {filename}")