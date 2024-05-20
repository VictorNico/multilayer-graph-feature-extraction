import math


def gini_impurity(y):
    """
    Calculates the Gini impurity of a binary classification problem
    :param y: list of labels
    :return: gini
    """

    if isinstance(y, list):
        # Initialize an empty dictionary to store the counts
        count_dict = {}

        # Iterate over each value in the list
        for value in y:
            # Check if the value is already present in the dictionary
            if value in count_dict:
                # If present, increment the count by 1
                count_dict[value] += 1
            else:
                # If not present, add the value to the dictionary with a count of 1
                count_dict[value] = 1

        p = [val / len(y) for (key, val) in count_dict.items()]
        gini = 1 - sum([p1 ** 2 for p1 in p])
        return gini

    else:
        raise 'Object must be a Python list.'


def entropy(y):
    """
    Given a Python List, it calculates the Shannon Entropy
    :param y: list of labels
    :return: entrop
    """

    if isinstance(y, list):
        # Initialize an empty dictionary to store the counts
        count_dict = {}

        # Iterate over each value in the list
        for value in y:
            # Check if the value is already present in the dictionary
            if value in count_dict:
                # If present, increment the count by 1
                count_dict[value] += 1
            else:
                # If not present, add the value to the dictionary with a count of 1
                count_dict[value] = 1

        p = [val / len(y) for (key, val) in count_dict.items()]
        entrop = 0.0
        epsilon = 1e-9

        for value in p:
            value = max(value, epsilon)  # Ensure value is not zero to avoid log(0) issue
            entrop += -value * math.log2(value)

        return entrop

    else:
        raise 'Object must be a Python list.'


def variance(y):
    """
    Given a Python List, it calculates the Variance
    :param y: list of labels
    :return: variance
    """

    if len(y) == 1:
        return 0
    else:
        mean = sum(y) / len(y)
        variance = sum((xi - mean) ** 2 for xi in y) / (len(y) - 1)
        return variance


getMask = lambda X, cond, val: [cond(x, val) for x in X]
NumCond = lambda x, val: x < val
CatCond = lambda x, val: val in x


def information_gain(y, mask, func=entropy):
    '''
    It returns the Information Gain of a variable given a loss function.
    y: target variable.
    mask: split choice.
    func: function to be used to calculate Information Gain in case of classification.
    '''

    a = sum(mask)
    b = len(mask) - a

    if a == 0 or b == 0:
        ig = 0
    else:
        if isinstance(y[0], (int, float)):
            ig = variance(y) - (a / (a + b) * variance([y[i] for i in range(len(y)) if mask[i]])) - (
                    b / (a + b) * variance([y[i] for i in range(len(y)) if not mask[i]]))
        else:
            ig = func(y) - a / (a + b) * func([y[i] for i in range(len(y)) if mask[i]]) - b / (a + b) * func(
                [y[i] for i in range(len(y)) if not mask[i]])

    return ig


def generate_combinations(nums, start, path, result):
    """
    Generates all possible combinations of given numbers.
    :param nums: list of numbers
    :param start: start index
    :param path: all possible paths
    :param result: vector of result
    :return:
    """
    if start == len(nums):
        if path:
            result.append(path)
        return

    generate_combinations(nums, start + 1, path + [nums[start]], result)
    generate_combinations(nums, start + 1, path, result)


def categorical_options(a):
    '''
    Creates all possible combinations from a list.
    a: List from where to get all possible combinations.
    '''

    unique_vals = list(set(a))
    options = []
    generate_combinations(unique_vals, 0, [], options)
    return options[1:-1]


def max_information_gain_split(x, y, func=entropy):
    '''
    Given a predictor & target variable, returns the best split, the error, and the type of variable based on a selected cost function.
    x: predictor variable as a list.
    y: target variable as a list.
    func: function to be used to calculate the best split.
    '''

    split_value = []
    ig = []

    numeric_variable = True if isinstance(x[0], (int, float)) else False

    if numeric_variable:
        options = sorted(list(set(x)))[1:]
    else:
        options = list(set(x))

    for val in options:
        mask = getMask(x, NumCond, val) if numeric_variable else getMask(x, CatCond, val)
        val_ig = information_gain(y, mask, func)
        ig.append(val_ig)
        split_value.append(val)

    if len(ig) == 0:
        return None, None, None, False
    else:
        best_ig = max(ig)
        best_ig_index = ig.index(best_ig)
        best_split = split_value[best_ig_index]
        return best_ig, best_split, numeric_variable, True


def get_best_split(y, data):
    '''
    Given a data, select the best split and return the variable, the value, the variable type, and the information gain.
    y: name of the target variable
    data: dictionary where to find the best split.
    '''

    masks = {}
    for column in data.keys():
        if column != y:
            x = data[column]
            ig, split_value, numeric_variable, flag = max_information_gain_split(x, data[y])
            masks[column] = [ig, split_value, numeric_variable, flag]

    if sum([flag for _, (_, _, _, flag) in masks.items()]) == 0:
        return None, None, None, None
    else:
        valid_columns = [(column, ig) for column, (ig, _, _, flag) in masks.items() if flag is True]
        valid_columns.sort(key=lambda x: x[1], reverse=True)

        split_variable = valid_columns[0][0]
        split_value = masks[split_variable][1]
        split_ig = masks[split_variable][0]
        split_numeric = masks[split_variable][2]

        return split_variable, split_value, split_ig, split_numeric


def make_split(variable, value, data, is_numeric):
    '''
  Given a data and a split conditions, do the split.
  variable: variable with which make the split.
  value: value of the variable to make the split.
  data: data to be splitted.
  is_numeric: boolean considering if the variable to be splitted is numeric or not.
  '''
    if is_numeric:
        index = [i for i, el in enumerate(data[variable]) if el < value]
        index2 = list(set(range(len(data[variable]))) - set(index))
        data_1 = {key: [data[key][i] for i in index] for key in data.keys()}
        data_2 = {key: [data[key][i] for i in index2] for key in data.keys()}
    else:
        index = [i for i, el in enumerate(data[variable]) if value in el]
        index2 = list(set(range(len(data[variable]))) - set(index))
        data_1 = {key: [data[key][i] for i in index] for key in data.keys()}
        data_2 = {key: [data[key][i] for i in index2] for key in data.keys()}
    return data_1, data_2


def make_prediction(data, target_factor):
    '''
    Given the target variable, make a prediction.
    :param data: pandas series for target variable
    :param target_factor: boolean considering if the variable is a factor or not
    :return: prediction result
    '''

    # Make predictions
    if target_factor:
        pred = max(set(data), key=data.count)
    else:
        pred = data.mean()

    return pred

def init_tree(xy_current):
    stack = []  # Stack to store nodes to be processed
    root = {
        'data': xy_current,
        'depth': 0,
        'node': {}
    }
    stack.append(root)
    return stack,root

def check_conditions(depth,xy_current,max_categories,max_depth,counter,min_samples_split):
    """
    check conditions for a decision tree
    :param depth:
    :param xy_current:
    :param max_categories:
    :param max_depth:
    :param counter:
    :param min_samples_split:
    :return:
    """
    if depth == 0:
        types = [(key,type(xy_current[key][0])) for key in list(xy_current.keys())]
        check_columns = [i for i, type in types if type == "object"]
        for column in check_columns:
            var_length = len(xy_current[column].value_counts())
            if var_length > max_categories:
                raise ValueError('The variable ' + column + ' has ' + str(
                    var_length) + ' unique values, which is more than the accepted ones: ' + str(max_categories))

    # Check for depth conditions
    if max_depth == None:
        depth_cond = True

    else:
        if counter < max_depth:
            depth_cond = True

        else:
            depth_cond = False

    # Check for sample conditions
    if min_samples_split == None:
        sample_cond = True

    else:
        if len(xy_current[list(xy_current.keys())[0]]) > min_samples_split:
            sample_cond = True

        else:
            sample_cond = False

    return depth_cond, sample_cond


def sub_tree(current_node,var,ig,question, depth,val,stack,left,right):
    """
    Given a current node, a variable, a predictor and a predictor variable,
    :param current_node:
    :param var:
    :param ig:
    :param question:
    :param depth:
    :param val:
    :param stack:
    :param left:
    :param right:
    :return:
    """
    # Update current node with split information

    current_node['col'] = var
    current_node['cutoff'] = ig
    current_node['val'] = val
    current_node['condition'] = question
    current_node['depth'] = depth

    # Create left and right child nodes
    current_node['left'] = {}
    current_node['right'] = {}

    # Push child nodes onto the stack
    stack.append({
        'data': left,
        'depth': depth + 1,
        'node': current_node['left']
    })
    stack.append({
        'data': right,
        'depth': depth + 1,
        'node': current_node['right']
    })

def leaf_tree(xy_current,y,target_factor,var,ig,val,current_node):
    pred = make_prediction(xy_current[y], target_factor)
    current_node['col'] = var
    current_node['cutoff'] = ig
    current_node['val'] = val
    current_node['condition'] = pred

def interative_train_tree(data, y, target_factor, max_depth=None, min_samples_split=None, min_information_gain=1e-20,
                          counter=0, max_categories=20):
    '''
        Trains a Decission Tree
        data: Data to be used to train the Decission Tree
        y: target variable column name
        target_factor: boolean to consider if target variable is factor or numeric.
        max_depth: maximum depth to stop splitting.
        min_samples_split: minimum number of observations to make a split.
        min_information_gain: minimum ig gain to consider a split to be valid.
        max_categories: maximum number of different values accepted for categorical values. High number of values will slow down learning process. R
    '''

    stack,root = init_tree(data)
    while len(stack) != 0:

        current = stack.pop()
        xy_current = current['data']
        depth = current['depth']
        current_node = current['node']


        depth_cond, sample_cond = check_conditions(depth,xy_current,max_categories,max_depth,counter,min_samples_split)
        # Check for ig condition
        if depth_cond & sample_cond:

            var, val, ig, var_type = get_best_split(y, xy_current)

            # If ig condition is fulfilled, make split
            if ig is not None and ig >= min_information_gain:
                left, right = make_split(var, val, xy_current, var_type)

                # Instantiate sub-tree
                split_type = "<=" if var_type else "in"
                question = "{} {}  {}".format(var, split_type, val)

                sub_tree(current_node, var, ig, question, depth, val, stack, left, right)

            # If it doesn't match IG condition, make prediction
            else:
                leaf_tree(xy_current, y, target_factor, var, ig, val, current_node)


        # Drop dataset if doesn't match depth or sample conditions
        else:
            leaf_tree(xy_current,y,target_factor,None,None,None,current_node)

    return root['node']

def predict(observation, tree):

    # print(arbol.keys())
    question = tree['condition']

    if question.split()[1] == '<=':

        if observation[question.split()[0]] <= float(question.split()[2]):
            answer = tree['left']
        else:
            answer = tree['right']

    else:

        if observation[question.split()[0]] in (question.split()[2]):
            answer = tree['left']
        else:
            answer = tree['right']

    # If the answer is not a dictionary
    if not isinstance(answer['condition'], str):
        return answer['condition']

    return predict(observation, answer)
