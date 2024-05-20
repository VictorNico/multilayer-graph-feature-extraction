#!/usr/bin/env python
# coding: utf-8

# # Python Sequential Decision Tree

# ## [Version 1](https://anderfernandez.com/en/blog/code-decision-tree-python-from-scratch/)

# ![](https://sp-ao.shortpixel.ai/client/to_webp,q_glossy,ret_img,w_600/https://anderfernandez.com/wp-content/uploads/2021/01/image.png)

# In[1]:


import numpy as np
import pandas as pd
# from IPython.core.display import HTML
import itertools


# In[2]:


dataframe = pd.read_csv('500_Person_Gender_Height_Weight_Index.csv')
dataframe.head()


# In[3]:


dataframe['obese'] = (dataframe.Index >= 4).astype('int')
dataframe.drop('Index', axis = 1, inplace = True)


# In[4]:


print(dataframe.head())


# In[5]:


from sklearn.model_selection import train_test_split


# In[6]:


X = dataframe.drop(['obese'], axis=1)
Y = dataframe['obese']

data, test, y_train, y_test = train_test_split(X,Y)
data['obese'] = y_train
test['obese'] = y_test
print(data.shape)


# ## Calculate impurity using the Gini index

# $Gini = 1 – \sum^n_{i=1}(P_i)^2$

# In[7]:


def gini_impurity(y):
  '''
  Given a Pandas Series, it calculates the Gini Impurity. 
  y: variable with which calculate Gini Impurity.
  '''
  if isinstance(y, pd.Series):
    p = y.value_counts()/y.shape[0]
    gini = 1-np.sum(p**2)
    return(gini)

  else:
    raise('Object must be a Pandas Series.')


# In[8]:


print(gini_impurity(data.Gender))


# ## Calculate impurity with entropy

# $E(S) = \sum^c_{i=1}-p_ilog_2p_i$

# In[9]:


def entropy(y):
  '''
  Given a Pandas Series, it calculates the entropy. 
  y: variable with which calculate entropy.
  '''
  if isinstance(y, pd.Series):
    a = y.value_counts()/y.shape[0]
    entropy = np.sum(-a*np.log2(a+1e-9))
    return(entropy)

  else:
    raise('Object must be a Pandas Series.')

 


# In[10]:


print(entropy(data.Gender))


# ## How to choose the cuts for our decision tree

# $Information Gain_{Classification}= E(d) – \sum \frac{|s|}{|d|}E(s)$
# 
# 
# $Information Gain_{Regresion}= Variance(d) – \sum \frac{|s|}{|d|}Variance(s)$

# In[11]:


def variance(y):
  '''
  Function to help calculate the variance avoiding nan.
  y: variable to calculate variance to. It should be a Pandas Series.
  '''
  if(len(y) == 1):
    return 0
  else:
    return y.var()

def information_gain(y, mask, func=entropy):
  '''
  It returns the Information Gain of a variable given a loss function.
  y: target variable.
  mask: split choice.
  func: function to be used to calculate Information Gain in case os classification.
  '''
  
  a = sum(mask)
  b = mask.shape[0] - a
  
  if(a == 0 or b ==0): 
    ig = 0
  
  else:
    if y.dtypes != 'O':
      ig = variance(y) - (a/(a+b)* variance(y[mask])) - (b/(a+b)*variance(y[-mask]))
    else:
      ig = func(y)-a/(a+b)*func(y[mask])-b/(a+b)*func(y[-mask])
  
  return ig


# In[12]:


print(information_gain(data['obese'], data['Gender'] == 'Male'))


# ## How to calculate the best split for a variable

# In[13]:


def categorical_options(a):
  '''
  Creates all possible combinations from a Pandas Series.
  a: Pandas Series from where to get all possible combinations. 
  '''
  a = a.unique()

  opciones = []
  for L in range(0, len(a)+1):
      for subset in itertools.combinations(a, L):
          subset = list(subset)
          opciones.append(subset)

  return opciones[1:-1]

def max_information_gain_split(x, y, func=entropy):
  '''
  Given a predictor & target variable, returns the best split, the error and the type of variable based on a selected cost function.
  x: predictor variable as Pandas Series.
  y: target variable as Pandas Series.
  func: function to be used to calculate the best split.
  '''

  split_value = []
  ig = [] 

  numeric_variable = True if x.dtypes != 'O' else False

  # Create options according to variable type
  if numeric_variable:
    options = x.sort_values().unique()[1:]
  else: 
    options = categorical_options(x)

  # Calculate ig for all values
  for val in options:
    mask =   x < val if numeric_variable else x.isin(val)
    val_ig = information_gain(y, mask, func)
    # Append results
    ig.append(val_ig)
    split_value.append(val)

  # Check if there are more than 1 results if not, return False
  if len(ig) == 0:
    return(None,None,None, False)

  else:
  # Get results with highest IG
    best_ig = max(ig)
    best_ig_index = ig.index(best_ig)
    best_split = split_value[best_ig_index]
    return(best_ig,best_split,numeric_variable, True)




# In[14]:


weight_ig, weight_slpit, _, _ = max_information_gain_split(data['Weight'], data['obese'],)  


print(
  "The best split for Weight is when the variable is less than ",
  weight_slpit,"\nInformation Gain for that split is:", weight_ig
)


# ## How to choose the best split

# In[15]:


print(data.drop('obese', axis= 1).apply(max_information_gain_split, y = data['obese']))


# ## How to train a decision tree in Python from scratch

# ### Determining the depth of the tree
# We already have all the ingredients to calculate our decision tree. Now, we must create a function that, given a mask, makes us a split.
# 
# In addition, we will include the different hyperparameters that a decision tree generally offers. Although we could include more, the most relevant are those that prevent the tree from growing too much, thus avoiding overfitting. These hyperparameters are as follows:
# 
# __max_depth__: maximum depth of the tree. If we set it to None, the tree will grow until all the leaves are pure or the hyperparameter min_samples_split has been reached.
# 
# __min_samples_split__: indicates the minimum number of observations a sheet must have to continue creating new nodes.
# 
# __min_information_gain__: the minimum amount the Information Gain must increase for the tree to continue growing.
# With this in mind, let’s finish creating our decision tree from 0 in Python. To do this, we will:
# 
# - Make sure that the conditions established by min_samples_split and max_depth are being fulfilled.
# - Make the split.
# - Ensure that min_information_gain if fulfilled.
# - Save the data of the split and repeat the process.
# 
# 
# To do this, first of all, I will create three functions: one that, given some data, returns the best split with its corresponding information, another that, given some data and a split, makes the split and returns the prediction and finally, a function that given some data, makes a prediction.
# 
# Note: the prediction will only be given in the branches and basically consists of returning the mean of the data in the case of the regression or the mode in the case of the classification.
# 
# 

# In[16]:


def get_best_split(y, data):
  '''
  Given a data, select the best split and return the variable, the value, the variable type and the information gain.
  y: name of the target variable
  data: dataframe where to find the best split.
  '''
  masks = data.drop(y, axis= 1).apply(max_information_gain_split, y = data[y])
  if sum(masks.loc[3,:]) == 0:
    return None, None, None, None

  else:
    # Get only masks that can be splitted
    masks = masks.loc[:,masks.loc[3,:]]

    # Get the results for split with highest IG
    split_variable = masks.iloc[0].astype(np.float32).idxmax()
    #split_valid = masks[split_variable][]
    split_value = masks[split_variable][1] 
    split_ig = masks[split_variable][0]
    split_numeric = masks[split_variable][2]

    return(split_variable, split_value, split_ig, split_numeric)


def make_split(variable, value, data, is_numeric):
  '''
  Given a data and a split conditions, do the split.
  variable: variable with which make the split.
  value: value of the variable to make the split.
  data: data to be splitted.
  is_numeric: boolean considering if the variable to be splitted is numeric or not.
  '''
  if is_numeric:
    data_1 = data[data[variable] < value]
    data_2 = data[(data[variable] < value) == False]

  else:
    data_1 = data[data[variable].isin(value)]
    data_2 = data[(data[variable].isin(value)) == False]

  return(data_1,data_2)

def make_prediction(data, target_factor):
  '''
  Given the target variable, make a prediction.
  data: pandas series for target variable
  target_factor: boolean considering if the variable is a factor or not
  '''

  # Make predictions
  if target_factor:
    pred = data.value_counts().idxmax()
  else:
    pred = data.mean()

  return pred


# ### Training our decision tree in Python
# Now that we have these three functions, we can, let’s train the decision tree that we just programmed in Python.
# 
# - We ensure that both min_samples_split and max_depth are fulfilled.
# 
# - If they are fulfilled, we get the best split and obtain the Information Gain. If any of the conditions are not fulfilled, we make the prediction.
# 
# - We check that the Information Gain Comprobamos passes the minimum amount set by min_information_gain.
# 
# - If the condition above is fulfilled, we make the split and save the decision. If it is not fulfilled, then we make the prediction.
# 
# We will do this process recursively, that is, the function will call itself. The result of the function will be the rules you follow to make the decision:
# 
# 

# In[17]:


def recursive_train_tree(data,y, target_factor, max_depth = None,min_samples_split = None, min_information_gain = 1e-20, counter=0, max_categories = 20):
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

  # Check that max_categories is fulfilled
  if counter==0:
    types = data.dtypes
    check_columns = types[types == "object"].index
    for column in check_columns:
      var_length = len(data[column].value_counts()) 
      if var_length > max_categories:
        raise ValueError('The variable ' + column + ' has '+ str(var_length) + ' unique values, which is more than the accepted ones: ' +  str(max_categories))

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
    if data.shape[0] > min_samples_split:
      sample_cond = True

    else:
      sample_cond = False

  # Check for ig condition
  if depth_cond & sample_cond:

    var,val,ig,var_type = get_best_split(y, data)

    # If ig condition is fulfilled, make split 
    if ig is not None and ig >= min_information_gain:

      counter += 1

      left,right = make_split(var, val, data,var_type)

      # Instantiate sub-tree
      split_type = "<=" if var_type else "in"
      question =   "{} {}  {}".format(var,split_type,val)
      # question = "\n" + counter*" " + "|->" + var + " " + split_type + " " + str(val) 
      subtree = {question: []}


      # Find answers (recursion)
      yes_answer = recursive_train_tree(left,y, target_factor, max_depth,min_samples_split,min_information_gain, counter)

      no_answer = recursive_train_tree(right,y, target_factor, max_depth,min_samples_split,min_information_gain, counter)

      if yes_answer == no_answer:
        subtree = yes_answer

      else:
        subtree[question].append(yes_answer)
        subtree[question].append(no_answer)

    # If it doesn't match IG condition, make prediction
    else:
      pred = make_prediction(data[y],target_factor)
      return pred

   # Drop dataset if doesn't match depth or sample conditions
  else:
    pred = make_prediction(data[y],target_factor)
    return pred

  return subtree



# In[18]:


max_depth = 5
min_samples_split = 20
min_information_gain  = 1e-5


decisiones = recursive_train_tree(data,'obese',True, max_depth,min_samples_split,min_information_gain)


print(decisiones)


# It is done! The decision tree we just coded in Python has created all the rules that it will use to make predictions.
# 
# Now, there would only be one thing left: convert those rules into concrete actions that the algorithm can use to classify new data. Let’s go for it!
# 
# Predict using our decision tree in Python
# To make the prediction, we are going to take an observation and the decision tree. These decisions can be converted into real conditions by splitting them.
# 
# So, to make the prediction we are going to:
# 
# - Break the decision into several chunks.
# 
# - Check the type of decision that it is (numerical or categorical).
# 
# - Considering the type of variable that it is, check the decision boundary. If the decision is fulfilled, return the result, if it is not, then continue with the decision..

# In[19]:


def clasificar_datos(observacion, arbol):
    # print(arbol.keys())
    question = list(arbol.keys())[0] 
    
    if question.split()[1] == '<=':
    
        if observacion[question.split()[0]] <= float(question.split()[2]):
          answer = arbol[question][0]
        else:
          answer = arbol[question][1]
    
    else:
    
        if observacion[question.split()[0]] in (question.split()[2]):
          answer = arbol[question][0]
        else:
          answer = arbol[question][1]
    
    # If the answer is not a dictionary
    if not isinstance(answer, dict):
        return answer
    else:
        residual_tree = answer
    return clasificar_datos(observacion, answer)


# In[20]:


def compute_confusion_matrix(true_labels, predicted_labels, labels):
    num_classes = len(labels)
    confusion_matrix = np.zeros((num_classes, num_classes))

    for true, predicted in zip(true_labels, predicted_labels):
        true_index = labels.index(true)
        predicted_index = labels.index(predicted)
        confusion_matrix[true_index, predicted_index] += 1

    return confusion_matrix

def plot_confusion_matrix(confusion_matrix, labels):
    df = pd.DataFrame(confusion_matrix, index=labels, columns=labels)
    html_table = df.to_html()

    return HTML(html_table)


def precision(y_true, y_pred):
    true_positives = sum([(y_true[i] == 1) and (y_pred[i] == 1) for i in list(range(len(y_pred)))])
    false_positives = sum([(y_true[i] == 0) and (y_pred[i] == 1) for i in list(range(len(y_pred)))])
    precision = true_positives / (true_positives + false_positives)
    return precision

def accuracy(y_true, y_pred):
    correct_predictions = sum([y_true[i] == y_pred[i]  for i in list(range(len(y_pred)))])
    total_predictions = len(y_true)
    accuracy = correct_predictions / total_predictions
    return accuracy

def f1_score(y_true, y_pred):
    precision_val = precision(y_true, y_pred)
    recall_val = recall(y_true, y_pred)
    f1 = 2 * ((precision_val * recall_val) / (precision_val + recall_val))
    return f1

def recall(y_true, y_pred):
    true_positives = sum([(y_true[i] == 1) and (y_pred[i] == 1) for i in list(range(len(y_pred)))])
    false_negatives = sum([(y_true[i] == 1) and (y_pred[i] == 0) for i in list(range(len(y_pred)))])
    recall = true_positives / (true_positives + false_negatives)
    return recall


# In[21]:


#data.drop('obese', axis= 1).apply(clasificar_datos, arbol = decisiones)
#clasificar_datos(data.drop('obese', axis= 1),decisiones)


# In[22]:


result = test.apply(lambda row: clasificar_datos(row.drop('obese'), arbol=decisiones), axis=1)
print(result)


# In[23]:


# Example usage
true_labels = test['obese'].values.tolist()
predicted_labels = result.tolist()
labels = np.unique(result).tolist()

cm = compute_confusion_matrix(true_labels, predicted_labels, labels)
html_table = plot_confusion_matrix(cm, labels)

# Display the HTML table inside the notebook
print(html_table)


# In[24]:


print(f"""
Accuracy: {accuracy(true_labels,predicted_labels)},
precision: {precision(true_labels,predicted_labels)},
Recall: {recall(true_labels,predicted_labels)},
F1-score: {f1_score(true_labels,predicted_labels)}
""")


# In[25]:


def interative_train_tree(data,y, target_factor, max_depth = None,min_samples_split = None, min_information_gain = 1e-20, counter=0, max_categories = 20):
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
    

    stack = []  # Stack to store nodes to be processed
    root = {
        'data': data,
        'depth': 0,
        'node': {}
    }
    stack.append(root)
    while len(stack) != 0:
        
        current = stack.pop()
        xy_current = current['data']
        depth = current['depth']
        current_node = current['node']
        # Check for ig condition
        # print(f"{root} --- {current}")
          # Check that max_categories is fulfilled
        if depth == 0:
            types = xy_current.dtypes
            check_columns = types[types == "object"].index
            for column in check_columns:
              var_length = len(xy_current[column].value_counts())
              if var_length > max_categories:
                raise ValueError('The variable ' + column + ' has '+ str(var_length) + ' unique values, which is more than the accepted ones: ' +  str(max_categories))
        
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
            if data.shape[0] > min_samples_split:
              sample_cond = True
            
            else:
              sample_cond = False
        
        # Check for ig condition
        if depth_cond & sample_cond:
            
            var,val,ig,var_type = get_best_split(y, xy_current)
            
            # If ig condition is fulfilled, make split 
            if ig is not None and ig >= min_information_gain:
                left,right = make_split(var, val, xy_current,var_type)
                
                # Instantiate sub-tree
                split_type = "<=" if var_type else "in"
                question =   "{} {}  {}".format(var,split_type,val)

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
    
            # If it doesn't match IG condition, make prediction
            else:
                pred = make_prediction(xy_current[y],target_factor)
                current_node['col'] = var
                current_node['cutoff'] = ig
                current_node['val'] = val
                current_node['condition'] = pred
                
            
        # Drop dataset if doesn't match depth or sample conditions
        else:
            pred = make_prediction(xy_current[y],target_factor)
            current_node['col'] = var
            current_node['cutoff'] = ig
            current_node['val'] = val
            current_node['condition'] = pred
    
    return root
    


# In[26]:


max_depth = 5
min_samples_split = 20
min_information_gain  = 1e-5


decisiones1 = interative_train_tree(data,'obese',True, max_depth,min_samples_split,min_information_gain)


print(decisiones1)


# In[27]:


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


# In[28]:


result1 = test.apply(lambda row: predict(row.drop('obese'), tree=decisiones1['node']), axis=1)


# In[29]:


dt = test.copy(deep=True)
dt['predicted'] = result1
print(dt)


# In[30]:


# Example usage
true_labels = test['obese'].values.tolist()
predicted_labels = result1.tolist()
labels = np.unique(result1).tolist()

cm = compute_confusion_matrix(true_labels, predicted_labels, labels)
html_table = plot_confusion_matrix(cm, labels)

# Display the HTML table inside the notebook
print(html_table)


# In[31]:


print(f"""
Accuracy: {accuracy(true_labels,predicted_labels)},
precision: {precision(true_labels,predicted_labels)},
Recall: {recall(true_labels,predicted_labels)},
F1-score: {f1_score(true_labels,predicted_labels)}
""")


# In[ ]:




