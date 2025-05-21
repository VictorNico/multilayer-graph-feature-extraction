"""
  Author: VICTOR DJIEMBOU
  addedAt: 02/12/2023
  changes:
    - 02/12/2023:
      - add get_na_columns methods
"""
#################################################
##          Libraries importation
#################################################

###### Begin

import pandas as pd
import numpy as np
# from sklearn.impute import KNNImputer

###### End


#################################################
##          Methods definition
#################################################








def get_na_columns(dataframe):
	"""Compute nan checker and pull out columns with nan values
	Args:
		dataframe: dataset to analyse

	Returns:
		A list of name of columns with NAs values and the proportion of NAs values
	"""

	NAs = [(dataframe.columns.tolist()[i],val/dataframe.shape[0]) for i, val in enumerate(dataframe.isna().sum().values.tolist()) if val > 0] if isinstance(dataframe, pd.DataFrame) else []
	return NAs

def impute_nan_values(dataframe, variables):
	"""Replace NAs values within KNN approche
	Args:
		dataframe: the dataset with NAs dimension

	Returns:
		new dataset with NAs replace by neighbor value
	"""

	# imputer = KNNImputer(n_neighbors=5)
	# df_imputed = pd.DataFrame(
	# 	imputer.fit_transform(dataframe), 
	# 	columns=dataframe.columns
	# 	) if isinstance(dataframe, pd.DataFrame) else dataframe # if it's a dataframe apply transformer else send back the same data source
	if isinstance(dataframe, pd.DataFrame) and isinstance(variables, list):
		data= dataframe.copy(deep=True)
		for (col, tho) in variables:
			if tho <0.5:
				# data[col] = data[col].fillna(data[col].quantile(0.5))
				# Assuming 'col' is the column name you want to fill missing values for

				# Check if the column data type is numeric
				if pd.api.types.is_numeric_dtype(data[col]):
				    # Replace missing values with the median value
				    data[col] = data[col].fillna(data[col].median())
				else:
				    # Handle non-numeric column types differently, based on your specific requirements
				    # For example, you can fill missing values with a specific value or perform a different transformation
				    # data[col] = data[col].fillna('Unknown')  # Example: fill missing values with a string 'Unknown'
				    data[col] = data[col].fillna(data[col].mode()[0])  # Example: fill missing values with the mode value

			else:
				data.drop([col],axis=1, inplace=True)
	return data