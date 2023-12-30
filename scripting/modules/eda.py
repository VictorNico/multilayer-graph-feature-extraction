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
				data[col] = data[col].fillna(data[col].quantile(0.5))
			else:
				data.drop([col],axis=1, inplace=True)
	return data