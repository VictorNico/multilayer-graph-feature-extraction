import pandas as pd
import numpy as np


def is_ordinal(column):
    unique_values = column.unique()
    sorted_values = sorted(unique_values)
    return list(unique_values) == sorted_values or list(unique_values) == sorted_values[::-1]

def detect_ordinal_columns(df):
    ordinal_columns = []
    for column in df.columns:
        if is_ordinal(df[column]):
            ordinal_columns.append(column)
    return ordinal_columns

def numeric_vector_is_nominal(series):
    """Use Value count to know if a column identify as numeric one has a nominal logic
    Args:
        Series: is the Series vector of column

    Returns
        The Flag boolean value of decusion
    """
    data = None
    if isinstance(series, pd.Series):
        data = series
    elif isinstance(series, list) or isinstance(series, np.ndarray):
        data = pd.Series(series)
    else:
        return None
    # Compter le nombre d'occurrences de chaque valeur
    occurrences = data.value_counts()

    # Vérifier si toutes les valeurs sont uniques
    suit_serie_nominale = (occurrences == 1).all()

    return suit_serie_nominale

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    # Exemple d'utilisation
    df = pd.DataFrame({'colonne1': ['faible', 'moyen', 'élevé','blala'],
                       'colonne2': ['A', 'B', 'C','D'],
                       'colonne3': [70, 25, 45, 60]})

    ordinal_columns = detect_ordinal_columns(df)
    cols = {col: True for col in df.columns if numeric_vector_is_nominal(col)}
    print(ordinal_columns)
    print(cols)