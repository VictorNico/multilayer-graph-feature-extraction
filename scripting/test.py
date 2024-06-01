if __name__ == '__main__':
    import pandas as pd

    # Dictionnaire de données
    df = pd.DataFrame({"A": ["foo", "foo", "foo", "foo", "foo",
                             "bar", "bar", "bar", "bar"],
                       "B": ["one", "one", "one", "two", "two",
                             "one", "one", "two", "two"],
                       "C": ["small", "large", "large", "small",
                             "small", "large", "small", "small",
                             "large"],
                       "D": [1, 2, 2, 3, 3, 4, 5, 6, 7],
                       "E": [2, 4, 5, 5, 6, 6, 8, 9, 9]})

    # Pivotiser le DataFrame
    pivot_table = pd.pivot_table(df, values=['D', 'E'], index=['A', 'C'],
                           aggfunc={'D': "mean",
                                    'E': ["min", "max", "mean"]})
    print(pivot_table)
    # Mise en forme du tableau
    table = pivot_table.to_latex(index=True,
                                 column_format='|l|l|l|r|r|r|r|r|r|r|r|r|r|r|r|r|',
                                 caption='Résultats des modèles',
                                 label='tab:results',
                                 escape=False)

    print(table)