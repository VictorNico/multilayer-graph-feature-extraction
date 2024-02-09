"""
    Author: VICTOR DJIEMBOU
    addedAt: 30/12/2023
    changes:
        - 30/12/2023:
            - add pipeline called
"""





# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #################################################
    ##          Libraries importation
    #################################################

    ###### Begin

    from modules.pipeline import *

    ###### End


    #################################################
    ##          définition du GUI
    #################################################
    #print_hi('PyCharm')
    mlnaPipeline(
        cwd= os.getcwd(),
        domain= 'JAPAN',
        dataset_link= './datasets/private/1. UCI Repository/Japan/crx.csv',
        target_variable= 'A16',
        dataset_delimiter=',', 
        all_nominal=True, 
        all_numeric=False, 
        verbose=True, 
        fix_imbalance=False, 
        levels=[3],
        to_remove= [], 
        encoding="utf-8",
        index_col=None,
        na_values= ['?',]
        )
