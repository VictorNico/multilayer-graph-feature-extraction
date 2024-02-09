"""
    Author: VICTOR DJIEMBOU
    addedAt: 07/01/2024
    changes:
        - 07/01/2024:
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
        domain= 'THOMAS',
        dataset_link= './datasets/private/5. thomas/Loan Data.csv',
        target_variable= 'BAD',
        dataset_delimiter=';', 
        all_nominal=True, 
        all_numeric=False, 
        verbose=True, 
        fix_imbalance=False, 
        levels=[2,3],
        to_remove= [], 
        encoding="utf-8",
        index_col=None
        )
