"""
    Author: VICTOR DJIEMBOU
    addedAt: 29/11/2023
    changes:
        - 29/11/2023:
            - add pipeline call with 0 hyperparameters
        - 01/12/2023:
            - update parameters call by adding cwd, domain, dataset_link, target_variable, dataset_delimiter
        - 02/12/2023:
            - update parameters call by adding all_nominal, all_numeric, verbose, fix_imbalance, levels, to_remove, encoding, index_col

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
        domain= 'CREDIT_RISK_DATASET',
        dataset_link= './datasets/credit_risk_dataset.csv',
        target_variable= 'loan_status',
        dataset_delimiter=',', 
        all_nominal=True, 
        all_numeric=False, 
        verbose=True, 
        fix_imbalance=False, 
        levels=[2,3,4,5,6],
        to_remove= ['loan_grade'], 
        encoding="utf-8",
        index_col=None
        )
