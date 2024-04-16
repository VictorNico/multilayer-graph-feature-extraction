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
        domain= 'GERMAN',
        dataset_link= './datasets/private/1. UCI Repository/German/german.csv',
        target_variable= 'Class',
        dataset_delimiter=',', 
        all_nominal=True, 
        all_numeric=False, 
        verbose=True, 
        fix_imbalance=False, 
        levels=[2],
        to_remove= [], 
        encoding="utf-8",
        index_col=None,
        alphas=[.1,.2,.3,.4,.5,.6,.7,.8,.85,.9]
        )
    contenu = f'END OF 444 AT {time.strftime("%Y_%m_%d_%H_%M_%S")} \n'
    with open("process.dtvni", "a") as fichier:
        fichier.write(contenu)
