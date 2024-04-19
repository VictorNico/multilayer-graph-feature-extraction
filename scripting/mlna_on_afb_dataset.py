"""
    Author: VICTOR DJIEMBOU
    addedAt: 02/12/2023
    changes:
        - 02/12/2023:
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
        domain= 'AFB',
        dataset_link= './datasets/New_datas_first.xlsx',
        target_variable= 'ENIMPAYEOUPAS',
        dataset_delimiter=',', 
        all_nominal=True, 
        all_numeric=False, 
        verbose=True, 
        fix_imbalance=False, 
        levels=[2],
        to_remove= ['Type'], 
        encoding="utf-8",
        index_col=None,
        alphas=[.1,.2,.3,.4,.5,.6,.7,.8,.85,.9], 
        portion=.1
        )
    contenu = f'END OF 222 AT {time.strftime("%Y_%m_%d_%H_%M_%S")} \n'
    with open("process.dtvni", "a") as fichier:
        fichier.write(contenu)
