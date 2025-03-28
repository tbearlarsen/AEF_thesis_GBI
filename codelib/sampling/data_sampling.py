import pandas as pd

def yearly_price(file, ppa=52, header=True):
    if header == True:
        yearly_columns = file.iloc[1:, ::ppa]

    else:
        yearly_columns = file.iloc[:, ::ppa]

    return yearly_columns