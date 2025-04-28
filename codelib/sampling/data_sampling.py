import pandas as pd

def yearly_price(file, ppa=52, header=True):
    """
    If when loading an excel file to a dataframe, the heading is recognized (i.e. you don't include header=None), then header should be False here
    """
    if header == True:
        yearly_columns = file.iloc[1:, ::ppa]

    else:
        yearly_columns = file.iloc[:, ::ppa]

    return yearly_columns