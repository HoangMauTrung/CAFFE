# cover xlsx to csv
# slpit name of image from url
# download img from url 

import numpy as np 
import pandas as pd 



def splitName(url):
    url = str(url)
    x = url.split('/')
    return x[-1]


df = pd.read_excel(r'file.xlsx')

name = [splitName(i) for i in df['image']]

df['name'] = name

df.to_csv('dataImage.csv')
# print (df.iloc[:10])