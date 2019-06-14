import requests 
import pandas as pd 
import numpy as np
import os

df = pd.read_csv(r'dataImage.csv')

for i in range(0, len(df['image'])):

    try:
        req = requests.get(str(df['image'][i]), stream=True)

        req.raise_for_status()
        with open(os.path.join('compare', str(df['name'][i])), 'wb') as fd:
            for chunk in req.iter_content(chunk_size=50000):
                print('Received a Chunk ', i)
                fd.write(chunk)
    except requests.HTTPError as e:
        print('Not Found for url : ',df['image'][i])
        status_code = e.response.status_code
    