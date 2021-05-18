# -*- coding: utf-8 -*-
"""
Created on Wed Apr 14 11:12:33 2021

@author: niecla
"""
import sys
import os.path
import pandas as pd
import json
import sqlite3

from sqlalchemy import create_engine

data_path = os.path.join(os.getcwd(), "..", "model")
pict_path = os.path.join(os.getcwd(), "..", "pictures")

# os.unlink(os.path.join(data_path, "test.db"))

# engine = create_engine(os.path.join(data_path, "test2.db"), echo=False)

# Open JSON data
with open(os.path.join(data_path, "benchmark_4x.json")) as f:
    data = json.load(f)


# Create A DataFrame From the JSON Data
df = pd.DataFrame(data)

# # create a connection to a sql database
# conn = sqlite3.connect(os.path.join(data_path, "test3.db"))
# c = conn.cursor()


# # c.execute('CREATE TABLE MODEL ')
# # conn.commit()


# df.to_sql('model', conn, if_exists='replace', index=False)
# # # conn.execute

# conn.close()


# import sqlite3
# from pandas import DataFrame

conn = sqlite3.connect(os.path.join(data_path, 'TestDB1.db'))
c = conn.cursor()

c.execute('CREATE TABLE CARS (Brand text, Price number)')
conn.commit()

Cars = {'Brand': ['Honda Civic','Toyota Corolla','Ford Focus','Audi A4'],
        'Price': [22000,25000,27000,35000]
        }

df = pd.DataFrame(Cars, columns= ['Brand', 'Price'])
df.to_sql('CARS', conn, if_exists='replace', index = False)
 
c.execute('''  
SELECT * FROM CARS
          ''')

for row in c.fetchall():
    print (row)