# This file generates the false usernames that the dataset is trained on
import csv
import random
import pandas as pd
# Name of csv file
filename = 'data/false-names.csv'

# Just the field of usernames
fields = ['username']

# Generate 10000 false names
rows = []
alphabet = "qwertyuiopasdfghjklzxcvbnmQWERTYUIOPASDFGHJKLZXCVBNM1234567890 "
for _ in range(100000):
    # Random length between 5 and 12
    length = random.randint(5, 12)

    username = "".join(random.choices(alphabet, k = length))
    rows.append([username])

# writing to csv file  
with open(filename, 'w') as csvfile:  
    # creating a csv writer object  
    csvwriter = csv.writer(csvfile)  
        
    # writing the fields  
    csvwriter.writerow(fields)  
        
    # writing the data rows  
    csvwriter.writerows(rows) 
