import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import pandas as pd
import warnings
warnings.simplefilter("ignore")

### Data Import
data = pd.read_csv("housing_data.csv")

### Remove any diplay restrictions

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
df = pd.DataFrame(data)
    
### Having taken a look at the data we may assume that the price of the house mostly depends on it's size, number of rooms, house age and sights nearby.
### In order to create a linear model we need to get rid of string values by converting them into numerical ones.
### The only variable we need to convert is the 'Sights' variable

### Let's find all the unique elements in the 'Sights' variable

unique_array = np.unique(df["Sights"].str.replace(" ", "")).reshape(1,-1)
arr = []

for x in unique_array:
    for y in x:
        arr.append(y.split(','))

elements = np.unique(np.array(arr).flatten())

### In order to proceed we need to convert the text values into numerical values and rationalize it somehow. For example, let's split the sights and pass 1 for each one if it does exist
sights_sep = df["Sights"].str.get_dummies(', ')
df = df.join(sights_sep)

### After converting the text data to the numerical values let's get down to creating our linear regression model

X = df.drop(['Town', 'Sights', 'Longitude', 'Latitude', 'Price '], axis=1)
Y = df['Price ']

### After getting rid of unnecessary data let's finally create our regression model

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=42)

regression = LinearRegression().fit(X_train, Y_train)
predictions = regression.predict(X_test)
accuracy = r2_score(Y_test, predictions)   ### R2 Score displays 0.938 which is a great score and we can use the model for further predictions

house_age = float(input("Pass the age of the house: "))
house_size = float(input("Pass the size of the house (sq ft): "))
house_rooms = float(input("Pass the number of rooms: "))
house_sights = input("Pass the sights nearby (using commas): ")
sights_arr = np.zeros(11, dtype=int)
house_sights_arr = np.sort(house_sights.replace(" ", "").split(","))


for x in range(len(sights_sep.columns)):
    if sights_sep.columns[x] in house_sights_arr:
        sights_arr[x] = 1
print(sights_arr)
fin = np.array([house_age, house_size, house_rooms], dtype=int)
to_predict = np.concatenate((fin,sights_arr)).reshape(1,-1)
print(to_predict)
prediction = np.array(regression.predict(to_predict), dtype=int)
print(f"The house parameters: {int(house_age)} years old, {house_size} sq. ft. size, {int(house_rooms)} rooms with such sights as: {house_sights}\nEstimated price: {prediction[0]} thousands of dollars")

