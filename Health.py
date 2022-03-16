import pandas as pd
import numpy as np 
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
data = pd.read_csv("D:/machine learning/Health Insurance/Health_Insurance.csv")
 
data["sex"] = data["sex"].map({"female": 0,"male": 1 })
data["smoker"] = data["smoker"].map({"no":0,"yes":1})


x = data.drop(columns=["region","charges"]).values
y = data["charges"].values
  
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2 , random_state = 42)

model = RandomForestRegressor()
model.fit(x_train,y_train)

pickle.dump(model,open("Health.pkl","wb"))
reg = pickle.load(open("Health.pkl","rb"))   
print(reg.predict([[19,1,23.44,0,1]]))