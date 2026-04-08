import  pandas as pd  
# df = pd.read_csv('car_cleaned_data.csv')
df = pd.read_csv(r'C:\Users\Haha CORPORATION\Desktop\project deployment 2026\car price prediction\car_cleaned_data.csv')
df.head()
df.columns
X=df[['Present_Price', 'Kms_Driven',
       'Car_Age', 'Fuel_Type_CNG', 'Fuel_Type_Diesel',
       'Fuel_Type_Petrol', 'Transmission_Automatic', 'Transmission_Manual']]
y=df['Selling_Price']
# Fitting Simple Linear Regression to the Trainin
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(X, y) 

import pickle
with open('rf_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)