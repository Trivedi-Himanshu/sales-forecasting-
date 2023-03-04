import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler as Scale
from feature_engine.outliers import Winsorizer

import statsmodels.formula.api as smf
import statsmodels.graphics.tsaplots as tsa_plots
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
import pmdarima as pm

#pip install feature_engine

df = pd.read_excel(r'E:\Himanshu\DS project (Cement Sales and Demand)\360\All India_Features_07012023 (1).xlsx')
df

df_1 = df.iloc[2:,:]

df_1.columns = (['year','month','GDP_Construction_Rs_Crs','GDP_Real Estate_Rs_Crs','Oveall_GDP_Growth%','water_source','limestone','Coal(in metric tones)','Home_Interest_Rate','Trasportation_Cost','Population_in_crores','order_quantity_milliontonne','sale_quantity_milliontonne','unit_price','Total_Price'])
df_1.reset_index(inplace= True)
df_1.head()

print(df_1.info())
print(df_1.describe())

# Typecasting "object" to "integer"
df_1 = df_1.astype('float64')
df_1.info()


#dropping index column
df_1.drop('index', axis=1, inplace= True)
df_1.head()

#missing values 
df_1.isna().sum()

#duplicate entries 
df_1.duplicated().sum()

#normalizing the data 
df_norm = pd.DataFrame(Scale().fit_transform(df_1.iloc[:,[2,3,4,5,6,7,
8,9,10,13,14]]))

#giving the columns names 
df_norm.columns= df_1.iloc[:,[2,3,4,5,6,7,
8,9,10,13,14]].columns

df_y = df_1.iloc[:,[11,12]]
df_x = df_norm

df_2 = pd.concat([df_x, df_y], axis=1)
df_2

#adding the month feature to the dataset
df_2['month'] = df_1.month
df_2

#plt.plot(df_2['order_quantity_milliontonne'], df_2['sale_quantity_milliontonne'])

df_2.columns

# visualizing the data
plt.figure(1)
sns.boxplot(data= df_1[['month', 'GDP_Construction_Rs_Crs', 'GDP_Real Estate_Rs_Crs',
       'Oveall_GDP_Growth%']])

plt.figure(2)
sns.boxplot(data = df_1[['water_source', 'limestone',
       'Coal(in metric tones)', 'Home_Interest_Rate', 'Trasportation_Cost']])

plt.figure(3)
sns.boxplot(data = df_1[['Population_in_crores', 'unit_price', 'Total_Price',
       'order_quantity_milliontonne', 'sale_quantity_milliontonne']])

#outliers present in the data 
#handling the outliers by IQR method

winsor = Winsorizer(capping_method = 'iqr', fold= 1.5, tail= 'both', 
variables= ['Home_Interest_Rate','Oveall_GDP_Growth%',
'GDP_Construction_Rs_Crs'])

df_2 = winsor.fit_transform(df_1)

plt.figure(4)
sns.boxplot(data= df_2[['month', 'GDP_Construction_Rs_Crs', 'GDP_Real Estate_Rs_Crs',
       'Oveall_GDP_Growth%']])

plt.figure(5)
sns.boxplot(data = df_2[['water_source', 'limestone',
       'Coal(in metric tones)', 'Home_Interest_Rate', 'Trasportation_Cost']])

plt.figure(6)
sns.boxplot(data = df_2[['Population_in_crores', 'unit_price', 'Total_Price',
       'order_quantity_milliontonne', 'sale_quantity_milliontonne']])

df_2

df_3 = df_2.iloc[:, [0,1,11,12]]
df_3.info()

sns.pairplot(df_3)

train = df_3.iloc[:-12,:]
test = df_3.iloc[-12:,]

ar_model = pm.auto_arima(train.sale_quantity_milliontonne, start_p=0, start_q=0,
                      max_p=12, max_q=12, 
                      m= 1,
                      d=None,           
                      seasonal= True,   
                      start_P=0, trace=True,
                      error_action='warn', stepwise=True)


arima_model = ARIMA(train.sale_quantity_milliontonne, order=(10,1,12)).fit()
# result = arima_model.fit()
#print(result.summary())

start_index = len(train)
end_index = start_index + 11
forcast = arima_model.predict(start= start_index, end= end_index)
forcast

MAPE_best = np.mean(np.absolute(((test.sale_quantity_milliontonne - forcast)/test.sale_quantity_milliontonne)*100))
print(f'Accuracy of the model is = {100 - MAPE_best}')

import pickle 
pickle.dump(arima_model, open('arima_model_2.pkl','wb'))

