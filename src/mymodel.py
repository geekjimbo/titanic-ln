import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
import yaml 

config = yaml.load(  open('config/config.yml')  )

url = "http://tinyurl.com/titanic-csv" 
data = pd.read_csv(url)

# make all dataset numeric by encoding categorical data
# Target Data Set:
#    Name (numeric)
#	 PClass (numeric)
#	 Age	
#	 SexCode
#	 Survived

columns = ["Name", "PClass", "Age", "SexCode", "Survived"]
df = data[columns]

# encode PClass categorical variable
labelencoder_pclass = LabelEncoder()
_pclass = labelencoder_pclass.fit_transform(df['PClass'])
df['pclass'] = _pclass

# encode Name as categorical variable
labelencoder_name = LabelEncoder()
_name = labelencoder_name.fit_transform(df['Name'])
df['name'] = _name

# produce a candidate df with only numeric values
columns = ["name", "pclass", "Age", "SexCode", "Survived"]
df_candidate = df[columns]

# resolve nulls
df_candidate['Age'] = df_candidate['Age'].fillna(0)
df_candidate['SexCode'] = df_candidate['SexCode'].fillna(0)
df_candidate['pclass'] = df_candidate['pclass'].fillna(0)

#print(df_candidate)


# build the model with lineal regression
# split tranning and testing sets
X = df_candidate[['name', 'pclass', 'Age', 'SexCode']]
y = df_candidate['Survived']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# fit the model
linreg  = LinearRegression()
linreg.fit(X_train, y_train)

# write down df
df_candidate.to_csv(config['target_path'], index=False)

# evaluate model
accuracy = linreg.score(X_test,y_test)
print(accuracy*100,'%')



