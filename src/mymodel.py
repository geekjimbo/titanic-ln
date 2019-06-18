import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB, BernoulliNB
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

gausian_nb = GaussianNB()
gausian_nb.fit(X_train, y_train)


# encode Age
age_df = pd.get_dummies(df_candidate['Age'])
arr = list(age_df.columns)
age_cols_names = []
for i in arr:
	age_cols_names.append('age_'+str(i))

age_df.columns = age_cols_names

# encode Pclass
pclass_df = pd.get_dummies(df_candidate['pclass'])
arr = list(pclass_df.columns)
pclass_cols_names = []
for i in arr:
	pclass_cols_names.append('pclass_'+str(i))

pclass_df.columns = pclass_cols_names



# produce one single binary df
df_binary = df_candidate[['SexCode', 'Survived']] 
df_final=pd.concat([df_binary, pclass_df, age_df], axis=1)

# write down df
df_final.to_csv(config['target_path'], index=False)


# split tranning and testing sets
y2 = df_final['Survived']
X2 = df_final.drop(['Survived'], axis=1)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.33, random_state=42)


binary_nb = BernoulliNB()
binary_nb.fit(X_train2, y_train2)

# evaluate model
accuracy = linreg.score(X_test,y_test)
print("lineal regression accuracy:")
print(accuracy*100,'%')

print("GaussianNB accuracy:")
print(gausian_nb.score(X_test, y_test))

print("BernoulliNB accuracy:")
print(binary_nb.score(X_test2, y_test2))




