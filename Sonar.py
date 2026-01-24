
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split #split into training and test data
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""# Information

"""

#loading datasheet
sonar_data= pd.read_csv('/content/Copy of sonar data.csv', header=None)
sonar_data.head()

#finding rows and columns
sonar_data.shape #208 rows and 61columns

sonar_data.describe()

sonar_data[60].value_counts() #60 is the row number where the diff lies

"""if our observation for both is nearly equal, we will have a good prediction, more the data more accurate your model is

M- mine
r- rock

column 60 are the labels and rest are the features

"""

sonar_data.groupby(60).mean()

x= sonar_data.drop(columns=60, axis=1) #droping a column, axis is 1 if a row, axis is 0
y= sonar_data[60]

"""all the values are stored in x except the 60th column(drop remove it and axis is for the margin column or row)
and the value of 60th row goes to y so that we can abe it

"""

print (x)
print(y) #data succesfully splited

"""# Training and test data

"""

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size= 0.1,stratify=y,random_state=1)
#by test size 0.1 we mean 10% of the data should be test data
#stratify sets the parameter of equal data spliting i.e in rock and mine
#here it would be split through 'y' which is column 60
#random state for spliting it a specific way, so we have 1 and 2 with diff. split types

print(x.shape,x_train.shape, x_test.shape)

"""# Model Training--> Logistic regression

"""

mode1= LogisticRegression()
#training the logistic model with training data

mode1.fit(x_train,y_train)

"""#Model Evaluation

"""

#accuracy on training data

#here the machine has seen the training data not the test data, till now
#accuracy for test data will be lower than that for training,
#accuracy anywhere above 70 is good, more training data the better
x_train_prediction=mode1.predict(x_train)
training_data_accuracy=accuracy_score(x_train_prediction,y_train)

print('for training data:', training_data_accuracy)

"""#For Test data"""

#accuracy on test data
x_test_prediction=mode1.predict(x_test)
test_data_accuracy=accuracy_score(x_test_prediction,y_test)

print('for test data:', test_data_accuracy)

"""so our system has

83- for trainging and

76- for testing and thats really good

# Making Predictive system
"""

input_data= (0.0335,0.0258,0.0398,0.0570,0.0529,0.1091,0.1709,0.1684,0.1865,0.2660,0.3188,0.3553,0.3116,0.1965,0.1780,0.2794,0.2870,0.3969,0.5599,0.6936,0.7969,0.7452,0.8203,0.9261,0.8810,0.8814,0.9301,0.9955,0.8576,0.6069,0.3934,0.2464,0.1645,0.1140,0.0956,0.0080,0.0702,0.0936,0.0894,0.1127,0.0873,0.1020,0.1964,0.2256,0.1814,0.2012,0.1688,0.1037,0.0501,0.0136,0.0130,0.0120,0.0039,0.0053,0.0062,0.0046,0.0045,0.0022,0.0005,0.0031)
#changing the data to a numpy array
as_narray= np.asarray(input_data)

#reshaping nparray as we are predicting for one instance
ipd_reshaped= as_narray.reshape(1,-1) #1 as there is only one instance
prediction= mode1.predict(ipd_reshaped) #mode1 carries our logistic regression model and we call it

print(input('given thing:'))
if prediction[0] == 'R':
  print('Rock')
else:
  print('Mine')
