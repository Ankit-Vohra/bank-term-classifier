#Importing The Libraries
import pandas as pd
from sklearn.metrics import confusion_matrix
from  sklearn.metrics import accuracy_score

#Reading The Dataset
dataset = pd.read_csv("bank-additional-full.csv",sep=";")

#Spliiting X and Y Variables from Dataset
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:,-1].values

l=[1,2,3,4,5,6,7,8,9,14] #Column numbers having categorical data

#Label Encoding of Categorical Data 
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
for i in l:
  X[:,int(i)] = labelencoder.fit_transform(X[:,int(i)])


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#Using Logistic Regression Model
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression(penalty = 'l1' ,C=0.5)
log_reg.fit(X_train,y_train)
#Predicting the test set
y_pred_lr = log_reg.predict(X_test)
#Printing Confusion Matrix
cm_lr = confusion_matrix(y_test, y_pred_lr)
#Calculating The Accuracy of the model
acc_lr=accuracy_score(y_pred_lr,y_test)


#Using Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
ran_for = RandomForestClassifier(n_estimators = 400, criterion = 'gini', random_state = 0)
'''
As we go on increasing the estimators 
we can increase the accuracy 
but run time will increase considerably and on a proportionate scale accuracy won't increase.
'''
ran_for.fit(X_train, y_train)
#Predicting the test set
y_pred_rf = ran_for.predict(X_test)
#Printing Confusion Matrix
cm_rf = confusion_matrix(y_test, y_pred_lr)
#Calculating The Accuracy of the model
acc_rf=accuracy_score(y_pred_rf,y_test)


#Using K Nearest Neighbours Classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(22) 
#Using different values of number of nearest neighbours by running in a for loop,
#best value we get is 22
knn.fit(X_train,y_train)
y_pred_knn = knn.predict(X_test)
#Calculating The Accuracy of the model
acc_knn=accuracy_score(y_pred_knn,y_test)
#Printing Confusion Matrix
cm_knn = confusion_matrix(y_test, y_pred_knn)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#Using SVM Classifier
from sklearn.svm import SVC
svm = SVC(kernel = 'rbf', random_state = 0)
svm.fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
#Calculating The Accuracy of the model
acc_svm=accuracy_score(y_pred_svm,y_test)
#Printing Confusion Matrix
cm_svm = confusion_matrix(y_test, y_pred_svm)


from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)
#Calculating The Accuracy of the model
y_pred_dt = classifier.predict(X_test)
acc_dt=accuracy_score(y_pred_dt,y_test)
#Printing Confusion Matrix
cm_dt = confusion_matrix(y_test, y_pred_dt)


accuracy={"Logistic Regression":acc_lr,"Random Forest":acc_rf,"K-Nearest Neighbors":acc_knn,"Support Vector Machine":acc_svm,"Decision Tree":acc_dt}
print("Accuracy of different models is given as:")
for i in accuracy.keys():
  print(i," : ",round(accuracy[i]*100,2),"%")

#Random Forest Gives the highest Accuracy as compared to other classification models