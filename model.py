import numpy as np 
import pandas as pd 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score ,classification_report
from sklearn.preprocessing import LabelEncoder 
from sklearn.tree import DecisionTreeClassifier
import pickle

def load_dataset():
    """"Loading the Dataset"""
    df=pd.read_csv(r"C:\Users\alisa\OneDrive\Desktop\Data_Versioning_Using_DVC\dataset\IRIS.csv") 
    le = LabelEncoder()
    df['species']=le.fit_transform(df['species'])
    X=df.drop(columns='species')
    y=df['species']
    return X,y 
def preprocessing(X,y):
    X_train , X_test ,y_train,y_test = train_test_split(X,y,test_size=0.4,random_state=42) 
    return X_train , X_test ,y_train,y_test 
def traning(X_train,y_train):
    model=DecisionTreeClassifier()
    model.fit(X_train,y_train) 
    return model 
def evaluate(model,X_test,y_test):
    y_pred = model.predict(X_test)
    score = accuracy_score(y_test,y_pred)
    report = classification_report(y_test,y_pred)
    print("Accuracy=",score) 
    print("cls report=",report)
    return score , report 
def model_save(model,filename='model.pkl'):
    with open(filename,'wb') as file:
        pickle.dump(model,file)
    print(f"model save as {filename}") 
def main():
    X,y = load_dataset()
    X_train , X_test ,y_train,y_test=preprocessing(X,y)
    model = traning(X_train,y_train)
    evaluate(model,X_test,y_test)
    model_save(model) 
if __name__ == "__main__":
    main()