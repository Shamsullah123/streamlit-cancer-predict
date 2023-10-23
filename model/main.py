import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pickle as pk

def create_model(data) :
    x = data.drop(['diagnosis'], axis=1)
    y = data['diagnosis']

    # Scale the data

    scalar = StandardScaler()
    x = scalar.fit_transform(x)
    
    # Split the data into traning and testing purpose

    X_train, X_test, Y_train, Y_test = train_test_split(x, y, test_size= 0.2 , random_state=42)
    # train the model
    model = LogisticRegression()
    model.fit(X_train, Y_train)
    # Test the model
    y_pred = model.predict(X_test)
    print("Accuracy of our model is : ", accuracy_score(Y_test, y_pred))
    print("Classification report is : ", classification_report(Y_test, y_pred))
    # Return the model and Scale 
    return model, scalar

def get_clean_data():
    # Read the data from the CSV file for clearning purpose
    data = pd.read_csv("data/data.csv")
    #drop the Nul variable and unneccessary variable
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    # Encode the diagosis variable
    data['diagnosis'] = data['diagnosis'].map({'M' : 1, 'B' : 0})
    
    return data

def main():
    # to get clean cancer data
    data = get_clean_data()
    # create a model  

    model, scalar = create_model(data)
    with open('model/model.pkl', 'wb') as f: 
        pk.dump(model,f)
    
    with open('model/scaler.pkl', 'wb') as f:
        pk.dump(scalar, f)

if __name__ == '__main__':
    main()