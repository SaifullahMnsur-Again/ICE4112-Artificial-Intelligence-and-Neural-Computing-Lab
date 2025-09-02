from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
import pandas as pd
import os

def main():
    
    # this lets you get the path of current directory and avoids relative path related issues
    current_dir = os.getcwd()

    # construct the absolute dataset path from relative path
    dataset_path = f'{current_dir}/dataset/spam_email_dataset_01.csv'

    # get the dataset into a pandas dataframe with error handling
    try:
        df = pd.read_csv(dataset_path)
    except Exception as e:
        print(f"Failed to load dataset '{dataset_path}': {e}")
        return
    
    # print dataframe
    print("="*25 + " Dataset " + "="*25)
    print(df)

    # check the dataset, collect the column name and replace with Message and Category
    message_column_name = 'Message'
    category_column_name = 'Category'

    # collet those data
    X = df[message_column_name]
    y = df[category_column_name]

    print("="*5 + f" {message_column_name} and {category_column_name} collected from the dataset " + "="*5)

    # use vectorizer and fit transform the message data which counts the frequency of words for each message
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(X)

    # split the data into 80-20 where random 20% will be reserved for testing and rest will be used in training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # make a Multinomial naive bayes model and train(fit) the training data set with it
    model = MultinomialNB()
    model.fit(X_train, y_train)

    # use test dataset to predict the model
    y_pred = model.predict(X_test)

    # calculate the prediction result with respect to testing dataset
    accuracy = accuracy_score(y_test, y_pred)

    print(f"\nAccuracy: {round(accuracy*100, 2)}%")
    
    print("="*7 + " Completed " + "="*7)

if __name__ == '__main__':
    main()