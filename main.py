from mypackage.module2 import input_taking
from mypackage.module1 import main

import pickle
import os

classifier_path = 'naive_bayes_classifier.pkl'

if os.path.exists(classifier_path):
    with open(classifier_path, 'rb') as f:
        classifier = pickle.load(f)
    print("\nClassifier loaded successfully from file.")
    input_text=input("Enter your text for sentiment analysis: ")
    sentiment_output = input_taking(input_text)
    
else:
    if __name__ == "__main__":
        main()

    