import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer 
from nltk.classify import NaiveBayesClassifier
from sklearn.model_selection import train_test_split
import re
import pickle 
from tqdm import tqdm  # for progress bars
import os 

lemmatizer = WordNetLemmatizer()


def clean_text(text):
        text = text.lower()
        #remove HTML tags
        text = re.sub('<.*?>', '', text) 
        #remove special characters
        text = re.sub('[^a-zA-Z]', ' ', text) 
        #remove extra whitespace
        text = re.sub(r'\s+', ' ', text) 
        return text.strip()   

stop_words=set(stopwords.words('english'))
print("\nFirst few stopwords:")
print(list(stop_words)[:10])

def preprocess_text(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    #display the first few stopwords
    # Remove stopwords and non-alphabetic tokens
    cleaned_tokens= [ t for t in tokens if t.isalpha() and t.lower() not in stop_words ]
    # Lemmatize the tokens
    lemmized_words=[lemmatizer.lemmatize(t) for t in cleaned_tokens]
    # return the pre_processed tokens
    return lemmized_words
    


def extract_features(words):
    return {word: True for word in words  }





def main():
    # Download necessary NLTK resources
    # comment the following lines after the first run to avoid re-downloading
    # nltk.download('punkt')
    # nltk.download('gutenberg')
    # nltk.download('averaged_perceptron_tagger')
    # nltk.download('stopwords')
    # nltk.download('all')
    # nltk.download('wordnet')

    data= pd.read_csv('data.csv')
    data=data.dropna()

    data['review'] = data['review'].apply(clean_text)

    #display the first few rows of the dataframe
    print("First few rows of the dataset:")
    print(data.head())

    # Apply preprocessing to the actual data frame from pandas 

    documents=[(preprocess_text(row['review']),row['sentiment']) for idx, row in tqdm(data.iterrows(), total=len(data), desc="Preprocessing reviews")]
    print("\nFirst few preprocessed documents:")
    print(documents[:5])

    # Extract features from the preprocessed documents
    extracted_features=[(extract_features(word),label) for (word,label) in tqdm(documents, desc="Extracting features")]

    print("\nFirst few extracted features:")
    print(extracted_features[:5])


    train_data, test_data = train_test_split(extracted_features, test_size=0.2, random_state=42)
    classifier_path = 'naive_bayes_classifier.pkl'
    if os.path.exists(classifier_path):
        with open(classifier_path, 'rb') as f:
            classifier = pickle.load(f)
        print("\nClassifier loaded successfully from file.")
    else:
        classifier = NaiveBayesClassifier.train(train_data)
        with open(classifier_path, 'wb') as f:
            pickle.dump(classifier, f)
        print("\nClassifier trained and saved successfully.")


    # Evaluate the classifier on the test set
    accuracy = nltk.classify.accuracy(classifier,test_data)

    print("accuracy:", accuracy)


    # Show the most informative features

    classifier.show_most_informative_features(10)

    with open(classifier_path, 'wb') as f:
        pickle.dump(classifier, f)
    print("\nClassifier trained and saved successfully.")


























