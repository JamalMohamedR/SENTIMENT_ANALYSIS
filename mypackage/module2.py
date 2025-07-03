from mypackage.module1 import preprocess_text, extract_features
import pickle

def input_taking(inputed_text):
    # Preprocess the input text
    pre_processing_text=preprocess_text(inputed_text)

    extracted_feautres_input=extract_features(pre_processing_text)

    with open('naive_bayes_classifier.pkl', 'rb') as f:
        classifier = pickle.load(f)

    # Predict sentiment
    predicted_sentiment = classifier.classify(extracted_feautres_input)


    print(f"\nSentiment for the input text: {inputed_text}\nSentiment:) {predicted_sentiment}")
