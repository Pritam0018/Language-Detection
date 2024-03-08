import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import streamlit as st 

st.title("LANGUAGE DETECTION")
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
data = pd.read_csv("D:\language_detection\Language Detection.csv")
x = np.array(data["Text"])
y = np.array(data["Language"])
cv = CountVectorizer()
X = cv.fit_transform(x)
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20,random_state=42)
model = MultinomialNB()
model.fit(X_train,y_train)
data_=st.text_input("Enter here")
data = cv.transform([data_]).toarray()
output = model.predict(data)
if st.button("predict"):
    st.text(" ".join(output))