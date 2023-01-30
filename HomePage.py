import streamlit as st
import joblib, os
import re
from nltk.stem.porter import PorterStemmer
import nltk
from nltk.corpus import stopwords
import pandas as pd

ps = PorterStemmer()
nltk.data.path.append('./nltk_data')



            


# load Vectorizer For Gender Prediction
news_vectorizer = open("Models/tfidfvect.pkl","rb")
news_cv = joblib.load(news_vectorizer)


    # Helper methods
def preprocess(txt):
    review = re.sub('[^a-zA-Z]', ' ', txt)
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    txt  = review
    return review

            #load our models
def load_prediction_models(model_file):
	loaded_model = joblib.load(open(os.path.join(model_file),"rb"))
	return loaded_model


# Get the Keys
def get_keys(val,my_dict):
	for key,value in my_dict.items():
		if val == value:
			return key


    


def main():

    # st.title('FakeApple')

    html_temp = """
	<div style="background-color:blue;padding:10px">
	<h1 style="color:white;text-align:center;">News Detection ML App </h1>
	</div>
	"""
    st.markdown(html_temp,unsafe_allow_html=True)


    
    activites = ['Home', 'Predection', 'nlp Process','Results']
    choice = st.sidebar.selectbox("Menu",activites)
    

    if choice == 'Predection':
        st.info("Prediction with ML")

        news_text = st.text_area("Enter News Here","Type Here")
        txt = preprocess(news_text)
        vect_text = news_cv.transform([txt]).toarray()


        all_ml_models = ["Logistic Regresion","SVM","RandomForest"]
        model_choice = st.selectbox("Choose ML Model",all_ml_models)

        labels = {" Fake News": 0," Real News ": 1}

        if st.button("Predict"):
            st.text("preprocessed Text : \n{}".format(txt))
            if model_choice == 'Logistic Regresion':
                
                modelname = load_prediction_models("Models\lr_clf.pkl")
                Predection = modelname.predict(vect_text)
                proba = modelname.predict_proba(vect_text)
                # st.write(proba)
            elif model_choice == 'SVM':
                modelname = load_prediction_models("Models/svm_clf.pkl")
                Predection = modelname.predict(vect_text)
                proba = modelname.predict_proba(vect_text)
                # st.write(Predection)
                
            elif model_choice == 'RandomForest':
                modelname = load_prediction_models("Models/rf_clf.pkl")
                Predection = modelname.predict(vect_text)
                proba = modelname.predict_proba(vect_text)
                # st.write(Predection)
                

            #final_r = get_keys(Predection[0] ,labels)
            #st.success("I Think it's  -   {}".format(final_r))
            res_col1 ,res_col2 = st.columns(2)
            with res_col1:
               
                st.success("Prediction")
                final_r = get_keys(Predection[0] ,labels)
                st.write(final_r)

            with res_col2:
                st.info("Probability")
                st.write("Fake by  : {:.0%}\t".format(proba[0][0]))
                st.write("Real bY  : {:.0%}".format(proba[0][1]))


                #"Fake by  : {}\n".format(proba[0][0])
                #"Real bY  : {}".format(proba[0][1])

    if choice == 'NLP Process':
        st.info("Natural Language Processing of Text")




if __name__ == '__main__':
    main()
    