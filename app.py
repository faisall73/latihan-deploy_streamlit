import joblib
import streamlit as st 

models = joblib.load(r"models\model_logistic_regression.pkl")
tfidf = joblib.load(r"models\tfidf_vectorizer.pkl")

st.title ("aplikasi klasifikasi komentar publik")
st.write("aplokasi ini di buat menggunakan teknologi NLP degan memanfaatkan model mechine learning logistic Regression")
input = st.text_input("masukan komentar anda")
if st.button("sumbit"):
     if input.strip() == "":
        st.warning("komentar tidak boleh kosong")
     else:
        vector = tfidf.transform([input])
        prediksi = models.predict(vector) [0]
        label_map = {
            0: "negatif",
            1: "positif"
        }
        st.subheader("hasil analisis komentar")
        st.write("**komentar**",label_map.get(prediksi, prediksi))