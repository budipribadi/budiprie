# import required libraries
import pandas as pd
import numpy as np
import pickle
import time
import streamlit as st
import altair as alt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import  RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, completeness_score, precision_score, recall_score, classification_report, confusion_matrix, f1_score, roc_curve, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_auc_score



# page Configuration
st.set_page_config(layout="wide", page_title="CAPSTONE project DQLAB", page_icon=":heart:")
st.sidebar.title("Navigation")
nav = st.sidebar.selectbox("Go to", ("Home", "Dataset", "Exploratory Data Analysis", "Dimensionality Reduction",  "Machine Learning Modelling", "Prediction", "About me"))

# Data Set Page
url =  "https://storage.googleapis.com/dqlab-dataset/heart_disease.csv"
df = pd.read_csv(url)

def heart():
    st.sidebar.header('User Input Features:')
    uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
    if uploaded_file is not None:
        input_df = pd.read_csv(uploaded_file)
    else:
        def user_input_manual():
            st.sidebar.write('Manual Input')
            cp = st.sidebar.slider('Chest pain type', 1,4,2)
            thalach = st.sidebar.slider("Maximum heart rate achieved", 71, 202, 80)
            slope = st.sidebar.slider("EKG Slope", 0, 2, 1)
            oldpeak = st.sidebar.slider("Oldpeak", 0.0, 6.2, 1.0)
            exang = st.sidebar.slider("Exercise induced angina", 0, 1, 1)
            ca = st.sidebar.slider("Number of major vessels", 0, 3, 1)
            thal = st.sidebar.slider("Thalium test result", 1, 3, 1)
            sex = st.sidebar.selectbox("sex", ('Female', 'Male'))
            if sex == "Female":
                sex = 0
            else:
                sex = 1
            age = st.sidebar.slider("Age", 29, 77, 30)
            data = {'cp': cp,
                    'thalach': thalach,
                    'slope': slope,
                    'oldpeak': oldpeak,
                    'exang': exang,
                    'ca':ca,
                    'thal':thal,
                    'sex': sex,
                    'age':age}
            features = pd.DataFrame(data, index=[0])
            return features
        input_df = user_input_manual()

    if st.sidebar.button('Predict!'):
            df = input_df.copy()
            st.write(df)
            with open("generate_heart_disease.pkl", "rb") as file:
                 loaded_model = pickle.load(file)
            prediction = loaded_model.predict(df)
            result = ['No Heart Disease' if prediction == 0 else 'Yes Heart Disease']
            st.subheader('Prediction: ')
            output = str(result[0])
            with st.spinner('Wait for it...'):
                time.sleep(4)
                st.success(f"Prediction of this app is: **{output}**")

# Home page
if nav == "Home":
    st.title("Capstone Project DQLAB")
    st.write('''
    **Machine Learning and Artificial Intelligence Bootcamp batch 8**
    
    Assalamu alaikum warakhmatullahi wabarakatuh
    
    Saya **Budi Pribadi**, pemerhati teknologi informasi, sangat tertarik pada Machine Learning dan Artificial Intelligence,  
    saya mengikuti Machine Learning dan Artificial Intelligence Bootcamp di DQLAB. ini adalah proyek pertama saya.
    ''')
    st.image("https://hips.hearstapps.com/hmg-prod.s3.amazonaws.com/images/heart-disease-1552649741.jpg", width=300, caption=("Cardiovascular Disease"))

    st.write('''        
    ***Project Overview***  
    Cardiovascular disease (CVDs) atau penyakit jantung merupakan penyebab kematian nomor satu secara global dengan 17,9 juta kasus kematian setiap tahunnya.   
    Penyakit jantung disebabkan oleh hipertensi, obesitas, dan gaya hidup yang tidak sehat. Deteksi dini penyakit jantung perlu dilakukan pada kelompok risiko tinggi agar  
    dapat segera mendapatkan penanganan dan pencegahan. Sehingga tujuan bisnis yang ingin dicapai yaitu membentuk model prediksi penyakit jantung pada pasien   
    berdasarkan feature-feature yang ada untuk membantu para dokter melakukan diagnosa secara tepat. Harapannya agar penyakit jantung dapat ditangani lebih awal.   
    Dengan demikian, diharapkan juga angka kematian akibat penyakit jantung dapat turun
    
    ''')

    st.write('''   
    ***Project Objective***   
    Tujuan dari Capstone Project adalah melakukan data preprocessing termasuk Exploratory Data Analysis untuk menggali insight dari data pasien penderita    
    penyakit jantung hingga proses feature selection dan dimensionality reduction. Hasil akhir yang ingin dicapai yaitu mendapatkan insight data  penderita penyakit   
    jantung dan data yang siap untuk dimodelkan pada tahap selanjutnya.
    ''')

elif nav == "Dataset":
    st.title("Gambaran Dataset")
    st.write('''
    
    Dataset yang digunakan adalah data Heart Disease yang diunduh dari UCI ML: https://archive.ics.uci.edu/dataset/45/heart+disease

    Dataset yang digunakan ini berasal dari tahun 1988 dan terdiri dari empat database: Cleveland, Hungaria, Swiss, dan Long Beach V.    
    Bidang "target" mengacu pada adanya penyakit jantung pada pasien. Ini adalah bilangan bulat bernilai 0 = tidak ada penyakit dan 1 = penyakit.  
    Dataset heart disease terdiri dari 1025 baris data dan 13 atribut + 1 target. Dataset ini memiliki 14 kolom yaitu:

    1. **age**: variabel ini merepresentasikan usia pasien yang diukur dalam tahun.
    2. **sex**: variabel ini merepresentasikan jenis kelamin pasien dengan   
          nilai 1 untuk laki-laki dan   
          nilai 0 untuk perempuan.
    3. **cp (Chest pain type)**: variabel ini merepresentasikan jenis nyeri dada yang dirasakan oleh pasien dengan 4 nilai kategori yang mungkin:   
          nilai 0 mengindikasikan nyeri dada tipe angina,   
          nilai 1 mengindikasikan nyeri dada tipe nyeri tidak stabil,   
          nilai 2 mengindikasikan nyeri dada tipe nyeri tidak stabil yang parah, dan   
          nilai 3 mengindikasikan nyeri dada yang tidak terkait dengan masalah jantung.
    4. **trestbps (Resting blood pressure)**: variabel ini merepresentasikan tekanan darah pasien pada saat istirahat, diukur dalam mmHg (milimeter   
       air raksa (merkuri)).
    5. **chol** (Serum cholestoral): variabel ini merepresentasikan kadar kolesterol serum dalam darah pasien, diukur dalam mg/dl (miligram per desiliter).
    6. **fbs** (Fasting blood sugar): variabel ini merepresentasikan kadar gula darah pasien saat puasa (belum makan) dengan   
         nilai 1 jika kadar gula darah > 120 mg/dl dan   
         nilai 0 jika tidak.
    7. **restecg (Resting electrocardiographic results)**: variabel ini merepresentasikan hasil elektrokardiogram pasien saat istirahat dengan 3 nilai kategori   
       yang mungkin:   
          nilai 0 mengindikasikan hasil normal,   
          nilai 1 mengindikasikan adanya kelainan gelombang ST-T, dan   
          nilai 2 mengindikasikan hipertrofi ventrikel kiri.
    8. **thalach (Maximum heart rate achieved)**: variabel ini merepresentasikan detak jantung maksimum yang dicapai oleh pasien selama tes olahraga,   
       diukur dalam bpm (denyut per menit).
    9. **exang (Exercise induced angina)**: variabel ini merepresentasikan apakah pasien mengalami angina (nyeri dada) yang dipicu oleh aktivitas olahraga, dengan   
         nilai 1 jika ya dan   
         nilai 0 jika tidak.
    10. **oldpeak**: variabel ini merepresentasikan seberapa banyak ST segmen menurun atau depresi saat melakukan aktivitas fisik dibandingkan saat istirahat.
    11. **slope**: variabel ini merepresentasikan kemiringan segmen ST pada elektrokardiogram (EKG) selama latihan fisik maksimal dengan 3 nilai kategori.
    12. **ca (Number of major vessels)**: variabel ini merepresentasikan jumlah pembuluh darah utama (0-3) yang terlihat pada pemeriksaan flourosopi.
    13. **thal**: variabel ini merepresentasikan hasil tes thalium scan dengan 3 nilai kategori yang mungkin:  
        nilai 1: menunjukkan kondisi normal.  
        nilai 2: menunjukkan adanya defek tetap pada thalassemia.  
        nilai 3: menunjukkan adanya defek yang dapat dipulihkan pada thalassemia.
     14. **target***: 0 = tidak ada penyakit dan 1 = penyakit.
    ''')

   # show Dataset
    st.write('''
    ***Show Dataset***
    ''')
    st.dataframe(df.head())
    st.write(f'''
    ***Dataset Shape*** {(df.shape)} ''')
    st.write('''
    ***Data Summary***
    ''')
    st.dataframe(df.describe())
    st.write('''
    ***Dataset Count  Visualisation***
    ''')
    views = st.radio("Select Visulisation", ("Age", "Sex", "Chest pain type", "Resting blood pressure", "Serum cholestoral", "Fasting blood sugar","Resting electrocardiographic results",
                     "Maximum heart rate achieved", "Exercise induced angina", "Oldpeak", "Slope", "Number of major vessels", "Thal", "Target" ), horizontal=True)

    if views == "Age":
      st.bar_chart(df['age'].value_counts())
    elif views == "Sex":
      st.bar_chart(df['sex'].value_counts())
      st.write('''nilai 0 untuk perempuan    
                  nilai 1 untuk laki-laki 
                  ''')
    elif views == "Chest pain type":
      st.bar_chart(df['cp'].value_counts())
      st.write('''nilai 0 mengindikasikan nyeri dada tipe angina,   
          nilai 1 mengindikasikan nyeri dada tipe nyeri tidak stabil,   
          nilai 2 mengindikasikan nyeri dada tipe nyeri tidak stabil yang parah, dan   
          nilai 3 mengindikasikan nyeri dada yang tidak terkait dengan masalah jantung
          ''')
    elif views == "Resting blood pressure":
      st.bar_chart(df['trestbps'].value_counts())
    elif views == "Serum cholestoral":
      st.bar_chart(df['chol'].value_counts())
    elif views == "Fasting blood sugar":
      st.bar_chart(df['fbs'].value_counts())
      st.write('''nilai 1 jika kadar gula darah > 120 mg/dl    
                  nilai 0 jika tidak
      ''')
    elif views == "Resting electrocardiographic results":
      st.bar_chart(df['restecg'].value_counts())
      st.write('''nilai 0 mengindikasikan hasil normal,   
                  nilai 1 mengindikasikan adanya kelainan gelombang ST-T  
                  nilai 2 mengindikasikan hipertrofi ventrikel kiri
           ''')
    elif views == "Maximum heart rate achieved":
      st.bar_chart(df['thalach'].value_counts())
    elif views == "Exercise induced angina":
      st.bar_chart(df['exang'].value_counts())
      st.write('''nilai 1 jika ya    
                 nilai 0 jika tidak.
      ''')
    elif views == "Oldpeak":
      st.bar_chart(df['oldpeak'].value_counts())
    elif views == "Slope":
      st.bar_chart(df['slope'].value_counts())
    elif views == "Number of major vessels":
      st.bar_chart(df['ca'].value_counts())
    elif views == "Thal":
      st.bar_chart(df['thal'].value_counts())
      st.write('''nilai 1: menunjukkan kondisi normal.  
                  nilai 2: menunjukkan adanya defek tetap pada thalassemia.  
                  nilai 3: menunjukkan adanya defek yang dapat dipulihkan pada thalassemia.
      ''')
    elif views == "Target":
      st.bar_chart(df['target'].value_counts())
      st.write('''0 = tidak ada penyakit dan 1 = penyakit.''')

elif nav == "Exploratory Data Analysis":
    st.header("Exploratory Data Analysis")
    st.write('''
    ***Data Uniqueness***
    ''')
    st.write(df.nunique().to_frame().transpose())
    st.write(''' Data **'ca'** dan **'thal'** keunikan datanya melebihi kategori yang ada''')
    st.write('''Before Handling:''')
    st.write('''Data 'ca' ''')
    st.write(df['ca'].value_counts().to_frame().transpose())
    st.write('''Data 'ca' terdapat kategori 4 yang salah input sebanyak 18 record''')
    st.dataframe(df[df['ca']==4])
    st.write('''Data 'thal' ''')
    st.write(df['thal'].value_counts().to_frame().transpose())
    st.write('''Data 'thal' terdapat kategori 0 yang salah input sebanyak 7 record''')
    st.dataframe(df[df['thal'] ==0])
    st.write('''Handling:''')
    st.write('''Ganti kolom 'ca' yang bernilai '4' menjadi NaN''')
    st.write('''Ganti kolom 'thal' yang bernilai '0' menjadi NaN''')
    st.write('''After Handling:''')
    st.write('''Data 'ca' ''')
    st.dataframe(df.loc[df['ca']==4, 'ca'] == np.NaN)
    st.write('''Data 'thal' ''')
    st.dataframe(df.loc[df['thal']==0, 'thal'] == np.NaN)

    st.write('''
    ***Handling Missing Value***
    ''')
    st.write('''Mengisi missing value dengan modus''')
    modus_ca = df['ca'].mode()[0]
    df['ca'] = df['ca'].fillna(modus_ca)
    modus_thal = df['thal'].mode()[0]
    df['thal'] = df['thal'].fillna(modus_ca)
    st.write('''After Handling:''')
    st.write(df.isnull().sum().to_frame().transpose())

    st.write('''
        ***Handling Duplicate***
    ''')
    st.write('''Before Handling''')
    st.write(df.duplicated().sum())
    st.write('''Handling:''')
    st.write('''Menghapus data duplikat dan mempertahankan data pertama''')
    st.write('''After Handling''')
    df.drop_duplicates(inplace=True, keep='first' )
    st.write(df.duplicated().sum())
    st.write(df.shape)

    st.write('''
    ***Handling Outliers***
    ''')
    st.write('''Menampilkan outlier pada data numerik ''')

    views = st.radio("Select Visulisation", ("Age", "Resting blood pressure", "Serum cholestoral", "Maximum heart rate achieved", "Oldpeak"))

    if views == "Age":
        box_plot = alt.Chart(df).mark_boxplot().encode(x='target', y='age')
        st.altair_chart(box_plot)
    elif views == "Resting blood pressure":
        box_plot = alt.Chart(df).mark_boxplot().encode(x='target', y='trestbps')
        st.altair_chart(box_plot)
    elif views == "Serum cholestoral":
        box_plot = alt.Chart(df).mark_boxplot().encode(x = 'target', y = 'chol')
        st.altair_chart(box_plot)
    elif views == "Maximum heart rate achieved":
        box_plot = alt.Chart(df).mark_boxplot().encode(x='target', y='thalach')
        st.altair_chart(box_plot)
    elif views == "Oldpeak":
        box_plot = alt.Chart(df).mark_boxplot().encode(x='target', y='oldpeak')
        st.altair_chart(box_plot)


    st.write('''
    **Jumlah data outlier:**
    ''')
    continous_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
    def outliers(data_out, drop=False):
        for each_feature in data_out.columns:
            feature_data = data_out[each_feature]
            Q1 = np.percentile(feature_data, 25.)
            Q3 = np.percentile(feature_data, 75.)
            IQR = Q3 - Q1
            outlier_step = IQR * 1.5
            outliers = feature_data[
                ~((feature_data >= Q1 - outlier_step) & (feature_data <= Q3 + outlier_step))].index.tolist()
            if not drop:
                st.write('Untuk feature {}, Jumlah Outlier: {}'.format(each_feature, len(outliers)))
            if drop:
                df.drop(outliers, inplace=True, errors='ignore')
                st.write('Outliers dari {} feature dihapus'.format(each_feature))
    outliers(df[continous_features])
    st.write('''**Handling: menghapus data outlier:**''')
    outliers(df[continous_features], drop=True)
    st.write('''
        **Jumlah data outlier setelah handling:**
        ''')
    outliers(df[continous_features])

    st.write('''
    ***Correlation Matrix***
    ''')
    st.write('''Menampilkan korelasi antar fitur''')
    plt.figure(figsize=(20, 20))
    corr = df.corr()
    sns.heatmap(corr, annot=True, linewidth=.5, cmap="magma")
    plt.title('Korelasi Antar Fitur', fontsize=30)
    st.pyplot(plt.gcf())

    st.write('''Urutan korelasi terhadap target''')
    corr_matrix = df.corr()
    st.dataframe(corr_matrix['target'].sort_values().to_frame().transpose())
    st.write('''
    Kesimpulan:  
    'cp', 'thalach', dan 'slope' berkorelasi positif cukup kuat dengan 'target'.   
    'oldpeak', 'exang', 'ca', 'thal', 'sex', dan 'age' berkorelasi cukup kuat dengan 'target'.  
    'fbs', 'chol', 'trestbps', dan 'restecg' memiliki korelasi yang lemah dengan 'target'.  
     
     Feature yang dipilih yaitu :  
     'cp', 'thalach', 'slope', 'oldpeak', 'exang', 'ca', 'thal', 'sex', dan 'age'   
     untuk dianalisa lebih lanjut.
    ''')

elif nav == "Dimensionality Reduction":
    st.header("Scaling Data dan Dimensionality Reduction")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df.drop("target", axis=1))

    feature_number = len(X_scaled[0])
    pca = PCA(n_components=feature_number)

    pca.fit(X_scaled)
    variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(variance_ratio)

    st.write('''Menampilkan Scree Plot''')
    plt.plot(range(1, len(variance_ratio) + 1), variance_ratio, marker='o')
    plt.xlabel('Komponen Utama ke-')
    plt.ylabel('Varians (Nilai Eigen)')
    plt.title('Scree Plot')
    st.pyplot(plt.gcf())

    pca = PCA(n_components=9)
    heart_data_reduced = pca.fit_transform(X_scaled)

    feature_names = df.drop('target', axis=1).columns.to_list()
    component_names = [f"PC{i + 1}" for i in range(pca.n_components_)]

    for component, component_name in zip(pca.components_, component_names):
        feature_indices = component.argsort()[::-1]
        retained_features = [feature_names[idx] for idx in feature_indices[:pca.n_components_]]

    df = df[['cp', 'thalach', 'slope', 'oldpeak', 'exang', 'ca', 'thal', 'sex', 'age', 'target']]
    st.write('''Feature yang terpilih''')
    st.dataframe(df.columns)


elif nav == "Machine Learning Modelling":
    st.header("Machine Learning Modelling")
    df = df[['cp', 'thalach', 'slope', 'oldpeak', 'exang', 'ca', 'thal', 'sex', 'age', 'target']]
    X = df.drop(columns='target')
    y = df[['target']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.write('''
    ***Machine Learning Models Accuracy***
    ''')
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    log_before = accuracy_score(y_test, y_pred)
    st.write("Accuracy score of Logistic Regression is ", accuracy_score(y_test, y_pred))

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    dec_before = accuracy_score(y_test, y_pred)
    st.write("Accuracy score of Decision Tree is ", accuracy_score(y_test, y_pred))

    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    ran_before = accuracy_score(y_test, y_pred)
    st.write("Accuracy score of Random Forest is ", accuracy_score(y_test, y_pred))

    clf = MLPClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    mlp_before = accuracy_score(y_test, y_pred)
    st.write("Accuracy score of MLP is ", accuracy_score(y_test, y_pred))

    st.write('''
    ***Hyperparameter Tuning***
    ''')
    clf = LogisticRegression()

    param_grid = {
        'max_iter': [10, 50, 100, 200, 500],
        'multi_class': ['auto', 'multinomial'],
        'solver': ['lbfgs', 'newton-cholesky']
        }

    gs1 = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='roc_auc'
        )

    fit_clf_lg = gs1.fit(X_train, y_train)

    st.write("Logistic Regression best score:", (fit_clf_lg.best_score_))

    clf = DecisionTreeClassifier()

    param_grid = {'min_samples_leaf': [1, 2, 3, 4, 5],
        'max_depth': [None, 3, 5, 10],
        'criterion': ['gini', 'entropy']}

    gs1 = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring='roc_auc'
        )

    fit_clf_dt = gs1.fit(X_train, y_train)

    st.write("Decision Tree best score:", (fit_clf_dt.best_score_))

    clf = RandomForestClassifier()

    param_grid = {'n_estimators': [10, 50, 100, 200, 500],
        'max_depth': [None, 3, 5, 10],
        'criterion': ['gini', 'entropy']}

    gs1 = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
         cv=5,
        n_jobs=-1,
        scoring='roc_auc'
        )

    fit_clf_rf = gs1.fit(X_train, y_train)

    st.write("Random Forest best score:", (fit_clf_rf.best_score_))

    clf = MLPClassifier()

    param_grid1 = {'hidden_layer_sizes': [(10,), (50,), (100,)],
        'activation': ['relu', 'tanh', 'logistic'],
        'solver': ['sgd', 'adam']}

    gs1 = GridSearchCV(
        estimator=clf,
        param_grid=param_grid1,
        cv=5,
        n_jobs=-1,
        scoring='roc_auc'
        )

    fit_clf_mlp = gs1.fit(X_train, y_train)

    st.write("MLP best score:", (fit_clf_mlp.best_score_))

    st.write("""
    ***Comparison before and After Hyperparameter Tuning***
    """)
    y_pred = fit_clf_lg.predict(X_test)
    log_after_tuned = accuracy_score(y_test, y_pred)

    y_pred = fit_clf_dt.predict(X_test)
    dec_after_tuned = accuracy_score(y_test, y_pred)

    y_pred = fit_clf_rf.predict(X_test)
    ran_after_tuned = accuracy_score(y_test, y_pred)

    y_pred = fit_clf_mlp.predict(X_test)
    mlp_after_tuned = accuracy_score(y_test, y_pred)

    st.write("Logistic Regression Accuracy Before Tuning: ", log_before)
    st.write("Logistic Regression Accuracy After Tuning: ", log_after_tuned)
    st.write("Decision Tree Accuracy Before Tuning: ", dec_before)
    st.write("Decision Tree Accuracy After Tuning: ", dec_after_tuned)
    st.write("Random Forest Accuracy Before Tuning: ", ran_before)
    st.write("Random Forest Accuracy After Tuning: ", ran_after_tuned)
    st.write("MLP Accuracy Before Tuning: ", mlp_before)
    st.write("MLP Accuracy After Tuning: ", mlp_after_tuned)

    st.write('''
    ***ROC analysis***
    ''')
    y_pred_logreg = fit_clf_lg.predict_proba(X_test)[:, 1]
    y_pred_rf = fit_clf_rf.predict_proba(X_test)[:, 1]
    y_pred_dt = fit_clf_dt.predict_proba(X_test)[:, 1]
    y_pred_mlp = fit_clf_mlp.predict_proba(X_test)[:, 1]

    auc_logreg = roc_auc_score(y_test, y_pred_logreg)
    auc_rf = roc_auc_score(y_test, y_pred_rf)
    auc_dt = roc_auc_score(y_test, y_pred_dt)
    auc_mlp = roc_auc_score(y_test, y_pred_mlp)

    st.write(f"AUC-ROC for Logistic Regression: {auc_logreg}")
    st.write(f"AUC-ROC for Decisiont Tree: {auc_dt}")
    st.write(f"AUC-ROC for Random Forest: {auc_rf}")
    st.write(f"AUC-ROC for MLP: {auc_mlp}")


    def plot_roc_curves(y_test, y_pred_logreg, y_pred_rf, y_pred_dt, y_pred_mlp):

        fpr_logreg, tpr_logreg, _ = roc_curve(y_test, y_pred_logreg)
        fpr_rf, tpr_rf, _ = roc_curve(y_test, y_pred_rf)
        fpr_dt, tpr_dt, _ = roc_curve(y_test, y_pred_dt)
        fpr_mlp, tpr_mlp, _ = roc_curve(y_test, y_pred_mlp)

        plt.plot(fpr_logreg, tpr_logreg, label=f'Logistic Regression (AUC = {auc_logreg:.2f})')
        plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC = {auc_rf:.2f})')
        plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC = {auc_dt:.2f})')
        plt.plot(fpr_mlp, tpr_mlp, label=f'MLP (AUC = {auc_mlp:.2f})')

        plt.plot()

        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves for Heart Disease Prediction Models')
        plt.legend()
        st.pyplot(plt.gcf())

    plot_roc_curves(y_test, y_pred_logreg, y_pred_rf, y_pred_dt, y_pred_mlp)


    def find_rates_for_thresholds(y_test, Y_pred, thresholds):
        fpr_list = []
        tpr_list = []
        for threshold in thresholds:
            y_pred_binary = (y_pred > threshold).astype(int)
            tn, fp, fn, tp = confusion_matrix(y_test, y_pred_binary).ravel()
            fpr = fp / (fp + tn)
            tpr = tp / (tp + fn)
            fpr_list.append(fpr)
            tpr_list.append(tpr)
        return (fpr_list, tpr_list)


    thresholds = np.arange(0, 1.1, 0.1)
    fpr_logreg, tpr_logreg = find_rates_for_thresholds(y_test, y_pred_logreg, thresholds)
    fpr_rf, tpr_rf = find_rates_for_thresholds(y_test, y_pred_rf, thresholds)
    fpr_dt, tpr_dt = find_rates_for_thresholds(y_test, y_pred_dt, thresholds)
    fpr_mlp, tpr_mlp = find_rates_for_thresholds(y_test, y_pred_mlp, thresholds)

    summary_df = pd.DataFrame({
        'False Positive Rate (LogReg)': fpr_logreg,
        'True Positive Rate (LogReg)': tpr_logreg,
        'False Positive Rate (RF)': fpr_rf,
        'True Positive Rate (RF)': tpr_rf,
        'False Positive Rate (DT)': fpr_dt,
        'True Positive Rate (DT)': tpr_dt,
        'False Positive Rate (MLP)': fpr_mlp,
        'True Positive Rate (MLP)': tpr_mlp,
        'Thresholds': thresholds})

    summary_df


    def find_best_threshold(y_test, y_pred):
        fpr, tpr, thresholds = roc_curve(y_test, y_pred)
        optimal_idx = np.argmax(tpr - fpr)
        return thresholds[optimal_idx]

    best_threshold_logreg = find_best_threshold(y_test, y_pred_logreg)
    best_threshold_rf = find_best_threshold(y_test, y_pred_rf)
    best_threshold_dt = find_best_threshold(y_test, y_pred_dt)
    best_threshold_mlp = find_best_threshold(y_test, y_pred_mlp)

    st.write(f"Best threshold for Logistic Regression: {best_threshold_logreg}")
    st.write(f"Best threshold for Random Forest: {best_threshold_rf}")
    st.write(f"Best threshold for Decision Tree: {best_threshold_dt}")
    st.write(f"Best threshold for MLP: {best_threshold_mlp}")

    import pickle

    pklname = "generate_heart_disease.pkl"

    with open(pklname, 'wb') as file:
         pickle.dump(fit_clf_rf, file)

elif nav == "Prediction":
    st.header("Heart Disease Prediction")
    st.write(''' 
    Data yang digunakan adalah UCI ML Heart Disease Dataset
    ''')
    heart()

elif nav =="About me":
    st.header("Budi Pribadi")
    
    st.write('''**08161622331**''')
    st.write('''**bepe125@gmail.com**''')
