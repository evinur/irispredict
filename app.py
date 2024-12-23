import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset Iris
def load_data():
    # Menggunakan dataset iris dari file CSV yang disediakan
    iris = pd.read_csv("Iris.csv")
    return iris

# Load dataset
df = load_data()

# Pisahkan fitur dan target
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# Normalisasi fitur
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split data menjadi train dan test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# KNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
knn_pred = knn.predict(X_test)
knn_accuracy = accuracy_score(y_test, knn_pred)

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_accuracy = accuracy_score(y_test, dt_pred)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_pred = nb.predict(X_test)
nb_accuracy = accuracy_score(y_test, nb_pred)

# Create a sidebar with buttons
st.sidebar.title("Menu")
# Gunakan session_state untuk menyimpan status tab
if "tab" not in st.session_state:
    st.session_state.tab = "Home"  # Default tab

tab1_button = st.sidebar.button("Home")
tab2_button = st.sidebar.button("Prediksi Bunga Iris")

# Tetap pada tab yang dipilih jika tombol diklik
if tab1_button:
    st.session_state.tab = "Home"
elif tab2_button:
    st.session_state.tab = "Prediksi Bunga Iris"


if st.session_state.tab == "Home":
    # Menampilkan aplikasi
    st.title('Aplikasi Prediksi Jenis Bunga Iris')
    st.write('Aplikasi ini menggunakan model Machine Learning, yaitu (K-Nearest Neighbor, Decission Tree dan Naive Bayes) untuk memprediksi jenis bunga iris berdasarkan fitur sepal_length, sepal_width, petal_length, dan petal_width.')
    st.header('Data Bunga Iris')
    st.dataframe(df)

    # Menampilkan Missing Value
    st.header('Jumlah Missing Value')
    st.write(df.isnull().sum())

    # Menampilkan normalisasi data
    st.header('Normalisasi Data')
    st.dataframe(MinMaxScaler().fit_transform(df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]))

    # Menampilkan akurasi model
    st.header('Akurasi Model')
    st.write(f"Akurasi KNN : {knn_accuracy * 100:.2f}%")
    st.write(f"Akurasi Decision Tree : {dt_accuracy * 100:.2f}%")
    st.write(f"Akurasi Naive Bayes : {nb_accuracy * 100:.2f}%")

    # Evaluasi model
    st.header('Evaluasi Model')

    # Klasifikasi KNN
    knn_report = classification_report(y_test, knn_pred, target_names=['setosa', 'versicolor', 'virginica'], output_dict=True)
    knn_report_df = pd.DataFrame(knn_report).transpose()
    st.write("Klasifikasi KNN")
    st.dataframe(knn_report_df)
    # Menampilkan confusion matrix untuk KNN
    st.write("Confusion Matrix KNN")
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, knn_pred), annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix KNN")
    st.pyplot(plt)

    # Klasifikasi Decision Tree
    dt_report = classification_report(y_test, dt_pred, target_names=['setosa', 'versicolor', 'virginica'], output_dict=True)
    dt_report_df = pd.DataFrame(dt_report).transpose()
    st.write("Klasiifikasi Decision Tree")
    st.dataframe(dt_report_df)
    # Menampilkan confusion matrix untuk Decision Tree
    st.write("Confusion Matrix Decision Tree")
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, dt_pred), annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix Decision Tree")
    st.pyplot(plt)

    # Klasifikasi Naive Bayes
    nb_report = classification_report(y_test, nb_pred, target_names=['setosa', 'versicolor', 'virginica'], output_dict=True)
    nb_report_df = pd.DataFrame(nb_report).transpose()
    st.write("Klasifikasi Naive Bayes")
    st.dataframe(nb_report_df)
    # Menampilkan confusion matrix untuk Naive Bayes
    st.write("Confusion Matrix Naive Bayes")
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, nb_pred), annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix Naive Bayes")
    st.pyplot(plt)

elif st.session_state.tab == "Prediksi Bunga Iris":
    # Prediksi
    st.header('Prediksi Jenis Bunga Iris')
    # Input data dari pengguna
    sepal_length = st.number_input('Sepal Length (cm)', min_value=0.0, max_value=10.0, value=5.0)
    sepal_width = st.number_input('Sepal Width (cm)', min_value=0.0, max_value=10.0, value=3.0)
    petal_length = st.number_input('Petal Length (cm)', min_value=0.0, max_value=10.0, value=4.0)
    petal_width = st.number_input('Petal Width (cm)', min_value=0.0, max_value=10.0, value=1.0)

    # Menampilkan hasil prediksi dan akurasi saat tombol prediksi diklik
    if st.button('Prediksi'):
        # Normalisasi input
        input_data = pd.DataFrame([[sepal_length, sepal_width, petal_length, petal_width]], columns=['sepal_length', 'sepal_width', 'petal_length', 'petal_width'])
        input_scaled = scaler.transform(input_data)

        # Prediksi dengan model
        prediction_knn = knn.predict(input_scaled)
        prediction_dt = dt.predict(input_scaled)
        prediction_nb = nb.predict(input_scaled)

        st.write(f"Prediksi dengan KNN : {prediction_knn[0]}")
        st.write(f"Prediksi dengan Decision Tree : {prediction_dt[0]}")
        st.write(f"Prediksi dengan Naive Bayes : {prediction_nb[0]}")
