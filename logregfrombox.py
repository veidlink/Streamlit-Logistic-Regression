import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


st.title("Сервис для визуализации логистической регрессии вашего датасета")

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
if uploaded_file is not None:
    st.write('### Так выглядит твой датафрейм:')
    input_df = pd.read_csv(uploaded_file)
    input_df = input_df.dropna()

    st.dataframe(input_df.head())

    X_names = st.multiselect('Select features', input_df.columns, key='Features')
    y_names = st.selectbox('Select target', input_df.columns)

    X = input_df[X_names]
    y = input_df[y_names]

    # Выводим поле ввода для значения alpha
    # lr = st.number_input("Введите значение learning rate", value=0.001)
    # alphaLASSO = st.number_input("Введите значение alpha LASSO", value=0.10)
    # alphaRIDGE = st.number_input("Введите значение alpha RIDGE", value=0.00)



    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


    # clf = LogisticRegression(C=1, penalty='l1', solver='liblinear')
    clf = LogisticRegression()
    # Обучаем модель
    clf.fit(X_train, y_train)

    # clf.fit(X_train,y_train)
    y_pred = clf.predict(X_test)


    feature_names = X_names # названия признаков

    coef = clf.coef_ # массив коэффициентов

    # Filter out features with weight equal to 0
    nonzero_features = [feature for feature, weight in zip(feature_names, coef[0]) if np.round(weight, 3) != 0]
    nonzero_coef = [weight for weight in coef[0] if np.round(weight, 3) != 0]
    nonzero_dict = dict(zip(nonzero_features, nonzero_coef))

    def accuracy(y_pred, y_real):
        return np.sum(y_pred==y_test)/len(y_test)

    acc = accuracy(y_pred, y_test)
    acc2 = accuracy(y_pred, y_train)

    st.write('Точность на тренировочном датафрейме:', acc2)
    st.write('Точность на тестовом датафрейме:', acc)

    st.write('Предсказанные значения:', y_pred)
    # Create bar chart with filtered features and coef using Seaborn
    st.write('''
    ### Barplot фич вашего датасета
    # ''')

    selected_features = st.multiselect('Выберите фичи', nonzero_dict, key='Barplot')
    selected_keys = [nonzero_dict[i] for i in selected_features]

    fig1 = plt.figure(figsize=(5,5))
    sns.barplot(x=selected_features, y=selected_keys)
    # Add title and axis labels
    plt.title('Nonzero features')
    plt.xlabel('Features')
    plt.ylabel('Weight')
    plt.xticks(rotation=90)
    sns.set_style("dark")
    # Show the plot
    st.pyplot(fig1)

    st.write('''
    ### Scatterplot фич вашего датасета
    # ''')

    selected_features = np.array(st.multiselect('Выберите фичи', nonzero_dict, key='Scatterplot'))
    selected_keys = np.array([nonzero_dict[i] for i in selected_features])

    fig2 = plt.figure(figsize=(4,4))
    sns.scatterplot(x=selected_features, y=selected_keys,
    hue=selected_features, legend=False, size=selected_keys, marker='s')
    # Add title and axis labels
    plt.title('Nonzero features')
    plt.xlabel('Features')
    plt.ylabel('Weight')
    plt.xticks(rotation=90)
    sns.set_style("dark")
    # Show the plot
    st.pyplot(fig2)

    st.write('''
    ### Lineplot фич вашего датасета
    # ''')

    selected_features = np.array(st.multiselect('Выберите фичи', nonzero_dict, key='Lineplot'))
    selected_keys = np.array([nonzero_dict[i] for i in selected_features])

    fig3 = plt.figure(figsize=(4,4))

    sns.lineplot(x=selected_features, y=selected_keys, marker='s')
    # Add title and axis labels
    plt.title('Nonzero features')
    plt.xlabel('Features')
    plt.ylabel('Weight')
    plt.xticks(rotation=90)
    sns.set_style("dark")
    st.pyplot(fig3)

else:
    st.write('### <span style="color:red"> Вы не загрузили набор данных!</span>', unsafe_allow_html=True)