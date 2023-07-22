import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import streamlit as st
from sklearn import datasets
from sklearn.model_selection import train_test_split

class LogReg:
    def __init__(self, lr=0.001, n_features=None, alphaLASSO=0.1, alphaRIDGE=0):
        self.lr = lr
        self.alphaLASSO=alphaLASSO
        self.alphaRIDGE=alphaRIDGE
        self.n_features = n_features
        self.weights = None
        self.bias = None

    def sigmoid(x):
        return 1/(1+np.exp(-x))
        
    def fit(self, X, y):
        self.n_samples = X.shape[0]
        self.n_features = X.shape[1]
        self.weights = np.zeros(self.n_features)
        self.bias = 0

        epochs = 1000

        for _ in range(epochs):
            linear_pred = np.dot(X, self.weights) + self.bias
            predictions = LogReg.sigmoid(linear_pred)
            
            grad_w = (1/self.n_samples) * (2*np.dot(X.T, (predictions-y))) + self.alphaLASSO*np.sign(self.weights) + self.alphaRIDGE*2*self.weights
            grad_b = (1/self.n_samples) * (2*np.sum(predictions-y))

            self.weights = self.weights - self.lr*grad_w
            self.bias = self.bias - self.lr*grad_b  
        
        self.final_linear_predictions = linear_pred
        self.final_predictions = predictions 

    def predict(self, X):
        linear_pred = np.dot(X, self.weights) + self.bias
        y_pred = LogReg.sigmoid(linear_pred)
        label_pred = [0 if y<=0.5 else 1 for y in y_pred]
        return label_pred

st.title("Сервис для визуализации логистической регрессии вашего датасета")

st.write('### Так выглядит твой датафрейм:')

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=["csv"])
input_df = pd.read_csv(uploaded_file)
input_df = input_df.dropna()

st.dataframe(input_df.head())

X_names = st.multiselect('Select features', input_df.columns, key='Features')
y_names = st.selectbox('Select target', input_df.columns)

X = input_df[X_names]
y = input_df[y_names]

clf = LogReg()

# Выводим поле ввода для значения alpha
# lr = st.number_input("Введите значение learning rate", value=0.001)
alphaLASSO = st.number_input("Введите значение alpha LASSO", value=0.10)
alphaRIDGE = st.number_input("Введите значение alpha RIDGE", value=0.00)



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

# Обновляем значение alpha в экземпляре класса LogReg
# clf.lr = lr
clf.alphaLASSO = alphaLASSO
clf.alphaRIDGE = alphaRIDGE

# Обучаем модель
clf.fit(X_train, y_train)

# clf.fit(X_train,y_train)
y_pred = clf.predict(X_test)



feature_names = X_names # названия признаков

weights = clf.weights # массив коэффициентов

 # Filter out features with weight equal to 0
nonzero_features = [feature for feature, weight in zip(feature_names, weights) if round(weight, 3) != 0]
nonzero_weights = [weight for weight in weights if round(weight, 3) != 0]
nonzero_dict = dict(zip(nonzero_features, nonzero_weights))

def accuracy(y_pred, y_real):
    return np.sum(y_pred==y_test)/len(y_test)

acc = accuracy(y_pred, y_test)
acc2 = accuracy(y_pred, y_train)

st.write('Точность на тренировочном датафрейме:', acc2)
st.write('Точность на тестовом датафрейме:', acc)

st.write('Предсказанные значения:', y_pred)
 # Create bar chart with filtered features and weights using Seaborn
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
