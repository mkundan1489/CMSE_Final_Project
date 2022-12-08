import pandas as pd
from PIL import Image
import pickle
#import webbrowser
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
# from sklearn import metrics
# from sklearn.utils import resample
from imblearn.over_sampling import SMOTENC, RandomOverSampler, KMeansSMOTE
# from pympler.util.bottle import redirect
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder
sns.set()
####
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, roc_curve, classification_report

intro = st.sidebar.button("INTRODUCTION")
if intro:
    st.title("Introduction to Thyroid Dataset")
    st.write("Thyroid dataset is a dataset for detection of thyroid diseases, in which patients diagnosed with "
             "hypothyroid or negative.")
    image = Image.open('ThyroidAnatomy.webp')
    st.image(image, caption='Thyroid Gland')
    st.write(
        "The dataset is obtained from the Kaggle. The Purpose of the this webb app is explore the different machine "
        "learning models and predict if the patients have thyroid.")
    st.write("Click on 'Features', to understand the features of the dataset.")


ft = st.sidebar.button('FEATURES')
if ft:
    st.text("Features of the Dataset:")
    st.write("age: Age of the patients")
    st.write("sex: Sex of the the patients")
    st.write("on_thyroxine: Wheteher the patient is on thyroxine")
    st.write("on_antithyroid_medication: Whether the patient is on antithyroid medication")
    st.write("sick: Whetehr the patient is sick")
    st.write("pregnant: Whetehr the patient is pregnant")
    st.write("thyroid_surgery: Whether the patient has undergone thyroid surgery")
    st.write("goitre: Whetehr the patient has goitre")
    st.write("tumor: Whether the patient has tumor")
    st.write("hypopituitary: whether patient has hypopituitary")
    st.write("psych: whether patient has psych")
    st.write("TSH: TSH (Thyroid Stimulating Hormone) level in the blodd(0.005 to 530)")
    st.write("T3: T3 level in the blood(0 to 11)")
    st.write("TT4: TT4(Thyroxine) level in the blood(2 to 430)")
    st.write("T4U: T4 Uptake in the blood level(0.25 to 2.12)")
    st.write("FTI: Thyroxine(T4)/Thyroid Binding Capacity (2 to 395)")
    st.write("Class: Negative or types of hypothyroid")

data = pd.read_csv('hypothyroid.csv')
data = data.drop(['TBG'], axis=1)
data = data.drop(['TSH_measured', 'T3_measured', 'TT4_measured', 'T4U_measured', 'FTI_measured', 'TBG_measured'],
                 axis=1)

# Now let's replace the '?' values with numpy nan
for column in data.columns:
    count = data[column][data[column] == '?'].count()
    if count != 0:
        data[column] = data[column].replace('?', np.nan)

# We can map the categorical values like below:
data['sex'] = data['sex'].map({'F': 0, 'M': 1})

# except for 'Sex' column all the other columns with two categorical data have same value 'f' and 't'.
# so instead of mapping indvidually, let's do a smarter work
for column in data.columns:
    if len(data[column].unique()) == 2:
        data[column] = data[column].map({'f': 0, 't': 1})

# this will map all the rest of the columns as we require. Now there are handful of column left with more than 2 categories.

data = pd.get_dummies(data, columns=['referral_source'])

lblEn = LabelEncoder()

data['Class'] = lblEn.fit_transform(data['Class'])

imputer = KNNImputer(n_neighbors=3, weights='uniform', missing_values=np.nan)
new_array = imputer.fit_transform(data)  # impute the missing values
# convert the nd-array returned in the step above to a Dataframe
new_data = pd.DataFrame(data=np.round(new_array), columns=data.columns)

## drop some unnecessary columns
new_data = new_data.drop(['on_thyroxine', 'query_on_thyroxine',
                          'on_antithyroid_medication',
                          'I131_treatment', 'query_hypothyroid', 'query_hyperthyroid', 'lithium',
                          'hypopituitary', 'psych', 'referral_source_STMW', 'referral_source_SVHC',
                          'referral_source_SVHD', 'referral_source_SVI', 'referral_source_other'], axis=1)

new_data = new_data.drop(['TSH'], axis=1)

new_data['Class'] = new_data['Class'].map({1: 'Negative', 0: 'Compensated Hypothyroid',
                                           2: 'Primary Hypothyroid', 3: 'Secondary Hypothyroid'})

x = new_data.drop(['Class'], axis=1)
y = new_data['Class']
rdsmple = RandomOverSampler()
x_sampled, y_sampled = rdsmple.fit_resample(x, y)

rdsmple = RandomOverSampler()
x_sampled, y_sampled = rdsmple.fit_resample(x, y)
x_sampled = pd.DataFrame(data=x_sampled, columns=x.columns)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_sampled, y_sampled, test_size=0.2, random_state=0)


def svm_classifier(X_train, X_test, y_train, y_test):
    classifier_svm = SVC(kernel='rbf', random_state=0)
    classifier_svm.fit(X_train, y_train)
    y_pred = classifier_svm.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    st.write("Train Score", ":", classifier_svm.score(X_train, y_train))
    st.write("Test Score", ":", classifier_svm.score(X_test, y_test))
    st.title("Confusion Matrix")
    st.write(cm)
    report = classification_report(y_test, y_pred, output_dict=True)
    df_forest = pd.DataFrame(report).transpose()
    st.title("Classification Report")
    st.write(df_forest)


def knn_classifier(X_train, X_test, y_train, y_test):
    classifier_knn = KNeighborsClassifier(metric='minkowski', p=2)
    classifier_knn.fit(X_train, y_train)
    y_pred = classifier_knn.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    st.write("Train Score", ":", classifier_knn.score(X_train, y_train))
    st.write("Test Score", ":", classifier_knn.score(X_test, y_test))
    st.title("Confusion Matrix")
    st.write(cm)
    report = classification_report(y_test, y_pred, output_dict=True)
    df_forest = pd.DataFrame(report).transpose()
    st.title("Classification Report")
    st.write(df_forest)


def tree_classifier(X_train, X_test, y_train, y_test):
    classifier_tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier_tree.fit(X_train, y_train)
    y_pred = classifier_tree.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    st.write("Train Score", ":", classifier_tree.score(X_train, y_train))
    st.write("Test Score", ":", classifier_tree.score(X_test, y_test))
    st.title("Confusion Matrix")
    st.write(cm)
    report = classification_report(y_test, y_pred, output_dict=True)
    df_forest = pd.DataFrame(report).transpose()
    st.title("Classification Report")
    st.write(df_forest)


def forest_classifier(X_train, X_test, y_train, y_test):
    classifier_forest = RandomForestClassifier(criterion='entropy', random_state=0)
    classifier_forest.fit(X_train, y_train)
    y_pred = classifier_forest.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    st.write("Train Score", ":", classifier_forest.score(X_train, y_train))
    st.write("Test Score", ":", classifier_forest.score(X_test, y_test))
    st.title("Confusion Matrix")
    st.write(cm)
    report = classification_report(y_test, y_pred, output_dict=True)
    df_forest = pd.DataFrame(report).transpose()
    st.title("Classification Report")
    st.write(df_forest)


mod = st.sidebar.checkbox("Check here to build models")
if mod:
    model = st.selectbox('Select the model you want to apply on the dataset.',
                         ('Select', 'SVM', 'KNN', 'Decision Tree', 'Random Forest'))
    # svm = st.sidebar.button("SVM")
    if model == 'SVM':
        st.title("SVM")
        svm_classifier(X_train, X_test, y_train, y_test)

    # knn = st.sidebar.button("KNN")
    if model == 'KNN':
        st.title("KNN")
        knn_classifier(X_train, X_test, y_train, y_test)

    # dt = st.sidebar.button("Decision Tree")
    if model == 'Decision Tree':
        st.title("Decision Tree")
        tree_classifier(X_train, X_test, y_train, y_test)

    # rf = st.sidebar.button("Random Forest")
    if model == 'Random Forest':
        st.title("Random Forest")
        forest_classifier(X_train, X_test, y_train, y_test)


def svm_classifier_acc(X_train, X_test, y_train, y_test):
    classifier_svm = SVC(kernel='rbf', random_state=0)
    classifier_svm.fit(X_train, y_train)
    y_pred = classifier_svm.predict(X_test)
    return classifier_svm.score(X_test, y_test)


def knn_classifier_acc(X_train, X_test, y_train, y_test):
    classifier_knn = KNeighborsClassifier(metric='minkowski', p=2)
    classifier_knn.fit(X_train, y_train)
    y_pred = classifier_knn.predict(X_test)
    return classifier_knn.score(X_test, y_test)


def tree_classifier_acc(X_train, X_test, y_train, y_test):
    classifier_tree = DecisionTreeClassifier(criterion='entropy', random_state=0)
    classifier_tree.fit(X_train, y_train)
    y_pred = classifier_tree.predict(X_test)
    return classifier_tree.score(X_test, y_test)


def forest_classifier_acc(X_train, X_test, y_train, y_test):
    classifier_forest = RandomForestClassifier(criterion='entropy', random_state=0)
    classifier_forest.fit(X_train, y_train)
    y_pred = classifier_forest.predict(X_test)
    return classifier_forest.score(X_test, y_test)


comp = st.sidebar.button('Compare all models')
if comp:
    st.title("Bar Graph of four models:")
    st.write("Below graph shows that SVM has least accuracy while the Random Forest model has the best accuracy among "
             "all.")
    models = ['SVM', 'KNN', 'Decision Tree', 'Random Forest']
    fig = plt.figure()
    plt.bar(models, [svm_classifier_acc(X_train, X_test, y_train, y_test),
                     knn_classifier_acc(X_train, X_test, y_train, y_test),
                     tree_classifier_acc(X_train, X_test, y_train, y_test),
                     forest_classifier_acc(X_train, X_test, y_train, y_test)])

    plt.xlabel("Machine Learning Models")
    plt.ylabel("Accuracy")
    plt.title("Accuracy VS Models")
    st.pyplot(fig)

usr = st.sidebar.checkbox("User Prediction")
if usr:
    st.write('We predict wether the patients have Thyroid. Negative means he/she does not have any kind of thyroid. '
             'Thyroid, here we predict are: Primary Hypothyroid, Secondary Hypothyroid, Compensated Hypothyroid.')
    mod = st.selectbox('Select the model you want to use for the prediction:.',
                       ('Select', 'SVM', 'KNN', 'Decision Tree', 'Random Forest'))
    if mod != 'Select':
        # 'age', 'sex', 'sick', 'pregnant', 'thyroid_surgery', 'goitre', 'tumor',
        # 'T3', 'TT4', 'T4U', 'FTI'],
        st.write("Please enter all the input:")
        age = st.slider('Age of the patient', 0, 100, 40)
        sex = st.radio('Select Sex', ('Male', 'Female'))
        if sex == 'Male':
            sex_n = 1
        else:
            sex_n = 0
        sick = st.radio('If the person is sick', ('Yes', 'No'))
        if sick == 'Yes':
            sick = 1
        else:
            sick = 0
        if sex == 'Female':
            preg = st.radio('If the patient is pregnant', ('Yes', 'No'))
            if preg == 'Yes':
                preg = 1
            else:
                preg = 0
        sur = st.radio('If patient has undergone surgery', ('Yes', 'No'))
        if sur == 'Yes':
            sur = 1
        else:
            sur = 0
        goi = st.radio('If the patient has goitre', ('Yes', 'No'))
        if goi == 'Yes':
            goi = 1
        else:
            goi = 0
        tum = st.radio('If the patient has tumor', ('Yes', 'No'))
        if tum == 'Yes':
            tum = 1
        else:
            tum = 0
        T3 = st.number_input('Enter the level of T3 (0 to 11)')
        TT4 = st.number_input('Enter the level of TT4 (2 to 430)')
        T4U = st.number_input('Enter the level of T4U (0.25 to 2.12)')
        FTI = st.number_input('Enter the level of FTI (2 to 395)')

        pred = st.button("Predict")
        if pred:
            if mod == 'Random Forest':
                classifier_forest = RandomForestClassifier(criterion='entropy')
                classifier_forest.fit(X_train, y_train)
                y_pred = classifier_forest.predict(X_test)
                filename = 'thyroid_forest.pkl'
                pickle.dump(classifier_forest, open(filename, 'wb'))
                model = open('thyroid_forest.pkl', 'rb')
                forest = pickle.load(model)
                if sex == 'Female':
                    st.write((forest.predict([[age, sex_n, sick, preg, sur, goi, tum, T3, TT4, T4U, FTI]])))
                if sex == 'Male':
                    st.write((forest.predict([[age, sex_n, sick, 0, sur, goi, tum, T3, TT4, T4U, FTI]])))



            #Decision Tree
            if mod == 'Decision Tree':
                classifier_tree = DecisionTreeClassifier(criterion='entropy')
                classifier_tree.fit(X_train, y_train)
                y_pred = classifier_tree.predict(X_test)
                filename = 'thyroid_tree.pkl'
                pickle.dump(classifier_tree, open(filename, 'wb'))
                model = open('thyroid_tree.pkl', 'rb')
                forest = pickle.load(model)
                if sex == 'Female':
                    st.write((classifier_tree.predict([[age, sex_n, sick, preg, sur, goi, tum, T3, TT4, T4U, FTI]])))
                if sex == 'Male':
                    st.write((classifier_tree.predict([[age, sex_n, sick, 0, sur, goi, tum, T3, TT4, T4U, FTI]])))

            #KNN
            if mod == 'KNN':
                classifier_knn = KNeighborsClassifier(metric='minkowski', p=2)
                classifier_knn.fit(X_train, y_train)
                y_pred = classifier_knn.predict(X_test)
                filename = 'thyroid_knn.pkl'
                pickle.dump(classifier_knn, open(filename, 'wb'))
                model = open('thyroid_knn.pkl', 'rb')
                forest = pickle.load(model)
                if sex == 'Female':
                    st.write((classifier_knn.predict([[age, sex_n, sick, preg, sur, goi, tum, T3, TT4, T4U, FTI]])))
                if sex == 'Male':
                    st.write((classifier_knn.predict([[age, sex_n, sick, 0, sur, goi, tum, T3, TT4, T4U, FTI]])))


            #SVM
            if mod == 'SVM':
                classifier_svm = SVC(kernel='rbf', random_state=0)
                classifier_svm.fit(X_train, y_train)
                y_pred = classifier_svm.predict(X_test)
                filename = 'thyroid_svm.pkl'
                pickle.dump(classifier_svm, open(filename, 'wb'))
                model = open('thyroid_svm.pkl', 'rb')
                forest = pickle.load(model)
                if sex == 'Female':
                    st.write((classifier_svm.predict([[age, sex_n, sick, preg, sur, goi, tum, T3, TT4, T4U, FTI]])))
                if sex == 'Male':
                    st.write((classifier_svm.predict([[age, sex_n, sick, 0, sur, goi, tum, T3, TT4, T4U, FTI]])))