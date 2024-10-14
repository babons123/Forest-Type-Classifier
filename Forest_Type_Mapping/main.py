import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt                 # Importazione librerie generali
import seaborn as sns
import argparse
from utils import str2bool, print_info

from sklearn.linear_model import LogisticRegression       # Importazione modelli
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.ensemble import StackingClassifier

from sklearn.metrics import accuracy_score                # Importazione metriche
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import confusion_matrix

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def read_args():              # Funzione per importazione dataset
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--dataset_path_train', type=str,
                        default='training.csv',
                        help='Path to the file containing the training set.')
    parser.add_argument('--dataset_path_test', type=str,
                        default='testing.csv',
                        help='Path to the file containing the test set.')
    parser.add_argument("--verbose", type=str2bool, default=True)
    parser.add_argument("--cv", type=int, default=5)
    args = parser.parse_args()
    return args


def exploratory_data_analysis(df: pd.DataFrame):

    df['class'], _ = pd.factorize(df['class'], sort=True)   # Feature 'class' da categorica a discreta

    # Elimino features (N.B. non implementato)
    # df = df.drop(['pred_minus_obs_H_b1', 'pred_minus_obs_H_b2', 'pred_minus_obs_H_b3', 'pred_minus_obs_H_b4', 'pred_minus_obs_H_b5', 'pred_minus_obs_H_b6', 'pred_minus_obs_H_b7', 'pred_minus_obs_H_b8', 'pred_minus_obs_H_b9', 'pred_minus_obs_S_b1', 'pred_minus_obs_S_b2', 'pred_minus_obs_S_b3', 'pred_minus_obs_S_b4', 'pred_minus_obs_S_b5', 'pred_minus_obs_S_b6', 'pred_minus_obs_S_b7', 'pred_minus_obs_S_b8', 'pred_minus_obs_S_b9'], axis=1)

    # Definisco variabile target
    y = df['class'].values
    y, _ = pd.factorize(y, sort=True)
    X = df.drop(['class'], axis=1).values

    return X, y


def count_class_s(df: pd.DataFrame):
    class_s_count = df[df['class'] == 3].shape[0]
    print(f"Il numero di volte in cui compare la classe 's' (Sugi) nel dataset è: {class_s_count}")


def count_class_h(df: pd.DataFrame):
    class_h_count = df[df['class'] == 1].shape[0]
    print(f"Il numero di volte in cui compare la classe 'h' (Hinoki) nel dataset è: {class_h_count}")


def count_class_d(df: pd.DataFrame):
    class_d_count = df[df['class'] == 0].shape[0]
    print(f"Il numero di volte in cui compare la classe 'd' (Mixed decidual) nel dataset è: {class_d_count}")


def count_class_o(df: pd.DataFrame):
    class_o_count = df[df['class'] == 2].shape[0]
    print(f"Il numero di volte in cui compare la classe 'o' (Other) nel dataset è: {class_o_count}")


def create_models():

    models = [
        KNeighborsClassifier(weights='distance'),
        LogisticRegression(multi_class='multinomial', solver='saga', class_weight='balanced'),
        SVC(class_weight='balanced'),
        DecisionTreeClassifier(class_weight='balanced')
    ]

    models_names = ['K-NN', 'Softmax Regression', 'SVM', 'Decision Tree']

    models_hparametes = [
        {'n_neighbors': list(range(1, 10, 2))}, # KNN
        {'penalty': ['l1', 'l2'], 'C': [1e-5, 4e-5, 1e-4, 4e-4, 1]}, # Softmax Regression
        {'C': [1e-4, 1e-2, 1, 1e1, 1e2], 'gamma': [0.001, 0.0001], 'kernel': ['linear', 'rbf']}, # SVM
        {'criterion': ['gini', 'entropy']}  # Decision Tree
    ]

    return models, models_names, models_hparametes


def create_ensemble_model(X_train, y_train, models, models_names, models_hparametes):

    chosen_hparameters = []
    estimators = []

    for model, model_name, hparameters in zip(models, models_names, models_hparametes):
        print('\n', model_name)
        clf = GridSearchCV(estimator=model, param_grid=hparameters, scoring='accuracy', cv=5)
        # clf = GridSearchCV(estimator=model, param_grid=hparameters, scoring='f1_weighted', cv=5)
        clf.fit(X_train, y_train)
        chosen_hparameters.append(clf.best_params_)
        estimators.append((model_name, clf))
        print('Miglior Configurazione Iperparametri : ', clf.best_params_)
        print('Accuracy :', clf.best_score_)
        # print('F1-Score :', clf.best_score_)

    print('\n############ Ensemble ############\n')

    # estimators.pop(1)

    clf_stack = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())
    return clf_stack


if __name__ == '__main__':

    args = read_args()                                                # Lettura dataset

    df_train = pd.read_csv(args.dataset_path_train, delimiter=',')    # Caricamento Dataframe train
    df_test  = pd.read_csv(args.dataset_path_test , delimiter=',')    # Caricamento Dataframe test

    # print_info(df_train)
    # print_info(df_test)

    X_train, y_train = exploratory_data_analysis(df_train)            # EDA train
    X_test , y_test  = exploratory_data_analysis(df_test)             # EDA test

    # count_class_s(df_train)              # conteggio classi (N.B. inserito nella relazione)
    # count_class_h(df_train)
    # count_class_d(df_train)
    # count_class_o(df_train)

    '''
        # Visualizzazione 1: Distribuzione delle classi
        plt.figure(figsize=(6, 4))
        sns.countplot(x='class', data=df_train)
        plt.title('Distribuzione delle classi nel dataset di addestramento')
        plt.xlabel('Classe')
        plt.ylabel('Conteggio')
        plt.show()
    '''

    # print(X_train)
    # print(y_train)
    # print(X_test)
    # print(y_test)

    scaler  = StandardScaler().fit(X_train)                           # Preparazione scaler su df training
    X_train = scaler.transform(X_train)                               # Scalamento dati training
    X_test  = scaler.transform(X_test)                                # Scalamento dati testing

    # print(scaler)
    # print(X_train)
    # print(X_test)

    models, models_names, models_hparametes = create_models()

    clf_ensemble = create_ensemble_model(X_train, y_train,
                                         models, models_names, models_hparametes)

    scores = cross_validate(clf_ensemble, X_train, y_train,
                            cv=args.cv, scoring=('f1_weighted', 'accuracy'))

    print('The cross-validated weighted F1-score of the Stacking Ensemble is ',
          np.mean(scores['test_f1_weighted']))
    print('The cross-validated Accuracy of the Stacking Ensemble is ',
          np.mean(scores['test_accuracy']))

    final_model = clf_ensemble
    # final_model = KNeighborsClassifier(n_neighbors=7, weights='distance')
    # final_model = LogisticRegression(multi_class='multinomial', solver='saga', class_weight='balanced')
    # final_model = SVC(class_weight='balanced')
    # final_model = DecisionTreeClassifier(class_weight='balanced')

    final_model.fit(X_train, y_train)

    y_pred = final_model.predict(X_test)

    print('/-------------------------------------------------------------------------------------------------------- /')
    print('Final Testing RESULTS')
    # print('Final Model : ', final_model)
    print('/-------------------------------------------------------------------------------------------------------- /')
    print('Accuracy : ', accuracy_score(y_test, y_pred))
    print('Precision : ', precision_score(y_test, y_pred, average='weighted'))
    print('Recall : ', recall_score(y_test, y_pred, average='weighted'))
    print('F1-Score : ', f1_score(y_test, y_pred, average='weighted'))

    # Visualizzazione confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    labels = ['d', 'h', 'o', 's']
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, cmap='Greens', fmt='g', xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
