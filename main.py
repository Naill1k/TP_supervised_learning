import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, ConfusionMatrixDisplay, confusion_matrix
from sklearn.inspection import permutation_importance
from lime.lime_tabular import LimeTabularExplainer
import time
import os


def _pobp_to_continent(code):
    c = int(code)
    
    if c in list(range(100, 158)) + [160] + list(range(162, 200)):
        return 1 # Europe
    
    if c in [158, 159, 161] + list(range(200, 300)):
        return 2 # Asia
    
    if c in list(range(61, 100)) + [300, 301, 302] + list(range(304, 310)):
        return 3 # North America (no US)
    
    if c in list(range(1, 60)):
        return 4 # United States
    
    if c in [303] + list(range(310, 400)):
        return 5 # Latin America
    
    if c in list(range(400, 500)):
        return 6 # Africa
    
    if c in list(range(500, 554)) + [60]:
        return 7 # Oceania
    
    return 8 # Other


def _relp_simplify(code):
    c = int(code)
    
    if c in [2, 3, 4, 7, 14]:
        return 2  # Children
    
    return code


def pre_process_features(X):
    X.drop(columns=['OCCP'], errors='ignore', inplace=True) # Remove OCCP feature

    X['POBP'] = X['POBP'].apply(_pobp_to_continent)  # Merge countries into continents

    X['RELP'] = X['RELP'].apply(_relp_simplify)  # Merge children categories

    return X


def pre_process_labels(y):
    y['PINCP'] = y['PINCP'].apply(lambda x: 1 if x else 0)  # Binarize labels

    return y


def plot_simple():
    for col in df.columns:
        data = df[col].value_counts().sort_index()

        plt.bar(range(len(data)), data, width=0.8)
        plt.title(f'Histogram of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        plt.tight_layout()

        filename = f"plots/{col.replace(' ', '_')}.png"
        plt.savefig(filename)
        plt.close()
        print(f"Saved {filename}")


def plot_RELP_per_SEX():
    df_plot = df[['SEX', 'RELP']].copy()
    series_m = df_plot[df_plot['SEX'] == 1]['RELP'].value_counts().sort_index()
    series_f = df_plot[df_plot['SEX'] != 1]['RELP'].value_counts().sort_index()

    width = 0.4

    plt.bar(np.arange(len(series_m.values)) - width/2, series_m.values, width=width, label='Man', edgecolor='black')
    plt.bar(np.arange(len(series_f.values)) + width/2, series_f.values, width=width, label='Woman', edgecolor='black')

    plt.xlabel('RELP')
    plt.ylabel('Count')
    plt.title('RELP par SEX')
    plt.legend()
    plt.tight_layout()
    filename = 'plots/RELP_by_SEX.png'
    plt.savefig(filename)
    plt.close()
    print(f"Saved {filename}")


def merge_POBP():
    ranges_EU = list(range(100, 158)) + [160] + list(range(162, 200))
    ranges_AS = [158, 159, 161] + list(range(200, 300))
    ranges_NA = list(range(61, 100)) + [300, 301, 302] + list(range(304, 310))
    ranges_US = list(range(1, 60))
    ranges_LA = [303] + list(range(310, 400))
    ranges_AF = list(range(400, 500))
    ranges_OCE = list(range(500, 554)) + [60]
    ranges_Other = list(range(554, 1000))

    series_EU = df[df['POBP'].isin(ranges_EU)]
    series_AS = df[df['POBP'].isin(ranges_AS)]
    series_NA = df[df['POBP'].isin(ranges_NA)]
    series_US = df[df['POBP'].isin(ranges_US)]
    series_LA = df[df['POBP'].isin(ranges_LA)]
    series_AF = df[df['POBP'].isin(ranges_AF)]
    series_OCE = df[df['POBP'].isin(ranges_OCE)]
    series_Other = df[df['POBP'].isin(ranges_Other)]

    nb_EU_True = len(series_EU[series_EU['PINCP'] == True])

    nb_AS_True = len(series_AS[series_AS['PINCP'] == True])

    nb_NA_True = len(series_NA[series_NA['PINCP'] == True])

    nb_US_True = len(series_US[series_US['PINCP'] == True])

    nb_LA_True = len(series_LA[series_LA['PINCP'] == True])

    nb_AF_True = len(series_AF[series_AF['PINCP'] == True])

    nb_OCE_True = len(series_OCE[series_OCE['PINCP'] == True])

    nb_Other_True = len(series_Other[series_Other['PINCP'] == True])


    print("EU_True", round(100*nb_EU_True/len(series_EU), 2), "%")
    print("AS_True", round(100*nb_AS_True/len(series_AS), 2), "%")
    print("NA_True", round(100*nb_NA_True/len(series_NA), 2), "%")
    print("US_True", round(100*nb_US_True/len(series_US), 2), "%")
    print("LA_True", round(100*nb_LA_True/len(series_LA), 2), "%")
    print("AF_True", round(100*nb_AF_True/len(series_AF), 2), "%")
    print("OCE_True", round(100*nb_OCE_True/len(series_OCE), 2), "%")
    print("Other_True", round(100*nb_Other_True/len(series_Other), 2), "%")



def train_model_default(classifier, X_train, y_train, X_test, y_test):
    model = classifier

    t0 = time.perf_counter()
    model.fit(X_train, y_train.values.ravel())
    t1 = time.perf_counter()

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    matrix = pd.crosstab(y_test.values.ravel(), preds, rownames=['Actual'], colnames=['Predicted'])

    print(f"Accuracy of {classifier.__class__.__name__}: {round(100*accuracy, 2)}% (MSE: {round(mse, 4)})")
    print(f"Training time: {round(t1 - t0, 3)} seconds")
    print(matrix)
    print()
    return model


def evaluate_with_cross_validation(classifier, X, y, n_fold=5):
    res = cross_validate(classifier, X, y.values.ravel(), cv=n_fold, n_jobs=-1)

    mean_score = res['test_score'].mean()
    std_score = res['test_score'].std()
    mean_fit = res['fit_time'].mean()

    print(f"Cross Validation {n_fold}-fold pour {classifier.__class__.__name__}:")
    print(f"\taccuracy moyenne = {round(100*mean_score,2)}% (+/- {round(100*std_score,2)}%),")
    print(f"\tfit moyen = {round(mean_fit,3)}s\n")


def permutation_feature_importance(model, X_test, y_test, n_repeats=50):
    result = permutation_importance(model, X_test, y_test, n_repeats=n_repeats, n_jobs=-1)

    importance_df = pd.DataFrame({'Feature': X_test.columns, 'Importance Mean': result.importances_mean, 'Importance Std': result.importances_std})
    importance_df = importance_df.sort_values(by='Importance Mean', ascending=False)
    print("Permutation Feature Importance:")
    print(importance_df)
    print()

    importance_df = importance_df.sort_values(by='Importance Mean', ascending=False)
    _, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(importance_df))))
    ax.barh(importance_df['Feature'], importance_df['Importance Mean'],
            xerr=importance_df['Importance Std'],
            align='center', color='skyblue', ecolor='grey', capsize=3)
    
    ax.set_xlabel('Average Importance')
    ax.set_title('Permutation Feature Importance')
    plt.tight_layout()

    filename = 'plots/permutation_importance.png'
    plt.savefig(filename)
    plt.close()


def visualize_lime_explanations(model, X_train, X_test, y_train, n_instances=5, num_features=10):
    explainer = LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=y_train.values.ravel()
    )

    rng = np.random.RandomState(42)
    examples = rng.choice(range(X_test.shape[0]), size=n_instances, replace=False)

    for i in examples:
        instance = X_test.iloc[i].values
        pred = model.predict([instance])[0]
        explanation = explainer.explain_instance(instance, model.predict_proba, num_features=num_features).as_list()

        df_explanation = pd.DataFrame({'feature': [t[0] for t in explanation], 'weight': [t[1] for t in explanation]})

        _, ax = plt.subplots(figsize=(8, max(3, 0.4 * len(df_explanation))))
        colors = ['green' if v > 0 else 'red' for v in df_explanation['weight']]

        ax.barh(df_explanation['feature'], df_explanation['weight'], color=colors)
        ax.set_title(f'LIME exp. idx {i}  —  pred: {pred}')
        ax.set_xlabel('Contribution au score de la classe')
        plt.tight_layout()

        png_path = f'plots/lime/explanation_idx{i}.png'
        plt.savefig(png_path)
        plt.close()



if __name__ == "__main__":
    df_features = pd.read_csv('2-Dataset/alt_acsincome_ca_features_85.csv', sep=',', encoding='utf-8', header=0)
    df_labels = pd.read_csv('2-Dataset/alt_acsincome_ca_labels_85.csv', sep=',', encoding='utf-8', header=0)

    X_train, X_test, y_train, y_test = train_test_split(df_features, df_labels, test_size=0.2)

    df = pd.concat([df_features, df_labels], axis=1)

    print("Training dataset size :", X_train.shape[0])
    print("Test dataset size :", X_test.shape[0])

    y_train = pre_process_labels(y_train)
    y_test = pre_process_labels(y_test)

    # print("Before pre-processing:\n")

    # # evaluate_with_cross_validation(RandomForestClassifier(), X_train, y_train)
    # train_model_default(RandomForestClassifier(), X_train, y_train, X_test, y_test)

    # # evaluate_with_cross_validation(GradientBoostingClassifier(), X_train, y_train)
    # train_model_default(GradientBoostingClassifier(), X_train, y_train, X_test, y_test)

    # # evaluate_with_cross_validation(AdaBoostClassifier(), X_train, y_train)
    # train_model_default(AdaBoostClassifier(), X_train, y_train, X_test, y_test)



    X_train = pre_process_features(X_train)
    X_test = pre_process_features(X_test)


    # print("\nAfter pre-processing:\n")

    # evaluate_with_cross_validation(RandomForestClassifier(), X_train, y_train)
    # train_model_default(RandomForestClassifier(), X_train, y_train, X_test, y_test)

    # # evaluate_with_cross_validation(GradientBoostingClassifier(), X_train, y_train)
    # train_model_default(GradientBoostingClassifier(), X_train, y_train, X_test, y_test)

    # evaluate_with_cross_validation(AdaBoostClassifier(), X_train, y_train)
    model = train_model_default(AdaBoostClassifier(), X_train, y_train, X_test, y_test)

    # permutation_feature_importance(model, X_test, y_test)

    visualize_lime_explanations(model, X_train, X_test, y_train, n_instances=5, num_features=10)