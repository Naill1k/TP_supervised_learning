import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, mean_squared_error
import time


df_features = pd.read_csv('2-Dataset/alt_acsincome_ca_features_85.csv', sep=',', encoding='utf-8', header=0)

df_labels = pd.read_csv('2-Dataset/alt_acsincome_ca_labels_85.csv', sep=',', encoding='utf-8', header=0)

X_train, X_test, y_train, y_test = train_test_split(df_features, df_labels, test_size=0.2)

df = pd.concat([df_features, df_labels], axis=1)


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


def train_model(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier()

    t0 = time.perf_counter()
    model.fit(X_train, y_train.values.ravel())
    t1 = time.perf_counter()

    preds = model.predict(X_test)

    accuracy = accuracy_score(y_test, preds)
    print(f"Accuracy: {accuracy}")

    mse = mean_squared_error(y_test, preds)
    print(f"MSE: {mse}")

    print(f"Training time: {round(t1 - t0, 3)} seconds\n")


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

 

if __name__ == "__main__":
    y_train = pre_process_labels(y_train)
    y_test = pre_process_labels(y_test)

    train_model(X_train, y_train, X_test, y_test)


    X_train = pre_process_features(X_train)
    X_test = pre_process_features(X_test)

    train_model(X_train, y_train, X_test, y_test)