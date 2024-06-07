import pandas as pd
import numpy as np
from itertools import combinations
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class loadData():
    def load_preprocessdata(dir, filenames, binaryEncoding):

        # Loading data from different csv's and concating
        print("[INFO] Loading data...")
        combined_df = pd.DataFrame()
        dfs = []

        for i, filename in enumerate(filenames):
            print(f"[INFO] Reading file {i+1} named {filename}", end='\r')
            df = pd.read_csv(dir+"/Data/MachineLearningCVE/"+filename)
            dfs.append(df)

        combined_df = pd.concat(dfs, 
                                ignore_index=True)
        
        # Preprocessing data
        print()
        # Removing null values
        print("[INFO] Dropping Null Values")
        combined_df.dropna(inplace=True)
        
        # Removing duplicates
        print("[INFO] Dropping Duplicates")
        combined_df.drop_duplicates(inplace=True)

        # Removing infinte values
        print("[INFO] Dropping Infinite Values")
        combined_df.replace([np.inf, -np.inf], 
                   np.nan,
                   inplace=True)
        combined_df.dropna(inplace=True)

        # Dropping columns with zero variance
        print("[INFO] Dropping Zero Variance Columns")
        zero_variance_cols = []
        for col in combined_df.columns:
            if combined_df[col].nunique() == 1:
                zero_variance_cols.append(col)
        combined_df.drop(columns=zero_variance_cols, 
                         inplace=True)
        
        # Cleaning column names
        print("[INFO] Cleaning column names")
        combined_df.columns = combined_df.columns.str.strip()
        combined_df.columns = combined_df.columns.str.lower()
        combined_df.columns = combined_df.columns.str.replace(' ', '_')
        combined_df.columns = combined_df.columns.str.replace('(', '')
        combined_df.columns = combined_df.columns.str.replace(')', '')
        combined_df[['flow_packets/s', 'flow_bytes/s']] = combined_df[['flow_packets/s', 'flow_bytes/s']].apply(pd.to_numeric) 

        # Dropping columns with identical values
        print("[INFO] Dropping columns with identical values")
        column_pairs = [(i, j) for i, j in combinations(combined_df.columns, 2) if combined_df[i].equals(combined_df[j])]
        ide_cols = [col_pair[1] for col_pair in column_pairs]
        combined_df.drop(columns=ide_cols, axis=1, inplace=True)

        # Manually encoding the target variable
        if binaryEncoding == 0:
            print("[INFO] Encoding target variables")
            combined_df['label'] = combined_df['label'].replace('BENIGN', 0)

            combined_df['label'] = combined_df['label'].replace('DoS Hulk', 1)
            combined_df['label'] = combined_df['label'].replace('DoS GoldenEye', 1)
            combined_df['label'] = combined_df['label'].replace('DoS slowloris', 1)
            combined_df['label'] = combined_df['label'].replace('DoS Slowhttptest', 1)

            combined_df['label'] = combined_df['label'].replace('DDoS', 2)

            combined_df['label'] = combined_df['label'].replace('PortScan', 3)

            combined_df['label'] = combined_df['label'].replace('FTP-Patator', 4)
            combined_df['label'] = combined_df['label'].replace('SSH-Patator', 4)
            combined_df['label'] = combined_df['label'].replace('Web Attack � Brute Force', 4)

            combined_df['label'] = combined_df['label'].replace('Web Attack � XSS', 5)
            combined_df['label'] = combined_df['label'].replace('Web Attack � Sql Injection', 5)

            combined_df['label'] = combined_df['label'].replace('Bot', 6)

            combined_df['label'] = combined_df['label'].replace('Infiltration', 7)

            combined_df['label'] = combined_df['label'].replace('Heartbleed', 8)

        else:
            print("[INFO] Binary Encoding target variables")
            combined_df['label'] = combined_df.label.apply(lambda x: 0 if x == "BENIGN" else 1)

        # Spliting data
        print('[INFO] Train Val Test Split (70-15-15)')
        X = combined_df.drop('label', axis=1)
        y = combined_df['label']
        
        X_train, X_remain, y_train, y_remain = train_test_split(X, y, 
                                                                random_state=42,
                                                                test_size=0.3)
        
        X_test, X_val, y_test, y_val = train_test_split(X_remain, y_remain, 
                                                          random_state=42,
                                                          test_size=0.5)

        # Applying StandardScaler to the data
        print("[INFO] Transforming X variables using StandardScaler")
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_val = scaler.transform(X_val)
        X_test = scaler.transform(X_test)

        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

