import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class performPCA():
    def PCA(train, val, test):
        print("[INFO] Performing PCA on the data")
        
        (X_train, y_train) = train
        (X_val, y_val) = val
        (X_test, y_test) = train

        # Define the desired explained variance threshold
        desired_variance_threshold = 0.95

        # Perform PCA to find the best number of components
        pca = PCA()
        pca.fit(X_train)

        explained_variance = pca.explained_variance_ratio_
        cumulative_explained_variance = np.cumsum(explained_variance)

        # Find the number of components that meet the desired variance threshold
        best_num_components = np.argmax(cumulative_explained_variance >= desired_variance_threshold) + 1
        print(f"[INFO] Best Number of components for PCA is {best_num_components}")

        pca = PCA(n_components=best_num_components)
        X_train_pca = pca.fit_transform(X_train)
        X_val_pca = pca.transform(X_val)
        X_test_pca = pca.transform(X_test)

        return (X_train_pca, y_train), (X_val_pca, y_val), (X_test_pca, y_test)
    