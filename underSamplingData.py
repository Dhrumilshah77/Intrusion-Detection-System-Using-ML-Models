from imblearn.under_sampling import RandomUnderSampler

class underSamplingData():
    def under_sample_random_sampler(train):

        print("[INFO] Under Sampling the target variable using Random Sampler")

        (X_train, y_train) = train

        majority_count = y_train.value_counts()[0]

        # Identify the total count of all other classes
        other_classes_total = y_train.value_counts().sum() - majority_count

        # Perform Random Undersampling to balance the dataset
        undersampler = RandomUnderSampler(sampling_strategy={0: other_classes_total}, random_state=42)
        X_train, y_train = undersampler.fit_resample(X_train, y_train)

        return (X_train, y_train)