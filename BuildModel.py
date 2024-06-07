from sklearn.metrics import classification_report

class BuildModel():
    def create_model(model, train):

        print("[INFO] Initializing Model")
        # Create the Decision Tree model
        model_obj = model

        (X_train, y_train) = train

        print("[INFO] Fitting the Model")
        # Train the model
        model_obj.fit(X_train, y_train)

        return model_obj

    def print_evaluation_metrics(model, data, set_name):
        (X, y_true) = data

        print(f"[INFO] Predicting using {set_name} data")
        y_pred = model.predict(X)
        
        report = classification_report(y_true, y_pred, output_dict=True)    
        precision = report['macro avg']['precision']
        recall = report['macro avg']['recall']
        f1 = report['macro avg']['f1-score']
        
        print(f"[RESULTS] Precision: {precision:.4f}")
        print(f"[RESULTS] Recall: {recall:.4f}")
        print(f"[RESULTS] F1 Score: {f1:.4f}")

        return precision

