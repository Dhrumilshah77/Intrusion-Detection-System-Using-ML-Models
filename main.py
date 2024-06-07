import os
from loadData import loadData
from underSamplingData import underSamplingData
from BuildModel import BuildModel
from performPCA import performPCA
from xgboost import XGBClassifier

from warnings import filterwarnings

filterwarnings('ignore')



if __name__== '__main__':

    base_dir = os.getcwd()
    
    # Specify the file names
    file_names = ["Monday-WorkingHours.pcap_ISCX.csv", 
                  "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
                  "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
                  "Tuesday-WorkingHours.pcap_ISCX.csv",
                  "Wednesday-workingHours.pcap_ISCX.csv",
                  "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
                  "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
                  "Friday-WorkingHours-Morning.pcap_ISCX.csv"]
    
    binary_encodings = [0, 1]
    for binary_encoding in binary_encodings:
        train, val, test = loadData.load_preprocessdata(base_dir, file_names, binary_encoding)

        train, val, test = performPCA.PCA(train, val, test)
        
        # Undersampling data
        train_under_random = underSamplingData.under_sample_random_sampler(train)
        
        # Creating model
        model = BuildModel.create_model(XGBClassifier(), train_under_random)

        # Evaluation Model
        if binary_encoding == 0:
            print("[RESULTS] Decision Tree Model Evaluation for multiclass Classification:")
            
        else:
            print("[RESULTS] Decision Tree Model Evaluation for binary Classification:")
        
        train_precison = BuildModel.print_evaluation_metrics(model, train_under_random, "Train")
        val_precision = BuildModel.print_evaluation_metrics(model, val, "Validation")
        test_precision = BuildModel.print_evaluation_metrics(model, test, "Test")


        
        