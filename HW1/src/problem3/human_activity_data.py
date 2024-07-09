import numpy as np

from problem3 import SimpleClassifier

"""
training model aka decision rule
training is just finding likelihood pdfs and prior probs for each class 
"""
def run_activity_inferences(cf2: SimpleClassifier):
    # Define the function to read the file and store the data in a 2D array

    features = []
    with open("HW1/human_activity_data", 'r') as file:
        for line in file:
            row = list(map(float, line.split()))
            features.append(row)

    count = 0
    for i in range(len(features)):
        if (len(features[i])) != 561:
            count+=1
    print(count)

    print(len(features[0]))
    print(len(features))

    labels = []
    with open("HW1/human_activity_data", 'r') as file:
        for line in file:
            row = list(map(int, line.split()))
            labels.append(row)

    print(len(labels))


    features = np.array(features)
    labels = np.array(labels).T[0]
    loss_matrix = np.logical_not(np.identity(len(features))).astype(int)

    print(labels)

    conf_mat = cf2.train_classifier(features, labels, loss_matrix)
    print(f"results: \n {conf_mat}")

    pError = 0
    pps = cf2.obs_prior_probs()
    for col in range(3, len(conf_mat[0])):
        for row in range(3, len(conf_mat)):
            if row is not col:
                pError += conf_mat[row][col] * pps[col]

    print("p of error: " + str(pError))

'''
for first 1100 features + labels
results: 
 [[0.     0.     0.     0.     0.     0.     0.    ]
 [0.     1.     0.     0.0414 0.     0.     0.    ]
 [0.     0.     1.     0.0138 0.0058 0.     0.    ]
 [0.     0.     0.     0.9448 0.     0.     0.    ]
 [0.     0.     0.     0.     0.8547 0.     0.    ]
 [0.     0.     0.     0.     0.1395 1.     0.    ]
 [0.     0.     0.     0.     0.     0.     1.    ]]
p of error: 0.021818181818181813
'''

'''
all features + labels in X_train y_train
results: 
 [[0.     0.     0.     0.     0.     0.     0.    ]
 [0.     0.9715 0.0354 0.0923 0.     0.     0.    ]
 [0.     0.0261 0.9627 0.1298 0.0008 0.     0.0007]
 [0.     0.0024 0.0019 0.7779 0.     0.     0.    ]
 [0.     0.     0.     0.     0.6633 0.0146 0.    ]
 [0.     0.     0.     0.     0.3305 0.9854 0.    ]
 [0.     0.     0.     0.     0.0054 0.     0.9993]]
p of error: 0.061479869423285836
'''
