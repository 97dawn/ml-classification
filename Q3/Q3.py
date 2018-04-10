"""
CSE353 HW2
Doeun Kim
doeun.kim@stonybrook.edu
"""
from sklearn import svm
import numpy as np
import pandas as pd


def one_or_two_or_three(x1, x2, x3, x4,x5, model1, model2):
    is_one = model1.predict([[x1, x2, x3, x4, x5]])
    is_two = model2.predict([[x1, x2, x3, x4, x5]])
    if is_one == 1:
        return 1
    else:
        if is_two == 1:
            return 2
        else:
            return 3

def main():
    avg_acc_1, avg_acc_2, avg_acc_3 = 0, 0, 0
    # train and test 10 times
    for test in range(10):
        # Accumulate the data sets into one data frame
        data1 = pd.read_csv('data1.txt', sep="\t", header=None)
        data2 = pd.read_csv('data2.txt', sep="\t", header=None)
        data3 = pd.read_csv('data3.txt', sep="\t", header=None)
        data = (data1, data2, data3)
        concatenated_df = pd.concat(data, ignore_index=True)

        # Shuffle the data frame
        concatenated_df = concatenated_df.sample(frac=1).reset_index(drop=True)

        # Split the data frame to training data frame and testing data frame
        training_df = concatenated_df.iloc[:int(len(concatenated_df) * 0.8), ]
        testing_df = concatenated_df.iloc[int(len(concatenated_df) * 0.8):, ]
        xs = training_df[[0,1,2,3,4]].as_matrix()
        y1 = np.where(training_df[5] == 1, 1, 0)  # If the label is class 1, return 1. Otherwise, return 0
        y2 = np.where(training_df[5] == 2, 1, 0)  # If the label is class 2, return 1. Otherwise, return 0

        # Model SVM
        C = 1.0
        model1 = svm.SVC(kernel='linear', C=C)
        model1.fit(xs, y1)
        model2 = svm.SVC(kernel='linear', C=C)
        model2.fit(xs, y2)

        num_of_1, num_of_2, num_of_3, pred_1, pred_2, pred_3 = 0, 0, 0, 0, 0, 0
        for i in range(len(testing_df)):
            pred = one_or_two_or_three(testing_df.iloc[i, 0], testing_df.iloc[i, 1], testing_df.iloc[i, 2], testing_df.iloc[i, 3], testing_df.iloc[i, 4], model1, model2)
            if testing_df.iloc[i, 5] == 1:
                num_of_1 += 1
                if pred == 1:
                    pred_1 += 1
            elif testing_df.iloc[i, 5] == 2:
                num_of_2 += 1
                if pred == 2:
                    pred_2 += 1
            elif testing_df.iloc[i, 5] == 3:
                num_of_3 += 1
                if pred == 3:
                    pred_3 += 1
        print("Test # ",test+1)
        print("Accuracy of Class 1: {:03.2f}\nAccuracy of Class 2: {:03.2f}\nAccuracy of Class 3: {:03.2f}\n\n".format((pred_1 / num_of_1),(pred_2 / num_of_2),(pred_3 / num_of_3)))
        avg_acc_1 += pred_1 / num_of_1
        avg_acc_2 += pred_2 / num_of_2
        avg_acc_3 += pred_3 / num_of_3
    print("Average Accuracy of Class 1: {:03.2f}\nAverage Accuracy of Class 2: {:03.2f}\nAverage Accuracy of Class 3: {:03.2f}".format(avg_acc_1/10, avg_acc_2/10, avg_acc_3/10))

if __name__ == '__main__':
    main()