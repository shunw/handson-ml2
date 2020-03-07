from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import joblib

def conf_matrix_accuracy(np_matrix): 
    '''
    method: sum up the diagonal data, then divide by the sum of total
    output: the accuracy of the confusion matrix
    '''
    row_n, col_n = np_matrix.shape
    diag_sum = 0
    for i in range(col_n): 
        diag_sum += np_matrix[i, i]
    # print (diag_sum)
    # print (np_matrix.sum())
    return diag_sum/np_matrix.sum()

y_actu = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
y_pred = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]

y_pred = [0, 2, 1, 3]
y_actu = [0, 1, 2, 3]
a = confusion_matrix(y_actu, y_pred)

print (conf_matrix_accuracy(a))
print (accuracy_score(y_actu, y_pred))
