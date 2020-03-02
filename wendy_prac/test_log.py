from sklearn.metrics import confusion_matrix
import joblib
y_actu = [2, 0, 2, 2, 0, 1, 1, 2, 2, 0, 1, 2]
y_pred = [0, 0, 2, 1, 0, 2, 1, 0, 2, 0, 2, 2]
# a = confusion_matrix(y_actu, y_pred)
# joblib.dump(a, 'a.pkl')
a = joblib.load('a.pkl')
print (a)