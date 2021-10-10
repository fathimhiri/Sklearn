from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt


#Calculating Confusion Matrix
CM = confusion_matrix(y_true,y_pred)
#print('Confusion Matrix is : \n', CM)

# drawing confusion matrix
sns.heatmap(CM, center = True)
plt.show()
