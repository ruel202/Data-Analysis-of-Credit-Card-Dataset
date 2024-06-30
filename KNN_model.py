import numpy as np
import pandas as pd
import math
from math import sqrt
import sklearn
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cross_validation import cross_val_score # type: ignore
import matplotlib as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
import random


"""Scaler"""
def scaler(df):
    scaler=StandardScaler()
    scaler.fit(df) #  .drop('train_feature',axis=1)
    scaled_features=scaler.transform(df) #  .drop('train_feature',axis=1) 
    df_feat= pd.DataFrame(scaled_features, columns=df.columns[:])
    return df_feat

# accuracy_score : the accuracy based on the inputs y_pred and y_pred, accuracy of the predicted results and the test model
# cv scores : accuracy for each folds and the average of them (      model1 model2 model3
#                                                              fold1 test     train   train
#                                                              fold2 train    test    train
#                                                              fold3 train    train   test)   #k-fold cross validation with 3-folds, we use this metric to evaluate the training process for diffenrent ml models, each return scores is the accuracy score of each training testing fold
#




"""Verification functions"""
class ver ():
    def __init__(self,y_test,y_pred):
        self.y_test= y_test
        self.y_pred=y_pred
    def acc_score(self):
        accuracy= accuracy_score(self.y_test,self.y_pred)
        return accuracy
        print("Accuracy scores :", accuracy)
    def error_rate(model, X_train,X_test,y_train,y_test):
        error_rate=[]
        sqrt_k = int(math.sqrt(len(y_test)))
        for i in range (1,sqrt_k*3):
            model.fit(X_train,y_train)
            y_pred= model.predict(X_test)
            error_rate.append(np.mean(y_pred!=y_test))
        return error_rate
    



"""KNN """
class KNN():
    def __init__(self,k,X_train,X_test,y_train,y_test,knn):
        self.k=k
        self.X_train=X_train
        self.X_test=X_test
        self.y_train=y_train
        self.y_test=y_test
        self.sqrt_k = int(math.sqrt(len(y_test)))
        self.k_range= range (1,self.sqrt_k*3)
        self.knn=KNeighborsClassifier(n_neighbors=self.k,algorithm='auto', leaf_size=30, metric='minkowski',metric_params=None, n_jobs=None , n_neighbors=k, p=2,weights='uniform')# p=2=euclide ; p=1=manhattan
        self.weight_option= ['uniform','distance']

    def train(self):
        self.knn.fit(self.X_train,self.y_train)
        y_pred= self.knn.predict(self.X_test)
        return np.array(y_pred)
        print("y_pred is : ", y_pred)
    def max_accuracy(self):
        k_acc_scores=[]
        for i in self.k_range:
            knn = KNeighborsClassifier(n_neighbors=i).fit(self.X_train,self.y_train)
            scores = knn.predict(self.X_test)
            k_acc_scores.append(metrics.accuracy_score(self.y_test,scores))
        return k_acc_scores
        print("Maximum accuracy:-",max(acc),"at K =",acc.index(max(acc)))
    def cros_val_score (self):#ross validation
        k_acc_scores=[]
        for i in self.k_range:
            scores= cross_val_score(self.knn, self.X_test, self.y_test, cv=5, scoring='accuracy')# cv= constant, None
            k_acc_scores.append(scores.mean())
        return k_acc_scores
    def KNN_Best_Params(self): #Gridsearchcv
        weigh_options= ['uniform','distance']
        p = [float('inf')]# 
        # Created Grid model
        param_grid = dict(n_neighbors =[random.randint(1,self.k_range*3)], weights = weigh_options , p = p) # testing the different number of n_neighbors
        grid = GridSearchCV(self.knn, param_grid, cv=10, scoring='accuracy')
        grid.fit(self.X_train,self.y_train)
        print('Best training score :', grid.best_score_, 'with parameters',grid.best_params_)
        knn_grid=KNeighborsClassifier(**grid.best_params_)
        knn_grid.fit(self.X_train,self.y_train)
        y_pred_test_grid = knn_grid.predict(self.X_test)
        y_pred_train_grid = knn_grid.predict(self.X_train)
        # Confusion Matrix
        cm_test_grid = confusion_matrix(self.y_test, y_pred_test_grid)
        cm_train_grid =confusion_matrix(self.y_train, y_pred_train_grid)
        # Accuracy Score
        acc_test_grid = accuracy_score(self.y_test, y_pred_test_grid)
        acc_train_grid = accuracy_score(self.y_train, y_pred_train_grid)
        print("Test Score: {}, Train Score: {}".format(acc_test_grid, acc_train_grid))
        print("CM Test: ",cm_test_grid)
        print("CM Train: ",cm_train_grid)
        return

    


'''Precision - Precision is the ratio of correctly predicted positive observations to the total predicted positive observations. The question that this metric answer is of all passengers that labeled as survived, how many actually survived? High precision relates to the low false positive rate. We have got 0.788 precision which is pretty good.

Precision = TP/TP+FP

Recall (Sensitivity) - Recall is the ratio of correctly predicted positive observations to the all observations in actual class - yes. The question recall answers is: Of all the passengers that truly survived, how many did we label? A recall greater than 0.5 is good.

Recall = TP/TP+FN

F1 score - F1 Score is the weighted average of Precision and Recall. Therefore, this score takes both false positives and false negatives into account. Intuitively it is not as easy to understand as accuracy, but F1 is usually more useful than accuracy, especially if you have an uneven class distribution. Accuracy works best if false positives and false negatives have similar cost. If the cost of false positives and false negatives are very different, it’s better to look at both Precision and Recall.

F1 Score = 2(Recall Precision) / (Recall + Precision)'''

    


'''GridSearchCV is a scikit-learn function that performs hyperparameter tuning by training and evaluating a machine learning model using different combinations of hyperparameters

To use GridSearchCV, you need to specify the following:

1.The hyperparameters to be tuned: This includes specifying a range of values for each hyperparameter.

2.The machine learning model: This includes the type of model you want to use and its parameters.

3.The performance metric to be used: This is the metric that will be used to evaluate the performance of the different hyperparameter combinations.'''
  



'''Principal component analysis, or PCA, is a dimensionality-reduction method that is often used to reduce the dimensionality of large data sets, by transforming a large set of variables into a smaller one that still contains most of the information in the large set.

Reducing the number of variables of a data set naturally comes at the expense of accuracy, but the trick in dimensionality reduction is to trade a little accuracy for simplicity. 

Because smaller data sets are easier to explore and visualize and make analyzing data much easier and faster for machine learning algorithms without extraneous variables to process.

So, to sum up, the idea of PCA is simple — reduce the number of variables of a data set, while preserving as much information as possible.'''
    

    
        
    

        

    
    
