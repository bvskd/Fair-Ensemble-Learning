import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
import statistics as stat
import sklearn.metrics as metrics
from sklearn.ensemble import AdaBoostClassifier
from scipy.spatial.distance import cdist

data = np.genfromtxt('crimerate.csv', delimiter = ',')

sample = data[:,0:-1]
label = data[:,-1]
label[np.where(label<=0.5)] = 0
label[label!=0] = 1



sample_train, sample_test, label_train, label_test = train_test_split(sample, label, test_size = 0.2)

k = 5
threshold = 0.5
wt = 0.9
w = np.ones(len(label_train))
alpha = []
epsilon = []
err_mat = np.zeros(len(label_train))
sacol = 2
#def mad(data, axis=1):
#    return np.mean(np.absolute(data - np.mean(data, axis)), axis)

def weighUnfairInstances(strain, ltrain, trlpred):
    dist_mat = cdist(strain, strain)
    for i in range(len(strain)):
        dist = dist_mat[i].tolist()
        del dist[i]
#        dist = [0 for x in range(len(strain))]
#        for j in range(len(strain)):
#            dist[j] = np.linalg.norm(strain[i] - strain[j])
    
        lst = np.asarray(ltrain)[np.argsort(dist)[:k].tolist()]
        #maj = np.where(lst == trlpred)
        #if(np.mean(lst) > trlpred[i]-threshold and np.mean(lst) < trlpred[i]+threshold):#for continuous labels
        if(stat.mode(lst) == trlpred[i]):#for discrete labels #trlpred[i] == ltrain[i] and stat.mode(lst) == trlpred[i]
            err_mat[i] = 0
        else:
            err_mat[i] = 1

#Single normal model --- might have to change it to single fair model
model1 = linear_model.LogisticRegression(class_weight='balanced', solver = 'liblinear')
model1.fit(sample_train, label_train, sample_weight = w)
label_pred1 = model1.predict(sample_test)
mino_label = label_pred1[np.where(sample_test[:,sacol]>=threshold)]
majo_label = label_pred1[np.where(sample_test[:,sacol]<threshold)]
mino_label_pred_eo = label_pred1[((sample_test[:,sacol]>=threshold) & (label_test == 1))]
majo_label_pred_eo = label_pred1[((sample_test[:,sacol]<threshold) & (label_test == 1))]
eo_boot_ratio1 = (len(mino_label_pred_eo[np.where(mino_label_pred_eo == 1)])/len(mino_label_pred_eo))
eo_boot_ratio2 = (len(majo_label_pred_eo[np.where(majo_label_pred_eo == 1)])/len(majo_label_pred_eo))
meandiff_one_eo = np.abs(np.mean(eo_boot_ratio1) - np.mean(eo_boot_ratio2))
meandiff_one = np.abs(np.mean(mino_label) - np.mean(majo_label))
err_one_test = metrics.mean_squared_error(label_test, label_pred1)

fair_flag = 0
iteration = 0
prev_meandiff = 0

test_pred_iter = []
train_pred_iter = []

while fair_flag == 0:
    model = linear_model.LogisticRegression(class_weight='balanced', solver = 'liblinear')
    model.fit(sample_train, label_train, sample_weight = w)
    test_pred_iter.append(model.predict(sample_test))
    train_pred_iter.append(model.predict(sample_train))
    weighUnfairInstances(sample_train, label_train, train_pred_iter[iteration])#Updates error matrix
    epsilon.append(np.sum(w * err_mat)/np.sum(w))
#    print(epsilon)
    alpha.append(np.log((1-epsilon[iteration])/epsilon[iteration]))
    w = (w * np.exp(alpha[iteration] * err_mat))
    
    
    mino_label = test_pred_iter[iteration][np.where(sample_test[:,sacol]>=threshold)]
    majo_label = test_pred_iter[iteration][np.where(sample_test[:,sacol]<threshold)]
    meandiff = np.abs(np.mean(mino_label) - np.mean(majo_label))
    
    iteration = iteration + 1
    
    print("iteration: " + repr(iteration) + " Mean Diff: " + repr(meandiff))
    
    if(meandiff == prev_meandiff or meandiff < 0.2):# 
        fair_flag = 1
    else:
        wt = wt - 0.1
    
    prev_meandiff = meandiff
    
label_pred = np.transpose(np.sum(np.matmul(np.diag(alpha), np.asmatrix(test_pred_iter)), axis = 0))/iteration
mino_label = label_pred[np.where(sample_test[:,sacol]>=threshold)]
majo_label = label_pred[np.where(sample_test[:,sacol]<threshold)]
mino_label_pred_eo = label_pred[((sample_test[:,sacol]>=threshold) & (label_test == 1))]
majo_label_pred_eo = label_pred[((sample_test[:,sacol]<threshold) & (label_test == 1))]
eo_boot_ratio1 = (len(mino_label_pred_eo[np.where(mino_label_pred_eo == 1)])/len(mino_label_pred_eo))
eo_boot_ratio2 = (len(majo_label_pred_eo[np.where(majo_label_pred_eo == 1)])/len(majo_label_pred_eo))
meandiff_eo_final = np.abs(np.mean(eo_boot_ratio1) - np.mean(eo_boot_ratio2))
meandiff_final = np.abs(np.mean(mino_label) - np.mean(majo_label))

err_ens_test = metrics.mean_squared_error(label_test, label_pred)

#Single normal model --- might have to change it to single fair model
model2 = AdaBoostClassifier(n_estimators = 100)
model2.fit(sample_train, label_train)
label_pred2 = model2.predict(sample_test)
mino_label = label_pred2[np.where(sample_test[:,sacol]>=threshold)]
majo_label = label_pred2[np.where(sample_test[:,sacol]<threshold)]
mino_label_pred_eo = label_pred2[((sample_test[:,sacol]>=threshold) & (label_test == 1))]
majo_label_pred_eo = label_pred2[((sample_test[:,sacol]<threshold) & (label_test == 1))]
eo_boot_ratio1 = (len(mino_label_pred_eo[np.where(mino_label_pred_eo == 1)])/len(mino_label_pred_eo))
eo_boot_ratio2 = (len(majo_label_pred_eo[np.where(majo_label_pred_eo == 1)])/len(majo_label_pred_eo))
meandiff_eo_ada = np.abs(np.mean(eo_boot_ratio1) - np.mean(eo_boot_ratio2))
meandiff_ada = np.abs(np.mean(mino_label) - np.mean(majo_label))
err_ada_test = metrics.mean_squared_error(label_test, label_pred2)

print("*****MEAN DIFFERENCE*****")
print("*****Traditional Odds MD*****")
#print("The mean difference in the train data: ", meandiff_before)
print("The mean difference in the predicted results: single: (" + repr(meandiff_one) + ")")
print("The mean difference in the predicted results: adaboost: ("+ repr(meandiff_ada) + ")")
print("The mean difference in the predicted results: fair adaboost: ("+ repr(meandiff_final) + ")")
print("*****Equalized Odds MD*****")
print("The mean difference in the predicted results: single: (" + repr(meandiff_one_eo) + ")")
print("The mean difference in the predicted results: adaboost: ("+ repr(meandiff_eo_ada) + ")")
print("The mean difference in the predicted results: fair adaboost: ("+ repr(meandiff_eo_final) + ")")

print("*****ERROR*****")
print("The error of predicted results: single: ("+ repr(err_one_test) + ")")
print("The error of predicted results: adaboost: ("+ repr(err_ada_test) + ")")
print("The error of predicted results: fair adaboost: ("+ repr(err_ens_test) + ")")