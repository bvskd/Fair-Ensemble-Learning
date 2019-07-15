import numpy as np
from sklearn import linear_model
from sklearn.model_selection import train_test_split
#import statistics as stat
import sklearn.metrics as metrics
from sklearn.ensemble import AdaBoostClassifier
from scipy.spatial.distance import cdist

##########################DATA HANDLING##############################

#Load Data --> COMPASS.csv, CreditCard.csv, crimecommunity.csv
data = np.genfromtxt('crimecommunity.csv', delimiter = ',')

#Split data into sample and label sets
sample = data[:,0:-1]
label = data[:, -1]
label[np.where(label==0)] = -1

#Use test_train_split to split sample and label further into sample_train, label_train, sample_test and label_test
#Using test:train ratio to be 1:1 and random state = 2
sample_train, sample_test, label_train, label_test = train_test_split(sample, label, test_size = 0.5, random_state = 2)

##########################DATA HANDLING##############################

######################VARIABLE DECLARION#############################
#Number of neighbors to look at for calculating delta_p (probability difference b^n majority and minority classes to be predicted as "1")
k = 15
#Need not be used but is being used to decide which instances belong to majority or minority class by checking sensitive attribute column
threshold = 0.5
#Instance weight matrix (higher weight instances that are unfairly treated)
w = np.ones(len(label_train))
#Normalizing the weights
w = w/len(label_train)
#weight of each model in the adaboost ensemble
alpha = []
#This value is calculated based on error matrix and helps in calculation of alpha
epsilon = []
#error matrix is set to "1" for all the instances that are unfairly treated so that can be given higher weight
err_mat = np.zeros(len(label_train))
#sensitive attribute column number
sacol = 0
#thresholding variable for delta_p
thresh = 0.3
#number of iterations for both Fair ADA and normal ADA
num_iter = 50

######################VARIABLE DECLARION#############################

#################HELPER METHOD FAIR ADABOOST#########################

#This method calculates error matrix based on "k" neighbors of each instance by using pairwise distance matrix
def weighUnfairInstances(strain, ltrain, trlpred):
    strain = sample_train[:,sacol+1:]
    dist_mat = cdist(strain, strain)
    for i in range(len(strain)):
        dist = dist_mat[i].tolist()
        lst = np.asarray(trlpred)[np.argsort(dist)[1:k+1].tolist()]
        sa = np.transpose(strain[np.argsort(dist)[1:k+1].tolist()])[sacol]        
        mino_neigh = lst[np.where(sa == 1)]
        majo_neigh = lst[np.where(sa == 0)]
        if (len(majo_neigh) == 0):
            p_majo = 0
        else:
            p_majo = len(np.where(majo_neigh == 1))/len(majo_neigh)
            
        if (len(mino_neigh) == 0):
            p_mino = 0
        else:
            p_mino = len(np.where(mino_neigh == 1))/len(mino_neigh)
            
        delta_p = np.abs(p_majo - p_mino)
        if(delta_p <= thresh):#for discrete labels #trlpred[i] == ltrain[i] and delta_p < thresh #stat.mode(lst) == trlpred[i]
            err_mat[i] = 0
        else:
            err_mat[i] = 1
            
#################HELPER METHOD FAIR ADABOOST#########################

####################SINGLE STANDARD MODEL############################

#Single standard logistic regression model
model1 = linear_model.LogisticRegression(class_weight='balanced', solver = 'liblinear')
model1.fit(sample_train, label_train)
#Single logistic regression model prediction
label_pred1 = model1.predict(sample_test)
#Calculation of statistical parity for single logistic regression model
mino_label = label_pred1[np.where(sample_test[:,sacol]>=threshold)]
majo_label = label_pred1[np.where(sample_test[:,sacol]<threshold)]
mino_prob = len(np.asarray(np.where(mino_label == 1))[0])/len(mino_label)
majo_prob = len(np.asarray(np.where(majo_label == 1))[0])/len(majo_label)
print("ONE - MINO: " + repr(len(np.asarray(np.where(mino_label == 1))[0])) + "/" + repr(len(mino_label)))
print("ONE - MAJO: " + repr(len(np.asarray(np.where(majo_label == 1))[0])) + "/" + repr(len(majo_label)))
# --- fix syntax ---
#np.sum(mino_label == 1)/len(mino_label)
# ------------------
meandiff_one = np.abs(majo_prob - mino_prob)
#Calculation of equalized odds SP for single logistic regression model
mino_label_pred_eo = label_pred1[((sample_test[:,sacol]>=threshold) & (label_test == 1))]
majo_label_pred_eo = label_pred1[((sample_test[:,sacol]<threshold) & (label_test == 1))]
eo_boot_ratio1 = (len(mino_label_pred_eo[np.where(mino_label_pred_eo == 1)])/len(mino_label_pred_eo))
eo_boot_ratio2 = (len(majo_label_pred_eo[np.where(majo_label_pred_eo == 1)])/len(majo_label_pred_eo))
meandiff_one_eo = np.abs(np.mean(eo_boot_ratio1) - np.mean(eo_boot_ratio2))
#Calculation of classification error for single logistic regression model
err_one_test = (1 - metrics.accuracy_score(label_test, label_pred1))

####################SINGLE STANDARD MODEL############################

###################STANDARD ADABOOST MODEL###########################
#Satndard Adaboost model: base model --> standard logistic regression
model2 = AdaBoostClassifier(n_estimators = num_iter, base_estimator = linear_model.LogisticRegression(class_weight='balanced', solver = 'liblinear'))
model2.fit(sample_train, label_train)
#Adaboost model prediction
label_pred2 = model2.predict(sample_test) # standard adaboost 
#Calculation of statistical parity for standard ADABOOST model
mino_label = label_pred2[np.where(sample_test[:,sacol]>=threshold)]
majo_label = label_pred2[np.where(sample_test[:,sacol]<threshold)]
mino_prob = len(np.asarray(np.where(mino_label == 1))[0])/len(mino_label)
majo_prob = len(np.asarray(np.where(majo_label == 1))[0])/len(majo_label)
print("ADA - MINO: " + repr(len(np.asarray(np.where(mino_label == 1))[0])) + "/" + repr(len(mino_label)))
print("ADA - MAJO: " + repr(len(np.asarray(np.where(majo_label == 1))[0])) + "/" + repr(len(majo_label)))
meandiff_ada = np.abs(majo_prob - mino_prob)
#Calculation of equilazied odds SP for standard ADABOOST model
mino_label_pred_eo = label_pred2[((sample_test[:,sacol]>=threshold) & (label_test == 1))]
majo_label_pred_eo = label_pred2[((sample_test[:,sacol]<threshold) & (label_test == 1))]
eo_boot_ratio1 = (len(mino_label_pred_eo[np.where(mino_label_pred_eo == 1)])/len(mino_label_pred_eo))
eo_boot_ratio2 = (len(majo_label_pred_eo[np.where(majo_label_pred_eo == 1)])/len(majo_label_pred_eo))
meandiff_eo_ada = np.abs(np.mean(eo_boot_ratio1) - np.mean(eo_boot_ratio2))
#Calculation of classification error for standard ADABOOST model
err_ada_test = (1 - metrics.accuracy_score(label_test, label_pred2))

###################STANDARD ADABOOST MODEL###########################

#####################FAIR ADABOOST MODEL#############################
#Loop variable (Index variable)
iteration = 0

#variables used to store test and train model predictions for each estimator
test_pred_iter = []
train_pred_iter = []
while iteration < num_iter:
    print("Iteration: " + repr(iteration + 1))
    #Using standard logistic regression model as base model similar to standard ADABOOSt
    model = linear_model.LogisticRegression(class_weight='balanced', solver = 'liblinear')
    model.fit(sample_train, label_train, sample_weight = w)
    #Test and train predections for each estimator
    test_pred_iter.append(model.predict(sample_test))
    train_pred_iter.append(model.predict(sample_train))
    #reestimating the error matrix after each estimator predictions
    weighUnfairInstances(sample_train, label_train, train_pred_iter[iteration])#Updates error matrix
    #epsilon calcuation
    epsilon.append(np.sum(w * err_mat)/np.sum(w))
    print(epsilon)
    #model weight calculation for each estimator
    alpha.append((0.5) * np.log((1-epsilon[iteration])/epsilon[iteration]))
    #Updating instance weights
    w = (w * np.exp(((-1 * label_train) * (alpha[iteration])) * (train_pred_iter[iteration])))
    #Normalizing instance weights
    w = w/np.sum(w)
    
    #This is being done just to see how the SP is going to vary for each estimator
    #Not necessary can be removed
    mino_label = test_pred_iter[iteration][np.where(sample_test[:,sacol]>=threshold)]
    majo_label = test_pred_iter[iteration][np.where(sample_test[:,sacol]<threshold)]
    meandiff = np.abs(np.mean(mino_label) - np.mean(majo_label))
    print("BOOT - MINO: " + repr(len(np.asarray(np.where(mino_label == 1))[0])) + "/" + repr(len(mino_label)))
    print("BOOT - MAJO: " + repr(len(np.asarray(np.where(majo_label == 1))[0])) + "/" + repr(len(majo_label)))
    
    #Incrementing loop index variable
    iteration = iteration + 1
#    print("iteration: " + repr(iteration) + " Mean Diff: " + repr(meandiff))

#Ensemble FAIR ADABOOST prediction    
label_pred = np.transpose(np.sum(np.matmul(np.diag(alpha), np.asmatrix(test_pred_iter)), axis = 0))/iteration
#Thresholding the FAIR ADABOST predictions
label_pred[label_pred>=0] = 1
label_pred[label_pred!=1] = -1
#Calcualting statistical parity for FAIR ADABOOST
mino_label = label_pred[np.where(sample_test[:,sacol]>=threshold)]
majo_label = label_pred[np.where(sample_test[:,sacol]<threshold)]
mino_prob = len(np.asarray(np.where(mino_label == 1))[0])/len(mino_label)
majo_prob = len(np.asarray(np.where(majo_label == 1))[0])/len(majo_label)
print("ENS - MINO: " + repr(len(np.asarray(np.where(mino_label == 1))[0])) + "/" + repr(len(mino_label)))
print("ENS - MAJO: " + repr(len(np.asarray(np.where(majo_label == 1))[0])) + "/" + repr(len(majo_label)))
meandiff_final = np.abs(majo_prob - mino_prob)
#Calculating equalized odds SP for FAIR ADABOOST
mino_label_pred_eo = label_pred[((sample_test[:,sacol]>=threshold) & (label_test == 1))]
majo_label_pred_eo = label_pred[((sample_test[:,sacol]<threshold) & (label_test == 1))]
eo_boot_ratio1 = (len(mino_label_pred_eo[np.where(mino_label_pred_eo == 1)])/len(mino_label_pred_eo))
eo_boot_ratio2 = (len(majo_label_pred_eo[np.where(majo_label_pred_eo == 1)])/len(majo_label_pred_eo))
meandiff_eo_final = np.abs(np.mean(eo_boot_ratio1) - np.mean(eo_boot_ratio2))
#Calculation of classification error for FAIR ADABOOST
err_ens_test = (1 - metrics.accuracy_score(label_test, label_pred))

#####################FAIR ADABOOST MODEL#############################

###########################RESULTS###################################

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

###########################RESULTS###################################