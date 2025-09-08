import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

data = np.genfromtxt('crimerate.csv', delimiter = ',')

#iter_count = list(range(10))
#
#md_iter_one = [0 for x in range(len(iter_count))]
#md_iter_ens = [0 for x in range(len(iter_count))]
#md_eo_iter_one = [0 for x in range(len(iter_count))]
#md_eo_iter_ens = [0 for x in range(len(iter_count))]
#
#for it in iter_count:
#Variables initilization
bootstaps = 20
trails = 10
sacol = 2
samplesperboot = 300
acc_one = [0 for x in range(trails)]
acc_ens = [0 for x in range(trails)]
meandiff_one_after = [0 for x in range(trails)]
meandiff_ens_after = [0 for x in range(trails)]
meandiff_b_min = [0 for x in range(trails)]
meandiff_b_max = [0 for x in range(trails)]
acc_b_min = [0 for x in range(trails)]
acc_b_max = [0 for x in range(trails)]
aa_naa_ratio_ens1 = [0 for x in range(trails)]
aa_naa_ratio_one1 = [0 for x in range(trails)]
aa_naa_ratio_ens2 = [0 for x in range(trails)]
aa_naa_ratio_one2 = [0 for x in range(trails)]
md_ens = [0 for x in range(trails)]
md_one = [0 for x in range(trails)]
md_eo_one = [0 for x in range(trails)]
md_eo_ens = [0 for x in range(trails)]
eo_one_ratio1 = [0 for x in range(trails)]
eo_one_ratio2 = [0 for x in range(trails)]
eo_ens_ratio1 = [0 for x in range(trails)]
eo_ens_ratio2 = [0 for x in range(trails)]

#Data split into samples, labels
sample = data[:,0:-1]
label = data[:,-1]
label[label>0.5] = 1
label[label<=0.5] = 0

#Calculating Mean difference in the training set
aa_label = label[np.where(sample[:,2]>=0.5)]
naa_label = label[np.where(sample[:,2]<0.5)]
aa_mean_before = np.mean(aa_label)
naa_mean_before = np.mean(naa_label)
meandiff_before = aa_mean_before - naa_mean_before

#Main method which does random ensemble of multiple unfair models
def main():
    #Random test-train split
    sample_train, sample_test, label_train, label_test = train_test_split(sample, label, test_size = 0.4)
    
    #Cross Validation code
    #sample_fold = np.zeros(5, dtype = object)
    #label_fold = np.zeros(5, dtype = object)
    #
    #i= 50
    #k = 0
    #for j in range(0, 5):    
    #    sample_fold[j] = sample_train[k:i,:]
    #    label_fold[j] = label_train[k:i]
    #    k = i
    #    i = i + 6
    #
    #mean_mse = []*8
    #var_mse = []*8
    #
    #hp = [1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
    #mse = np.zeros(5)
    #
    #for index in range(0, len(hp)):
    #    for n in range (0, 5):
    #        sample_test_fold = sample_fold[n]
    #        label_test_fold = label_fold[n]
    #        
    #        new_s = [x for i, x in enumerate(sample_fold) if i != n]
    #        sample_train_fold = np.concatenate(new_s)
    #        
    #        new_l = [x for i, x in enumerate(label_fold) if i != n]
    #        label_train_fold = np.concatenate(new_l)
    #
    #        # step 1: construct a model and set alpha to 1e-2
    #        model = linear_model.LogisticRegression(C = hp[index], solver='lbfgs') 
    #        # step 2: train a model on set (sample_train,label_train)
    #        model.fit(sample_train_fold, label_train_fold) 
    #        # step 3: apply the model to make prediction on (sample_train)
    #        label_pred_fold = model.predict(sample_test_fold)
    #        # step 4: evaluate MSE on (sample_train,label_train)
    #        mse[n] = mean_squared_error(label_test_fold, label_pred_fold) 
    #        
    #    mean_mse.append(np.mean(mse))
    #    var_mse.append(np.std(mse))
    #
    #C_opt = hp[mean_mse.index(min(mean_mse))]
    C_opt = 0.1
    
    #Model selection and prediction - non-ensemble single model
    model = linear_model.LogisticRegression(C = C_opt, solver='lbfgs')
    model.fit(sample_train, label_train)
    label_pred_b = model.predict(sample_test)
    acc_one = accuracy_score(label_test, label_pred_b)
    
    #Calculating mean difference in the prediction of a non-ensemble single model
    aa_label_pred = label_pred_b[np.where(sample_test[:,sacol]>0.5)]
    naa_label_pred = label_pred_b[np.where(sample_test[:,sacol]<0.5)]
    aa_label_pred_eo = label_pred_b[((sample_test[:,sacol]>0.5) & (label_test == 1))]
    naa_label_pred_eo = label_pred_b[((sample_test[:,sacol]<0.5) & (label_test == 1))]
    aa_mean_after = np.mean(aa_label_pred)
    naa_mean_after = np.mean(naa_label_pred)
    meandiff_one_after = aa_mean_after - naa_mean_after
    eo_one_ratio1 = (len(aa_label_pred_eo[np.where(aa_label_pred_eo == 1)])/len(aa_label_pred_eo))
    eo_one_ratio2 = (len(naa_label_pred_eo[np.where(naa_label_pred_eo == 1)])/len(naa_label_pred_eo))
    one_model_ratio1 = (len(aa_label_pred[np.where(aa_label_pred == 1)])/len(aa_label_pred))
    one_model_ratio2 = (len(naa_label_pred[np.where(naa_label_pred == 1)])/len(naa_label_pred))
    
    def bootstrap():
        #Making predictions of a single bootstrap model
        sample_train_b, label_train_b = resample(sample_train, label_train, replace="false", n_samples = samplesperboot)
        model = linear_model.LogisticRegression(C = 0.8, solver='lbfgs')
        model.fit(sample_train_b, label_train_b)
        label_pred_b = model.predict(sample_test)
        acc_iter = accuracy_score(label_test, label_pred_b)
        
        #Calculating mean diffence of a single bootstrap model
        aa_label_pred = label_pred_b[np.where(sample_test[:,sacol]>0.5)]
        naa_label_pred = label_pred_b[np.where(sample_test[:,sacol]<0.5)]
        aa_label_pred_eo = label_pred_b[((sample_test[:,sacol]>0.5) & (label_test == 1))]
        naa_label_pred_eo = label_pred_b[((sample_test[:,sacol]<0.5) & (label_test == 1))]
        aa_mean_after = np.mean(aa_label_pred)
        naa_mean_after = np.mean(naa_label_pred)
        meandiff_after_iter = aa_mean_after - naa_mean_after
        eo_boot_ratio1 = (len(aa_label_pred_eo[np.where(aa_label_pred_eo == 1)])/len(aa_label_pred_eo))
        eo_boot_ratio2 = (len(naa_label_pred_eo[np.where(naa_label_pred_eo == 1)])/len(naa_label_pred_eo))
        boot_ratio1 = (len(aa_label_pred[np.where(aa_label_pred == 1)])/len(aa_label_pred))
        boot_ratio2 = (len(naa_label_pred[np.where(naa_label_pred == 1)])/len(naa_label_pred))
        return label_pred_b, acc_iter, meandiff_after_iter, boot_ratio1, boot_ratio2, eo_boot_ratio1, eo_boot_ratio2

    #Declaring variables required for bootstrap
    label_pred = [0 for x in range(bootstaps)]
    meandiff_b = [0 for x in range(bootstaps)]
    acc_b = [0 for x in range(bootstaps)]
    boot_ratio1 = [0 for x in range(bootstaps)]
    boot_ratio2 = [0 for x in range(bootstaps)]
    eo_boot_ratio1 = [0 for x in range(bootstaps)]
    eo_boot_ratio2 = [0 for x in range(bootstaps)]
    
    for bs in range(bootstaps):        
        #Calling the boot strap method
        label_pred[bs], acc_b[bs], meandiff_b[bs], boot_ratio1[bs], boot_ratio2[bs], eo_boot_ratio1[bs], eo_boot_ratio2[bs] = bootstrap()
        
    #Calculating the mean of labels and MSE of all bootstrap models
    maj = np.mean(label_pred, axis = 0)
    maj[maj > 0.5] = 1
    maj[maj <= 0.5] = 0
    acc_ens = accuracy_score(label_test, maj)
    
    #Calculating the mean diff of the ensemble model
    aa_label_pred = maj[np.where(sample_test[:,sacol]>0.5)]
    naa_label_pred = maj[np.where(sample_test[:,sacol]<0.5)]
    aa_mean_after = np.mean(aa_label_pred)
    naa_mean_after = np.mean(naa_label_pred)
    meandiff_ens_after = aa_mean_after - naa_mean_after
    
    return acc_one, meandiff_one_after, acc_ens, meandiff_ens_after, np.min(meandiff_b), np.max(meandiff_b), np.min(acc_b), np.max(acc_b), one_model_ratio1, one_model_ratio2,np.mean(boot_ratio1), np.mean(boot_ratio2), (np.mean(boot_ratio1) - np.mean(boot_ratio2)), (one_model_ratio1 - one_model_ratio2), (eo_one_ratio1 - eo_one_ratio2), (np.mean(eo_boot_ratio1) - np.mean(eo_boot_ratio2)), eo_one_ratio1, eo_one_ratio2, np.mean(eo_boot_ratio1), np.mean(eo_boot_ratio2)
    
for t in range(trails):
    #Collecting all the information for each trail
    acc_one[t], meandiff_one_after[t], acc_ens[t], meandiff_ens_after[t], meandiff_b_min[t], meandiff_b_max[t], acc_b_min[t], acc_b_max[t], aa_naa_ratio_one1[t], aa_naa_ratio_one2[t], aa_naa_ratio_ens1[t], aa_naa_ratio_ens2[t], md_ens[t], md_one[t], md_eo_one[t], md_eo_ens[t], eo_one_ratio1[t], eo_one_ratio2[t], eo_ens_ratio1[t], eo_ens_ratio2[t] = main()

#md_iter_one[it] = np.mean(md_one)
#md_iter_ens[it] = np.mean(md_ens)
#md_eo_iter_one[it] = np.mean(md_eo_one)
#md_eo_iter_ens[it] = np.mean(md_eo_ens)

print("*****MEAN DIFFERENCE*****")
print("*****Traditional Odds MD*****")
#print("The mean difference in the train data: ", meandiff_before)
print("The mean difference in the predicted results: single(mean, var): (" + repr(np.mean(md_one)) + "," +  repr(np.var(md_one)) + ")")
print("The mean difference in the predicted results: ensemble(mean, var): ("+ repr(np.mean(md_ens)) + "," +  repr(np.var(md_ens)) + ")")
print("*****Equalized Odds MD*****")
print("The mean difference in the predicted results: single(mean, var): (" + repr(np.mean(md_eo_one)) + "," +  repr(np.var(md_eo_one)) + ")")
print("The mean difference in the predicted results: ensemble(mean, var): ("+ repr(np.mean(md_eo_ens)) + "," +  repr(np.var(md_eo_ens)) + ")")
#print("The min mean difference among all bootstrap iterations(mean, var): ("+ repr(np.mean(meandiff_b_min)) + "," +  repr(np.var(meandiff_b_min)) + ")")
#print("The max mean difference among all bootstrap iterations(mean, var): ("+ repr(np.mean(meandiff_b_max)) + "," +  repr(np.var(meandiff_b_max)) + ")")

print("*****ACCURACY SCORE*****")
print("The accuracy score of predicted results: single(mean, var): ("+ repr(np.mean(acc_one)) + "," +  repr(np.var(acc_one)) + ")")
print("The accuracy score of predicted results: ensemble(mean, var): ("+ repr(np.mean(acc_ens)) + "," +  repr(np.var(acc_ens)) + ")")
#print("The min accuracy score of all bootstrap iterations(mean, var): ("+ repr(np.mean(acc_b_min)) + "," +  repr(np.var(acc_b_min)) + ")")
#print("The max accuracy score of all bootstrap iterations(mean, var): ("+ repr(np.mean(acc_b_max)) + "," +  repr(np.var(acc_b_max)) + ")")

print("*****RATIO of Minority:Majority aquittal rate*****")
print("*****Traditional odds Ratio*****")
print("Single Model Ratio: (mean, var1, var2): (" + repr(np.mean(aa_naa_ratio_one1)/np.mean(aa_naa_ratio_one2)) + ", " + repr(np.var(aa_naa_ratio_one1)) + "," + repr(np.var(aa_naa_ratio_one2)) + ")")
print("Ensemble Model Ratio: (mean, var1, var2): (" + repr(np.mean(aa_naa_ratio_ens1)/np.mean(aa_naa_ratio_ens2)) + ", " + repr(np.var(aa_naa_ratio_ens1)) + "," + repr(np.var(aa_naa_ratio_ens2)) + ")")
print(np.mean(aa_naa_ratio_one1)/np.mean(aa_naa_ratio_one2) - np.mean(aa_naa_ratio_ens1)/np.mean(aa_naa_ratio_ens2))
print("*****Equalized Odds Ratio*****")
print("Single Model Ratio: (mean, var1, var2): (" + repr(np.mean(eo_one_ratio1)/np.mean(eo_one_ratio2)) + ", " + repr(np.var(eo_one_ratio1)) + "," + repr(np.var(eo_one_ratio2)) + ")")
print("Ensemble Model Ratio: (mean, var1, var2): (" + repr(np.mean(eo_ens_ratio1)/np.mean(eo_ens_ratio2)) + ", " + repr(np.var(eo_ens_ratio1)) + "," + repr(np.var(eo_ens_ratio2)) + ")")
print(np.mean(eo_one_ratio1)/np.mean(eo_one_ratio2) - np.mean(eo_ens_ratio1)/np.mean(eo_ens_ratio2))

#fig = plt.figure()  
#plt.plot(iter_count, md_iter_one)
#plt.plot(iter_count, md_eo_iter_one)
#plt.show()   
#
#fig = plt.figure()  
#plt.plot(iter_count, md_iter_ens)
#plt.plot(iter_count, md_eo_iter_ens)
#plt.show()   