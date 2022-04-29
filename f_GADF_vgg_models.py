from sklearn.metrics import accuracy_score
import numpy as np
import os
import random
import scipy.io
import scipy.signal
from scipy.spatial.distance import pdist,squareform
import cv2
import torch
from byol_pytorch import BYOL
from torchvision import models
from torchvision import transforms
from torchsummary import summary
from sklearn.model_selection import train_test_split
from copy import deepcopy
from pyts.image import GramianAngularField

device = 'cuda:2'

        
def single_model(X_train_final_acc, X_val_final_acc, Y_train_final_acc, Y_val_final_acc,X_test_ssm_acc,Y_test,architecture):

    if architecture=='vgg11':
        model = models.vgg11(pretrained=True)
    elif architecture=='vgg11bn':
        model = models.vgg11_bn(pretrained=True)
    elif architecture=='vgg13':
        model = models.vgg13(pretrained=True)
    elif architecture=='vgg13bn':
        model = models.vgg13_bn(pretrained=True)
    elif architecture=='vgg16':
        model = models.vgg16(pretrained=True)
    elif architecture=='vgg16bn':
        model = models.vgg16_bn(pretrained=True)
    elif architecture=='vgg19':
        model = models.vgg19(pretrained=True)
    elif architecture=='vgg19bn':
        model = models.vgg19_bn(pretrained=True)
        
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1] 
    features.extend([torch.nn.Linear(num_features, 26)]) 
    model.classifier = torch.nn.Sequential(*features) 
    model = model.to(device)
        
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
    
    best_val_acc = -1000
    best_val_model = None
    batch_size = 32
    val_acc_arr = -1000*np.ones((10,))
    es_patience=10+1
    num_epochs = 1000
    
    print('Started training')
    for epoch in range(0,num_epochs):  
        model.train(True)
        running_loss = 0.0
        running_acc = 0
        running_val_loss = 0.0
        for i in range(0,len(X_train_final_acc)//batch_size):
            inputs = normalize(torch.tensor(X_train_final_acc[i*batch_size:(i+1)*batch_size,:,:,:], dtype=torch.float)).to(device)
            labels = torch.tensor(Y_train_final_acc[i*batch_size:(i+1)*batch_size,], dtype=torch.long).to(device)
    
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
    
            running_loss += loss.item() * inputs.size(0)
            out = torch.argmax(outputs.detach(),dim=1)
            assert out.shape==labels.shape
            running_acc += (labels==out).sum().item()
    
        correct = 0
        model.train(False)
        with torch.no_grad():
            for i in range(0,len(X_val_final_acc)//batch_size):
                inputs = normalize(torch.tensor(X_val_final_acc[i*batch_size:(i+1)*batch_size,:,:,:], dtype=torch.float)).to(device)
                labels = torch.tensor(Y_val_final_acc[i*batch_size:(i+1)*batch_size,], dtype=torch.long).to(device)
                out = model(inputs)
                out = torch.argmax(out,dim=1)
                acc = (out==labels).sum().item()
                correct += acc
                val_loss = criterion(model(inputs), labels)
                running_val_loss += val_loss.item() * inputs.size(0)
        print(f"Train loss {epoch+1}: {running_loss/len(X_train_final_acc)},Train Acc:{running_acc*100/len(X_train_final_acc)},Validation loss: {running_val_loss/len(X_val_final_acc)},Validation Acc:{correct*100/len(X_val_final_acc)}%")
        
        if correct>best_val_acc:
            best_val_acc = correct
            best_val_model = deepcopy(model.state_dict())
            
        val_acc_arr = np.concatenate((val_acc_arr,np.asarray(np.reshape(correct*100/len(X_val_final_acc),(1,)))))
        if val_acc_arr[-es_patience] == np.max(val_acc_arr[-es_patience:]):
            break
        
    print('Finished Training')  
    
    predictions = np.zeros((len(X_test_ssm_acc),26))
    model.load_state_dict(best_val_model)
    model.train(False)
    
    for i in range(0,len(X_test_ssm_acc)):
        test_preds = model(normalize(torch.tensor(np.reshape(X_test_ssm_acc[i,:,:,:],(1,3,155,155)), dtype=torch.float).to(device)))
        prob = torch.nn.functional.softmax(test_preds, dim=1)
        prob = prob.cpu().detach().numpy()
        predictions[i] = prob
        
    Y_pred = np.argmax(np.asarray(predictions),axis=1)
    acc = accuracy_score(Y_test,Y_pred)
    print(acc)
    
    del model
    
    return predictions
    
X_train = np.load('data/X_train.npy')
Y_train = np.load('data/Y_train.npy')
X_test = np.load('data/X_test.npy')
Y_test = np.load('data/Y_test.npy')

transformer = GramianAngularField(method='difference')

X_train_ssm_acc = np.zeros((len(X_train),3,155,155))
for j in range(0,len(X_train)):
    x_dir = np.reshape(X_train[j,:,0],(1,155))
    ssm_x = transformer.transform(x_dir)
    ssm_x_res = ssm_x/np.max(ssm_x)

    y_dir = np.reshape(X_train[j,:,1],(1,155))
    ssm_y = transformer.transform(y_dir)
    ssm_y_res = ssm_y/np.max(ssm_y)
    
    z_dir = np.reshape(X_train[j,:,2],(1,155))
    ssm_z = transformer.transform(z_dir)
    ssm_z_res = ssm_z/np.max(ssm_z)
    
    X_train_ssm_acc[j,0,:,:] = ssm_x_res
    X_train_ssm_acc[j,1,:,:] = ssm_y_res
    X_train_ssm_acc[j,2,:,:] = ssm_z_res
    
X_test_ssm_acc = np.zeros((len(X_test),3,155,155))
for j in range(0,len(X_test)):
    x_dir = np.reshape(X_test[j,:,0],(1,155))
    ssm_x = transformer.transform(x_dir)
    ssm_x_res = ssm_x/np.max(ssm_x)
    
    y_dir = np.reshape(X_test[j,:,1],(1,155))
    ssm_y = transformer.transform(y_dir)
    ssm_y_res = ssm_y/np.max(ssm_y)
    
    z_dir = np.reshape(X_test[j,:,2],(1,155))
    ssm_z = transformer.transform(z_dir)
    ssm_z_res = ssm_z/np.max(ssm_z)
    
    X_test_ssm_acc[j,0,:,:] = ssm_x_res
    X_test_ssm_acc[j,1,:,:] = ssm_y_res
    X_test_ssm_acc[j,2,:,:] = ssm_z_res
    
X_train_ssm_gyro = np.zeros((len(X_train),3,155,155))
for j in range(0,len(X_train)):
    x_dir = np.reshape(X_train[j,:,3],(1,155))
    ssm_x = transformer.transform(x_dir)
    ssm_x_res = ssm_x/np.max(ssm_x)
    
    y_dir = np.reshape(X_train[j,:,4],(1,155))
    ssm_y = transformer.transform(y_dir)
    ssm_y_res = ssm_y/np.max(ssm_y)
    
    z_dir = np.reshape(X_train[j,:,5],(1,155))
    ssm_z = transformer.transform(z_dir)
    ssm_z_res = ssm_z/np.max(ssm_z)
    
    X_train_ssm_gyro[j,0,:,:] = ssm_x_res
    X_train_ssm_gyro[j,1,:,:] = ssm_y_res
    X_train_ssm_gyro[j,2,:,:] = ssm_z_res
    
X_test_ssm_gyro = np.zeros((len(X_test),3,155,155))
for j in range(0,len(X_test)):
    x_dir = np.reshape(X_test[j,:,3],(1,155))
    ssm_x = transformer.transform(x_dir)
    ssm_x_res = ssm_x/np.max(ssm_x)
    
    y_dir = np.reshape(X_test[j,:,4],(1,155))
    ssm_y = transformer.transform(y_dir)
    ssm_y_res = ssm_y/np.max(ssm_y)
    
    z_dir = np.reshape(X_test[j,:,5],(1,155))
    ssm_z = transformer.transform(z_dir)
    ssm_z_res = ssm_z/np.max(ssm_z)
    
    X_test_ssm_gyro[j,0,:,:] = ssm_x_res
    X_test_ssm_gyro[j,1,:,:] = ssm_y_res
    X_test_ssm_gyro[j,2,:,:] = ssm_z_res
    
X_train_final_acc, X_val_final_acc, Y_train_final_acc, Y_val_final_acc = train_test_split(X_train_ssm_acc, Y_train, test_size=0.2, random_state=1)
X_train_final_gyro, X_val_final_gyro, Y_train_final_gyro, Y_val_final_gyro = train_test_split(X_train_ssm_gyro, Y_train, test_size=0.2, random_state=1)

predictions1 = single_model(X_train_final_acc, X_val_final_acc, Y_train_final_acc, Y_val_final_acc,X_test_ssm_acc,Y_test,'vgg11')
predictions2 = single_model(X_train_final_gyro, X_val_final_gyro, Y_train_final_gyro, Y_val_final_gyro,X_test_ssm_gyro,Y_test,'vgg11')
predictions = np.asarray(predictions1+predictions2)
Y_pred_acc = np.argmax(np.asarray(predictions1),axis=1)
Y_pred_gyro = np.argmax(np.asarray(predictions2),axis=1)
Y_pred = np.argmax(np.asarray(predictions),axis=1)
print('vgg11 : Accelerometer = ' + str(accuracy_score(Y_test,Y_pred_acc)) + ' Gyroscope = ' + str(accuracy_score(Y_test,Y_pred_gyro)) + ' Fused = ' + str(accuracy_score(Y_test,Y_pred)))
np.save('results/final_results/GADF/acc_vgg11.npy',Y_pred_acc)
np.save('results/final_results/GADF/gyro_vgg11.npy',Y_pred_gyro)
np.save('results/final_results/GADF/fused_vgg11.npy',Y_pred)

predictions1 = single_model(X_train_final_acc, X_val_final_acc, Y_train_final_acc, Y_val_final_acc,X_test_ssm_acc,Y_test,'vgg11bn')
predictions2 = single_model(X_train_final_gyro, X_val_final_gyro, Y_train_final_gyro, Y_val_final_gyro,X_test_ssm_gyro,Y_test,'vgg11bn')
predictions = np.asarray(predictions1+predictions2)
Y_pred_acc = np.argmax(np.asarray(predictions1),axis=1)
Y_pred_gyro = np.argmax(np.asarray(predictions2),axis=1)
Y_pred = np.argmax(np.asarray(predictions),axis=1)
print('vgg11bn : Accelerometer = ' + str(accuracy_score(Y_test,Y_pred_acc)) + ' Gyroscope = ' + str(accuracy_score(Y_test,Y_pred_gyro)) + ' Fused = ' + str(accuracy_score(Y_test,Y_pred)))
np.save('results/final_results/GADF/acc_vgg11bn.npy',Y_pred_acc)
np.save('results/final_results/GADF/gyro_vgg11bn.npy',Y_pred_gyro)
np.save('results/final_results/GADF/fused_vgg11bn.npy',Y_pred)

predictions1 = single_model(X_train_final_acc, X_val_final_acc, Y_train_final_acc, Y_val_final_acc,X_test_ssm_acc,Y_test,'vgg13')
predictions2 = single_model(X_train_final_gyro, X_val_final_gyro, Y_train_final_gyro, Y_val_final_gyro,X_test_ssm_gyro,Y_test,'vgg13')
predictions = np.asarray(predictions1+predictions2)
Y_pred_acc = np.argmax(np.asarray(predictions1),axis=1)
Y_pred_gyro = np.argmax(np.asarray(predictions2),axis=1)
Y_pred = np.argmax(np.asarray(predictions),axis=1)
print('vgg13 : Accelerometer = ' + str(accuracy_score(Y_test,Y_pred_acc)) + ' Gyroscope = ' + str(accuracy_score(Y_test,Y_pred_gyro)) + ' Fused = ' + str(accuracy_score(Y_test,Y_pred)))
np.save('results/final_results/GADF/acc_vgg13.npy',Y_pred_acc)
np.save('results/final_results/GADF/gyro_vgg13.npy',Y_pred_gyro)
np.save('results/final_results/GADF/fused_vgg13.npy',Y_pred)

predictions1 = single_model(X_train_final_acc, X_val_final_acc, Y_train_final_acc, Y_val_final_acc,X_test_ssm_acc,Y_test,'vgg13bn')
predictions2 = single_model(X_train_final_gyro, X_val_final_gyro, Y_train_final_gyro, Y_val_final_gyro,X_test_ssm_gyro,Y_test,'vgg13bn')
predictions = np.asarray(predictions1+predictions2)
Y_pred_acc = np.argmax(np.asarray(predictions1),axis=1)
Y_pred_gyro = np.argmax(np.asarray(predictions2),axis=1)
Y_pred = np.argmax(np.asarray(predictions),axis=1)
print('vgg13bn : Accelerometer = ' + str(accuracy_score(Y_test,Y_pred_acc)) + ' Gyroscope = ' + str(accuracy_score(Y_test,Y_pred_gyro)) + ' Fused = ' + str(accuracy_score(Y_test,Y_pred)))
np.save('results/final_results/GADF/acc_vgg13bn.npy',Y_pred_acc)
np.save('results/final_results/GADF/gyro_vgg13bn.npy',Y_pred_gyro)
np.save('results/final_results/GADF/fused_vgg13bn.npy',Y_pred)

predictions1 = single_model(X_train_final_acc, X_val_final_acc, Y_train_final_acc, Y_val_final_acc,X_test_ssm_acc,Y_test,'vgg16')
predictions2 = single_model(X_train_final_gyro, X_val_final_gyro, Y_train_final_gyro, Y_val_final_gyro,X_test_ssm_gyro,Y_test,'vgg16')
predictions = np.asarray(predictions1+predictions2)
Y_pred_acc = np.argmax(np.asarray(predictions1),axis=1)
Y_pred_gyro = np.argmax(np.asarray(predictions2),axis=1)
Y_pred = np.argmax(np.asarray(predictions),axis=1)
print('vgg16 : Accelerometer = ' + str(accuracy_score(Y_test,Y_pred_acc)) + ' Gyroscope = ' + str(accuracy_score(Y_test,Y_pred_gyro)) + ' Fused = ' + str(accuracy_score(Y_test,Y_pred)))
np.save('results/final_results/GADF/acc_vgg16.npy',Y_pred_acc)
np.save('results/final_results/GADF/gyro_vgg16.npy',Y_pred_gyro)
np.save('results/final_results/GADF/fused_vgg16.npy',Y_pred)

predictions1 = single_model(X_train_final_acc, X_val_final_acc, Y_train_final_acc, Y_val_final_acc,X_test_ssm_acc,Y_test,'vgg16bn')
predictions2 = single_model(X_train_final_gyro, X_val_final_gyro, Y_train_final_gyro, Y_val_final_gyro,X_test_ssm_gyro,Y_test,'vgg16bn')
predictions = np.asarray(predictions1+predictions2)
Y_pred_acc = np.argmax(np.asarray(predictions1),axis=1)
Y_pred_gyro = np.argmax(np.asarray(predictions2),axis=1)
Y_pred = np.argmax(np.asarray(predictions),axis=1)
print('vgg16bn : Accelerometer = ' + str(accuracy_score(Y_test,Y_pred_acc)) + ' Gyroscope = ' + str(accuracy_score(Y_test,Y_pred_gyro)) + ' Fused = ' + str(accuracy_score(Y_test,Y_pred)))
np.save('results/final_results/GADF/acc_vgg16bn.npy',Y_pred_acc)
np.save('results/final_results/GADF/gyro_vgg16bn.npy',Y_pred_gyro)
np.save('results/final_results/GADF/fused_vgg16bn.npy',Y_pred)

predictions1 = single_model(X_train_final_acc, X_val_final_acc, Y_train_final_acc, Y_val_final_acc,X_test_ssm_acc,Y_test,'vgg19')
predictions2 = single_model(X_train_final_gyro, X_val_final_gyro, Y_train_final_gyro, Y_val_final_gyro,X_test_ssm_gyro,Y_test,'vgg19')
predictions = np.asarray(predictions1+predictions2)
Y_pred_acc = np.argmax(np.asarray(predictions1),axis=1)
Y_pred_gyro = np.argmax(np.asarray(predictions2),axis=1)
Y_pred = np.argmax(np.asarray(predictions),axis=1)
print('vgg19 : Accelerometer = ' + str(accuracy_score(Y_test,Y_pred_acc)) + ' Gyroscope = ' + str(accuracy_score(Y_test,Y_pred_gyro)) + ' Fused = ' + str(accuracy_score(Y_test,Y_pred)))
np.save('results/final_results/GADF/acc_vgg19.npy',Y_pred_acc)
np.save('results/final_results/GADF/gyro_vgg19.npy',Y_pred_gyro)
np.save('results/final_results/GADF/fused_vgg19.npy',Y_pred)

predictions1 = single_model(X_train_final_acc, X_val_final_acc, Y_train_final_acc, Y_val_final_acc,X_test_ssm_acc,Y_test,'vgg19bn')
predictions2 = single_model(X_train_final_gyro, X_val_final_gyro, Y_train_final_gyro, Y_val_final_gyro,X_test_ssm_gyro,Y_test,'vgg19bn')
predictions = np.asarray(predictions1+predictions2)
Y_pred_acc = np.argmax(np.asarray(predictions1),axis=1)
Y_pred_gyro = np.argmax(np.asarray(predictions2),axis=1)
Y_pred = np.argmax(np.asarray(predictions),axis=1)
print('vgg19bn : Accelerometer = ' + str(accuracy_score(Y_test,Y_pred_acc)) + ' Gyroscope = ' + str(accuracy_score(Y_test,Y_pred_gyro)) + ' Fused = ' + str(accuracy_score(Y_test,Y_pred)))
np.save('results/final_results/GADF/acc_vgg19bn.npy',Y_pred_acc)
np.save('results/final_results/GADF/gyro_vgg19bn.npy',Y_pred_gyro)
np.save('results/final_results/GADF/fused_vgg19bn.npy',Y_pred)
