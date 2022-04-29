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

device = 'cuda:3'

class GoogleNetModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        base = models.googlenet(pretrained=True)
        self.base = torch.nn.Sequential(*list(base.children())[:-1])
        in_features = base.fc.in_features
        self.drop = torch.nn.Dropout()
        self.final = torch.nn.Linear(in_features,26)
    
    def forward(self,x):
        x = self.base(x)
        x = self.drop(x.view(-1,self.final.in_features))
        return self.final(x)
        
def AlexNetModel():
    model = models.alexnet(pretrained=True)
    num_features = model.classifier[6].in_features
    features = list(model.classifier.children())[:-1] 
    features.extend([torch.nn.Dropout(),torch.nn.Linear(num_features, 26)]) 
    model.classifier = torch.nn.Sequential(*features)  
    return model
    
    
def single_model(X_train_final_acc, X_val_final_acc, Y_train_final_acc, Y_val_final_acc,X_test_ssm_acc,Y_test,architecture):

    if architecture=='alexnet':
        model = AlexNetModel()
    elif architecture=='googlenet':
        model = GoogleNetModel()
        
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

X_train_ssm_acc = np.zeros((len(X_train),3,155,155))
for j in range(0,len(X_train)):
    x_dir = np.reshape(X_train[j,:,0],(155,1))
    ssm_x = squareform(pdist(x_dir, metric='euclidean'))
    ssm_x_res = ssm_x/np.max(ssm_x)
    
    y_dir = np.reshape(X_train[j,:,1],(155,1))
    ssm_y = squareform(pdist(y_dir, metric='euclidean'))
    ssm_y_res = ssm_y/np.max(ssm_y)
    
    z_dir = np.reshape(X_train[j,:,2],(155,1))
    ssm_z = squareform(pdist(z_dir, metric='euclidean'))
    ssm_z_res = ssm_z/np.max(ssm_z)
    
    X_train_ssm_acc[j,0,:,:] = ssm_x_res
    X_train_ssm_acc[j,1,:,:] = ssm_y_res
    X_train_ssm_acc[j,2,:,:] = ssm_z_res
    
X_test_ssm_acc = np.zeros((len(X_test),3,155,155))
for j in range(0,len(X_test)):
    x_dir = np.reshape(X_test[j,:,0],(155,1))
    ssm_x = squareform(pdist(x_dir, metric='euclidean'))
    ssm_x_res = ssm_x/np.max(ssm_x)
    
    y_dir = np.reshape(X_test[j,:,1],(155,1))
    ssm_y = squareform(pdist(y_dir, metric='euclidean'))
    ssm_y_res = ssm_y/np.max(ssm_y)
    
    z_dir = np.reshape(X_test[j,:,2],(155,1))
    ssm_z = squareform(pdist(z_dir, metric='euclidean'))
    ssm_z_res = ssm_z/np.max(ssm_z)
    
    X_test_ssm_acc[j,0,:,:] = ssm_x_res
    X_test_ssm_acc[j,1,:,:] = ssm_y_res
    X_test_ssm_acc[j,2,:,:] = ssm_z_res
    
X_train_ssm_gyro = np.zeros((len(X_train),3,155,155))
for j in range(0,len(X_train)):
    x_dir = np.reshape(X_train[j,:,3],(155,1))
    ssm_x = squareform(pdist(x_dir, metric='euclidean'))
    ssm_x_res = ssm_x/np.max(ssm_x)
    
    y_dir = np.reshape(X_train[j,:,4],(155,1))
    ssm_y = squareform(pdist(y_dir, metric='euclidean'))
    ssm_y_res = ssm_y/np.max(ssm_y)
    
    z_dir = np.reshape(X_train[j,:,5],(155,1))
    ssm_z = squareform(pdist(z_dir, metric='euclidean'))
    ssm_z_res = ssm_z/np.max(ssm_z)
    
    X_train_ssm_gyro[j,0,:,:] = ssm_x_res
    X_train_ssm_gyro[j,1,:,:] = ssm_y_res
    X_train_ssm_gyro[j,2,:,:] = ssm_z_res
    
X_test_ssm_gyro = np.zeros((len(X_test),3,155,155))
for j in range(0,len(X_test)):
    x_dir = np.reshape(X_test[j,:,3],(155,1))
    ssm_x = squareform(pdist(x_dir, metric='euclidean'))
    ssm_x_res = ssm_x/np.max(ssm_x)
    
    y_dir = np.reshape(X_test[j,:,4],(155,1))
    ssm_y = squareform(pdist(y_dir, metric='euclidean'))
    ssm_y_res = ssm_y/np.max(ssm_y)
    
    z_dir = np.reshape(X_test[j,:,5],(155,1))
    ssm_z = squareform(pdist(z_dir, metric='euclidean'))
    ssm_z_res = ssm_z/np.max(ssm_z)
    
    X_test_ssm_gyro[j,0,:,:] = ssm_x_res
    X_test_ssm_gyro[j,1,:,:] = ssm_y_res
    X_test_ssm_gyro[j,2,:,:] = ssm_z_res
    
X_train_final_acc, X_val_final_acc, Y_train_final_acc, Y_val_final_acc = train_test_split(X_train_ssm_acc, Y_train, test_size=0.2, random_state=1)
X_train_final_gyro, X_val_final_gyro, Y_train_final_gyro, Y_val_final_gyro = train_test_split(X_train_ssm_gyro, Y_train, test_size=0.2, random_state=1)

predictions1 = single_model(X_train_final_acc, X_val_final_acc, Y_train_final_acc, Y_val_final_acc,X_test_ssm_acc,Y_test,'alexnet')
predictions2 = single_model(X_train_final_gyro, X_val_final_gyro, Y_train_final_gyro, Y_val_final_gyro,X_test_ssm_gyro,Y_test,'alexnet')
predictions = np.asarray(predictions1+predictions2)
Y_pred_acc = np.argmax(np.asarray(predictions1),axis=1)
Y_pred_gyro = np.argmax(np.asarray(predictions2),axis=1)
Y_pred = np.argmax(np.asarray(predictions),axis=1)
print('alexnet : Accelerometer = ' + str(accuracy_score(Y_test,Y_pred_acc)) + ' Gyroscope = ' + str(accuracy_score(Y_test,Y_pred_gyro)) + ' Fused = ' + str(accuracy_score(Y_test,Y_pred)))
np.save('results/final_results/SSM/acc_alexnet.npy',Y_pred_acc)
np.save('results/final_results/SSM/gyro_alexnet.npy',Y_pred_gyro)
np.save('results/final_results/SSM/fused_alexnet.npy',Y_pred)

predictions1 = single_model(X_train_final_acc, X_val_final_acc, Y_train_final_acc, Y_val_final_acc,X_test_ssm_acc,Y_test,'googlenet')
predictions2 = single_model(X_train_final_gyro, X_val_final_gyro, Y_train_final_gyro, Y_val_final_gyro,X_test_ssm_gyro,Y_test,'googlenet')
predictions = np.asarray(predictions1+predictions2)
Y_pred_acc = np.argmax(np.asarray(predictions1),axis=1)
Y_pred_gyro = np.argmax(np.asarray(predictions2),axis=1)
Y_pred = np.argmax(np.asarray(predictions),axis=1)
print('googlenet : Accelerometer = ' + str(accuracy_score(Y_test,Y_pred_acc)) + ' Gyroscope = ' + str(accuracy_score(Y_test,Y_pred_gyro)) + ' Fused = ' + str(accuracy_score(Y_test,Y_pred)))
np.save('results/final_results/SSM/acc_googlenet.npy',Y_pred_acc)
np.save('results/final_results/SSM/gyro_googlenet.npy',Y_pred_gyro)
np.save('results/final_results/SSM/fused_googlenet.npy',Y_pred)
    


