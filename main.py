import os
import numpy as np
from sklearn import preprocessing

import torch
import torch.nn as nn
from torch import optim

from utils2 import DCCA_AM, loading_cv_data
import logging
from custom_optimizer import CustomeOptimizer
from meta_lr_model import Meta_LR_Model

import pickle

# meta learning configuration
# meta_num_train = 30 # Number of steps we will train the meta learner
meta_num_train = 5 # temp

# loading data
eeg_dir = './Data/eeg_data_sep/'
eye_dir = './Data/eye_data_sep/'
file_list = os.listdir(eeg_dir)
file_list.sort()

# design hyper-parameters
epochs = 70
eeg_input_dim = 310
eye_input_dim = 33
output_dim = 12
learning_rate = 5 * 1e-4
batch_size = 50

emotion_categories = 5
device = 'cuda:0' if torch.cuda.is_available() else torch.device("mps")
res_dir = './res/cv3/'
if not os.path.exists(res_dir):
    os.makedirs(res_dir)

cv = 3


if __name__ == "__main__":
    # this ensures that the current MacOS version is at least 12.3+
    print(torch.backends.mps.is_available())
    # this ensures that the current current PyTorch installation was built with MPS activated.
    print(torch.backends.mps.is_built())
    print(device)

    # preparing data
    for f_id in file_list:
        print(f"This is f id: {f_id}")
        logging.basicConfig(filename='./logs/cv3.log', level=logging.DEBUG)
        logging.debug('{}'.format(f_id))
        logging.debug('Task-Epoch-CCALoss-PredicLoss-PredicAcc')
        train_all, test_all = loading_cv_data(eeg_dir, eye_dir, f_id, cv)

        np.random.shuffle(train_all)
        np.random.shuffle(test_all)

        sample_num = train_all.shape[0]
        batch_number = sample_num // batch_size

        train_eeg = train_all[:, 0:310]
        train_eye = train_all[:, 310:343]
        train_label = train_all[:, -1]

        scaler = preprocessing.MinMaxScaler()
        train_eeg = scaler.fit_transform(train_eeg)
        train_eye = scaler.fit_transform(train_eye)

        test_eeg = test_all[:, 0:310]
        test_eye = test_all[:, 310:343]
        test_label = test_all[:, -1]

        test_eeg = scaler.fit_transform(test_eeg)
        test_eye = scaler.fit_transform(test_eye)


        if torch.cuda.is_available():
            train_eeg = torch.from_numpy(train_eeg).to(torch.float).to(device)
            train_eye = torch.from_numpy(train_eye).to(torch.float).to(device)
            test_eeg = torch.from_numpy(test_eeg).to(torch.float).to(device)
            test_eye = torch.from_numpy(test_eye).to(torch.float).to(device)
            train_label = torch.from_numpy(train_label).to(torch.long).to(device)
            test_label = torch.from_numpy(test_label).to(torch.long).to(device)
        elif torch.backends.mps.is_available():
            train_eeg = np.float32(train_eeg)
            train_eye = np.float32(train_eye)
            test_eeg = np.float32(test_eeg)
            test_eye = np.float32(test_eye)

            train_eeg = torch.from_numpy(train_eeg).to(device)
            train_eye = torch.from_numpy(train_eye).to(device)
            test_eeg = torch.from_numpy(test_eeg).to(device)
            test_eye = torch.from_numpy(test_eye).to(device)

            train_label = torch.from_numpy(train_label).to(torch.long).to(device)
            test_label = torch.from_numpy(test_label).to(torch.long).to(device)

        # training process
        for hyper_choose in range(100):
            best_test_res = {}
            best_test_res['acc'] = 0
            best_test_res['predict_proba'] = None
            best_test_res['fused_feature'] = None
            best_test_res['transformed_eeg'] = None
            best_test_res['transformed_eye'] = None
            best_test_res['alpha'] = None
            best_test_res['true_label'] = None
            best_test_res['layer_size'] = None
            # try 100 combinations of different hidden units
            layer_sizes = [np.random.randint(100,200), np.random.randint(20,50), output_dim]
            logging.info('{}-{}'.format(layer_sizes[0], layer_sizes[1]))
            mymodel = DCCA_AM(eeg_input_dim, eye_input_dim, layer_sizes, layer_sizes, output_dim, emotion_categories, device).to(device)
            optimizer_classifier = torch.optim.RMSprop(mymodel.parameters(), lr=learning_rate)
            optimizer_model1 = torch.optim.RMSprop(iter(list(mymodel.parameters())[0:8]), lr=learning_rate/2)
            optimizer_model2 = torch.optim.RMSprop(iter(list(mymodel.parameters())[8:16]), lr=learning_rate/2)
            class_loss_func = nn.CrossEntropyLoss()

            meta_lr_model = Meta_LR_Model().device()
            meta_model_opt = optim.Adam(meta_lr_model.parameters(), lr=1e-2)

            # Outer Loop
            for outer_loop_epoch in range(meta_num_train):

                # reset losses
                meta_loss = 0  # meta loss for the meta model

                # reset networks
                simple_nn = mymodel

                # maintain grad on the parameters of the simple nn
                for name, param in simple_nn.named_buffers():
                    param.retain_grad()

                # reset optimizer
                opt = CustomeOptimizer(simple_nn)

                # Inner Loop
                for epoch in range(epochs):
                    mymodel.train()
                    best_acc = 0
                    total_classification_loss = 0
                    for b_id in range(batch_number+1):
                        if b_id == batch_number:
                            train_eeg_used = train_eeg[batch_size*batch_number:, :]
                            train_eye_used = train_eye[batch_size*batch_number:, :]
                            train_label_used = train_label[batch_size*batch_number:]
                        else:
                            train_eeg_used = train_eeg[b_id*batch_size:(b_id+1)*batch_size, :]
                            train_eye_used = train_eye[b_id*batch_size:(b_id+1)*batch_size, :]
                            train_label_used = train_label[b_id*batch_size:(b_id+1)*batch_size]

                        # predict_out, cca_loss, output1, output2, partial_h1, partial_h2, fused_tensor, transformed_1, transformed_2, alpha  = mymodel(train_eeg_used, train_eye_used)
                        predict_out, cca_loss, output1, output2, partial_h1, partial_h2, fused_tensor, alpha  = mymodel(train_eeg_used, train_eye_used)
                        predict_loss = class_loss_func(predict_out, train_label_used)

                        optimizer_model1.zero_grad()
                        optimizer_model2.zero_grad()
                        optimizer_classifier.zero_grad()

                        partial_h1 = torch.from_numpy(partial_h1).to(torch.float).to(device)
                        partial_h2 = torch.from_numpy(partial_h2).to(torch.float).to(device)

                        output1.backward(-0.1*partial_h1, retain_graph=True)
                        output2.backward(-0.1*partial_h2, retain_graph=True)
                        predict_loss.backward()

                        optimizer_model1.step()
                        optimizer_model2.step()
                        optimizer_classifier.step()

                        ##### Meta Learning #####

                        fc_loss = class_loss_func(predict_out, target)

                        # add losses to meta loss for stronger signal
                        meta_loss += fc_loss

                        # get an input to the meta model. We will use the actual loss as the input
                        meta_model_input = torch.tensor([fc_loss.item()]).to(
                            device)  # note that we want to remove the gradient from this as this is just an input

                        # get new lrs based on the activation percentages
                        meta_output_lr = meta_lr_model(meta_model_input)

                        opt.zero_grad()
                        class_loss_func.backward(retain_graph=True)
                        opt.step(meta_output_lr)
                        ##### Meta Learning #####

                    # for every epoch, evaluate the model on both train and test set
                    mymodel.eval()
                    predict_out_train, cca_loss_train, _, _, _, _, _, _  = mymodel(train_eeg, train_eye)
                    predict_loss_train = class_loss_func(predict_out_train, train_label)
                    accuracy_train = np.sum(np.argmax(predict_out_train.detach().cpu().numpy(), axis=1) == train_label.detach().cpu().numpy()) / predict_out_train.shape[0]

                    predict_out_test, cca_loss_test, output_1_test, output_2_test, _, _, fused_tensor_test, attention_weight_test  = mymodel(test_eeg, test_eye)
                    predict_loss_test = class_loss_func(predict_out_test, test_label)
                    accuracy_test = np.sum(np.argmax(predict_out_test.detach().cpu().numpy(), axis=1) == test_label.detach().cpu().numpy()) / predict_out_test.shape[0]

                    if accuracy_test > best_test_res['acc']:
                        best_test_res['acc'] = accuracy_test
                        best_test_res['layer_size'] = layer_sizes
                        best_test_res['predict_proba'] = predict_out_test.detach().cpu().data
                        best_test_res['fused_feature'] = fused_tensor_test
                        best_test_res['transformed_eeg'] = output_1_test.detach().cpu().data
                        best_test_res['transformed_eye'] = output_2_test.detach().cpu().data
                        best_test_res['alpha'] = attention_weight_test
                        best_test_res['true_label'] = test_label.detach().cpu().data

                    print('Epoch: {} -- Train CCA loss is: {} -- Train loss: {} -- Train accuracy: {}'.format(epoch, cca_loss_train, predict_loss_train.data, accuracy_train))
                    print('Epoch: {} -- Test CCA loss is: {} -- Test loss: {} -- Test accuracy: {}'.format(epoch, cca_loss_test, predict_loss_test.data, accuracy_test))
                    print('\n')
                    logging.info('\tTrain\t{}\t{}\t{}\t{}'.format(epoch, cca_loss_train, predict_loss_train.data, accuracy_train))
                    logging.info('\tTest\t{}\t{}\t{}\t{}'.format(epoch, cca_loss_test, predict_loss_test.data, accuracy_test))

                pickle.dump(best_test_res, open( os.path.join(res_dir, f_id[:-8]+'_'+str(hyper_choose)), 'wb'  ))

            ######################################################################################################################################################################################################
            # Compute meta loss
            ######################################################################################################################################################################################################\

            forgetfulness_loss = meta_loss

            meta_model_opt.zero_grad()
            forgetfulness_loss.backward()
            torch.nn.utils.clip_grad_norm_(meta_lr_model.parameters(), 1e-1)
            meta_model_opt.step()

            ######################################################################################################################################################################################################
            # test
            ######################################################################################################################################################################################################\
            with torch.no_grad():
                test_loss = []
                test_accuracy = []

                for batch_idx, (x, target) in enumerate(mnist_test_loader):
                    x, target = x.to(device), target.to(device)
                    outputs = simple_nn(x)
                    _, predicted = torch.max(outputs.data, 1)
                    batch_test_loss = criterion(outputs, target)
                    test_loss.append(batch_test_loss.item())
                    test_accuracy.append((predicted == target).sum().item() / predicted.size(0))

            ######################################################################################################################################################################################################
            # appending stuff (last batch)
            ######################################################################################################################################################################################################\
            print('test loss: {}, test accuracy: {}'.format(np.mean(test_loss), np.mean(test_accuracy)))
            print('meta_loss', meta_loss.item())
            print(f'current predicted learning rate is: {meta_output_lr.item()}')

            test_losses = []
            meta_losses = []
            predicted_lrs = []
            test_losses.append(np.mean(test_loss))
            meta_losses.append(meta_loss.item())
            predicted_lrs.append(meta_output_lr.item())
            print()
