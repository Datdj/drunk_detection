import torch
import os
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

def score_metrics(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    F1_score = f1_score(pred_flat, labels_flat, average='macro')
    accuracy = accuracy_score(pred_flat, labels_flat)
    return accuracy, F1_score

class Trainer():
    def __init__(self, model, trainloader, validloader, num_epochs, optimizer, loss_function):
        self.model = model
        self.trainloader = trainloader
        self.validloader = validloader
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.loss_function = loss_function

    def train(self, save_dir):
        best_val_f1 = 0
        for epoch in range(self.num_epochs):
            print('======== Epoch {:} / {:} ========'.format(epoch + 1, self.num_epochs))
            print('Training...')
            self.train_one_epoch(epoch)
            avg_val_loss, avg_val_acc, avg_val_f1 = self.evaluate()
            if best_val_f1 < avg_val_f1:
                best_val_f1 = avg_val_f1
                model_name = 'lstm_{}.pt'.format(best_val_f1)
                _path = os.path.join(save_dir, model_name)
                torch.save(self.model.state_dict(), _path)
                print(f'Model saved to ==> {_path}')
        return

    def train_one_epoch(self, epoch):
        self.model.train()
        running_loss = 0.0
        total_loss = 0.0
        running_accuracy = 0.0
        train_accuracy = 0.0
        train_f1 = 0.0
        for i, data in enumerate(self.trainloader):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # TODO:
            # - change input to float32 when generate data

            # forward + backward + optimize
            outputs = self.model(inputs.type(torch.float32))
            # labels = self.binary_to_one_hot(labels) # convert labels into one hot format
            loss = self.loss_function(outputs, labels.type(torch.long))

            # Compute accuracy, f1-score
            logits = outputs.detach().cpu().numpy()
            labels_ = labels.to('cpu').numpy()
            tmp_train_accuracy, tmp_train_f1 = score_metrics(logits, labels_)
            train_accuracy += tmp_train_accuracy
            running_accuracy += tmp_train_accuracy
            train_f1 += tmp_train_f1

            loss.backward()
            self.optimizer.step()

            # print statistics
            running_loss += loss.item()
            total_loss += loss.item()
            interval = 20
            if i % interval == interval - 1:
                print("[TRAIN] Epoch {}/{} | Batch {}/{} | Train Loss={} | Train Acc={}".format(epoch + 1, \
                self.num_epochs, i + 1, len(self.trainloader), running_loss / interval, running_accuracy / interval))
                running_loss = 0.0
                running_accuracy = 0.0
        print(" Train Loss: {0:.4f}".format(total_loss/len(self.trainloader)))
        print(" Train Accuracy: {0:.4f}".format(train_accuracy/len(self.trainloader)))
        print(" Train F1 score: {0:.4f}".format(train_f1/len(self.trainloader)))
        
    def binary_to_one_hot(self, y):
        n = y.shape[0]
        one_hot = torch.zeros(n, 2)
        one_hot[:, 0][y == 1] = 1
        one_hot[:, 1][y == 0] = 1
        return one_hot

    def evaluate(self):
        print("Running Validation...")
        self.model.eval()
        eval_loss = 0.0
        eval_accuracy = 0.0
        eval_f1 = 0.0 
        for batch in self.validloader:
            inputs, labels = batch
            with torch.no_grad():
                outputs = self.model(inputs.type(torch.float32))
                loss = self.loss_function(outputs, labels.type(torch.long))
                # Compute accuracy, f1-score
                logits = outputs.detach().cpu().numpy()
                labels_ = labels.cpu().numpy()
                tmp_eval_accuracy, tmp_eval_f1 = score_metrics(logits, labels_)
                
                eval_accuracy += tmp_eval_accuracy
                eval_f1 += tmp_eval_f1
                eval_loss += loss.item()
        avg_val_loss = eval_loss / len(self.validloader)
        avg_val_acc = eval_accuracy / len(self.validloader)
        avg_val_f1 = eval_f1 / len(self.validloader)
        print(" Valid Loss: {0:.4f}".format(avg_val_loss))
        print(" Valid Accuracy: {0:.4f}".format(avg_val_acc))
        print(" Valid F1 score: {0:.4f}".format(avg_val_f1))
        return avg_val_loss, avg_val_acc, avg_val_f1