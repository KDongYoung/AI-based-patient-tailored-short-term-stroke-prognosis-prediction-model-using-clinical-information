import torch
import pickle

# EVALUATION
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelBinarizer     
from Utils.Interpretation import XAI

class Trainer():
    def __init__(self, args, model, model_type):
        self.args=args
        self.model=model
        self.model_type=model_type    

        self.lb = LabelBinarizer()
        self.lb.fit(range(self.args['n_classes']))
        
        self.model_pkl_file=''
    
    '''
    ###########################################################################################
    #  Train 
    ###########################################################################################
    '''
    
    def train(self,  train_loader, fold):
        """        train model        """
        for x_train, y_train in train_loader:
            self.model.fit(x_train, y_train) # train

        # save model by each fold
        self.model_pkl_file=f"{self.args['total_path']}/{self.args['model_name']}_{self.args['normalize']}_{self.args['target']}_{fold}fold.pkl"
        with open(self.model_pkl_file, 'wb') as file:  
            pickle.dump(self.model, file)
            
    '''
    ###########################################################################################
    #  Evaluation
    ###########################################################################################
    '''    
    ## EVALUATE 
    def eval(self, phase, loader, fold):
        loss=torch.tensor(0)
        
        targets=[]
        preds=[]
        total_data=torch.tensor([])
        
        with torch.no_grad(): 
            for datas in loader:
                data, target = datas[0], datas[1].to(dtype=torch.int64)
                
                total_data=torch.cat([total_data, data])
                pred = self.model.predict(data) # predict    
                preds.append(torch.tensor(pred)) 
                targets.append(target)
        
        XAI(self.model, total_data.cpu().numpy(), fold, self.args["selected_feature_name"], self.args["save_root"] + str(self.args['seed']))
        
        preds=torch.cat(preds)
        targets=torch.cat(targets)
                    
        targets=targets.numpy()
        preds=preds.numpy()
    

        acc=accuracy_score(targets, preds)
        f1=f1_score(targets,preds, average='macro')
        auroc=roc_auc_score(self.lb.transform(targets), self.lb.transform(preds), average='macro', multi_class='ovr')

        print(phase.capitalize(),'Loss: {:.4f}, Acc: {:.4f}%, F1: {:.4f}, AUROC: {:.4f}'.format(loss, acc, f1, auroc))
        
        return loss.item(), acc, f1, auroc
    