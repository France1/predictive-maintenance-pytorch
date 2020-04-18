import torch

def eval_batch(model, loader, n_labels=1, gpu=False):
    """Calculate model accuracy   
    
    Inputs:
        model : object
            pytorch trained model
        loader: object
            pytorch dataloader
        n_labels: int
            number of target labels that model predicts
        gpu: bool
            specify if model runs on GPU
            
    Outputs:
        accuracy: list[float]
            accuracy of model classification for each target    
    """
    
    model.eval()
    
    corrects = torch.zeros([n_labels], dtype=torch.float)
    counts = 0
    
    for i, sample_batch in enumerate(loader):

        inputs, labels = sample_batch['sequence'], sample_batch['label'] 
        if gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()
                
        outputs = model(inputs)    

        preds = ()
        for lx in range(len(outputs)):
            _, pred = torch.max(outputs[lx], 1)
            preds += (pred.view(-1,1),)

        preds = torch.cat(preds,1)
        if gpu:
            preds = preds.cuda()
            corrects = corrects.cuda()
        
        corrects += (preds == labels).float().sum(0)
        counts += labels.shape[0]

    accuracy = corrects/counts
    accuracy = [round(i.item(),2) for i in accuracy]
    
    return accuracy