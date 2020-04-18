import torch
import torch.nn as nn

def train_singletask(model, train_loader, test_loader, epochs=10, batch_size=4, lr=1e-5, gpu=False):
    """
    Training routine for single task model configuration
    """
        
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    for i, sample_batch in enumerate(train_loader):
        
        inputs, labels = sample_batch['sequence'], sample_batch['label'] 
        if gpu:
            inputs = inputs.cuda()
            labels = labels.cuda()

        outputs = model(inputs) 
        
        _, predicted = torch.max(outputs 1)
        correct += (predicted == labels).float().sum()

        loss = criterion(outputs, labels)
        epoch_loss += loss.item()
        
        if i%10 == 0:
            print(f'epoch = {epoch+1:d}, iteration = {i:d}/{len(train_loader):d}, loss = {loss.item():.5f}')
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    print(f'Epoch {epoch} finished! Loss: {epoch_loss/i}. Accuracy: {correct.item()/len(train_dataset)}')