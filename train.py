from optparse import OptionParser
import os
import sys

import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from data_preparation.prepare_data import import_data
from model import make_loaders, CNN, MultiClassifier, LSTMattn, eval_batch


def train_multitask(model, train_loader, test_loader, dir_checkpoint, dir_writer, 
                    epochs=10, batch_size=4, lr=1e-5, save_cp=True, gpu=False):
    
    writer = SummaryWriter(dir_writer)
    
    print(f'''Start training: 
                Epocs = {epochs}
                Batch size = {batch_size}
                Learning rate = {lr} 
                Training size = {train_loader.dataset.__len__()}
                Validation size = {test_loader.dataset.__len__()}
                CUDA = {gpu}
          ''')
   
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)

    for epoch in range(epochs):

        print('Starting epoch {}/{}.'.format(epoch + 1, epochs))
        model.train()
        epoch_loss = 0

        for i, sample_batch in enumerate(train_loader):

            inputs, labels = sample_batch['sequence'], sample_batch['label'] 
            if gpu:
                inputs = inputs.cuda()
                labels = labels.cuda()

            outputs = model(inputs)
            # multi-task loss
            loss = 0
            for lx in range(len(outputs)):
                loss += criterion(outputs[lx], labels[:, lx])
            epoch_loss += loss.item()

            if i%10 == 0:
                print(f'epoch = {epoch+1:d}, iteration = {i:d}/{len(train_loader):d}, loss = {loss.item():.5f}')
                writer.add_scalar('train_loss_iter', loss.item(), i + len(train_loader) * epoch)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch finished ! Loss: {epoch_loss/i}')
        # training set accuracy
        accuracy = eval_batch(model, train_loader, n_labels=len(targets), gpu=gpu)
        print(f'Accuracy = {accuracy}')
        writer.add_scalars('accuracy', {f'label_{i}':a for i,a in enumerate(accuracy)}, len(train_loader) * (epoch+1))
        
        if save_cp:
            torch.save(model.state_dict(), dir_checkpoint + 'CP{}.pth'.format(epoch + 1))
            print('Checkpoint {} saved !'.format(epoch + 1))

        
    writer.close()
    

def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=10, type='int',
                      help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batchsize', default=4,
                      type='int', help='batch size')
    parser.add_option('-l', '--learning-rate', dest='lr', default=1.e-5,
                      type='float', help='learning rate')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu',
                      default=False, help='use cuda')
    parser.add_option('-c', '--load', dest='load',
                      default=False, help='load file model')

    (options, args) = parser.parse_args()
    return options


if __name__ == '__main__':
    
    data_dir = '<path-to-dataset>'
    writer_dir = './runs/experiment_1'
    checkpoint_dir = './checkpoints/experiment_1/'
    os.makedirs(checkpoint_dir, exist_ok=True)

    channels = ['CP', 'FS1', 'PS1', 'PS2', 'PS3', 'PS4', 'PS5', 'SE', 'VS1']
    targets = [0,1,2,3]
    sequence = 50
    
    args = get_args()

    # CNN model
#     model = CNN(sequence, input_dim=len(channels)) 
    # LSTM model
    model = LSTM(len(channels), hidden_dim=20, num_layers=1)
    # add multi-task classifier
    model.classifier = MultiClassifier(model.classifier.in_features)
    
    if args.load:
        model.load_state_dict(torch.load(args.load))
        print('Model loaded from {}'.format(args.load))

    if args.gpu:
        model.cuda()
        
    print('Preparing data')
    data = import_data(data_dir, sequence)
    train_loader, test_loader = make_loaders(data, channels, targets)
    
    try:
        train_multitask(model, train_loader, test_loader, checkpoint_dir, writer_dir, 
                        epochs=args.epochs, batch_size=args.batchsize, lr=args.lr, gpu=args.gpu)
        
        accuracy = eval_batch(model, test_loader, n_labels=4, gpu=args.gpu)
        print(f'Test set accuracy = {accuracy}')
    
    except KeyboardInterrupt:
        torch.save(model.state_dict(), 'INTERRUPTED.pth')
        print('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)


        
