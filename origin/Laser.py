import numpy as np
import argparse 
import os
import logging 
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import ipdb

from pandas import DataFrame
from sklearn.model_selection import train_test_split


def parse_args(args=None):
    parser = argparse.ArgumentParser(
            description = 'Training and Testing Prediction Models',
            usage = 'train.py [<args>] [-h | --help]'
    )
    
    parser.add_argument('-data_path', type = str, default = 'data/')
    parser.add_argument('-save_path', type = str, default = 'model/')
    parser.add_argument('-step', default=16, type=int)
    parser.add_argument('-square_size', default=30, type=int)
    parser.add_argument('-board_num', default=9, type=int)
    parser.add_argument('-board_size', default=1024, type=int)
    parser.add_argument('-learning_rate', default=1e-5, type=float)
    parser.add_argument('-batch_size', default=500, type=int)
    parser.add_argument('--epoch', default=100, type=int)
    parser.add_argument('-order', default=0, type=int)
    parser.add_argument('-test', action = 'store_true')
    
    return parser.parse_args(args)

def override_config(args):
    pass

def save_model(model, optimizer, save_variable_list,args):
    pass

def set_logger(args):
    '''
    Write logs to checkpoint and console
    '''
    
    
    log_file = os.path.join(args.save_path, 'train_'+str(args.order)+'.log')
    
    logging.basicConfig(
            format = '%(asctime)s %(levelname)-8s %(message)s',
            level=logging.INFO,
            datefmt='%Y-%m-%d %H:%M:%S',
            filename=log_file,
            filemode ='w'
    )
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('').addHandler(console)
    
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        #input_size: (1,50,50) Channel = 1, H = args.square_size, W = args.square_size
        self.channal_first_layer = 16
        self.conv1 = nn.Conv2d(1,self.channal_first_layer,3,padding=1)
        self.conv2 = nn.Conv2d(self.channal_first_layer,
                               self.channal_first_layer*2,3,padding=1)
        #self.conv3 = nn.Conv2d(self.channal_first_layer*2,
        #                       self.channal_first_layer*4,3,padding=1)
        
        self.deconv = nn.ConvTranspose2d(self.channal_first_layer*2,1,3,padding=1)
        #self.dropout = nn.Dropout(0.25)
        self.laser_W = nn.Parameter(torch.zeros(5,1)) #(5,self.channal_first_layer)
        def forward(self,x,laser_parameter):
            #in_size = x.size(0)
            x = x.reshape(x.shape[0],-1,x.shape[1],x.shape[1]).float()
            #effect = laser_parameter.dot(self.laser_W)
            effect = laser_parameter.reshape(-1,5).mm(self.laser_W)
            out = self.conv1(x)
            out = F.relu(out)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            effect = effect.reshape(out.shape[0],1,1,1) + torch.zeros(out.shape[0],
                                   out.shape[1],out.shape[2],out.shape[3]).to(device)
            out = out*effect
            out = self.conv2(out)
            out = F.relu(out)
            #out = self.conv3(out)
            #out= self.F.relu(out)
            
            out = self.deconv(out)
            out = F.relu(out)
            out = x-out
            return out
        
def loss_function(data,target,test = False):
    loss = torch.mean(torch.norm((data-target).reshape(len(data),-1),dim=1))
    if test==True:
        global norm_min, norm_max
        data = data*(norm_max - norm_min) + norm_min
        target = target*(norm_max - norm_min) + norm_min
        practice_distance = torch.mean(torch.abs(data-target))
        return loss, practice_distance
    else:
        return loss
    
def train(model, device, train_data, train_result, optimizer, epoch, batch_size, square_size):
    model.train()
    train_loader = torch.utils.data.DataLoader(np.concatenate([train_data, train_result],
    axis=2), batch_size = batch_size, shuffle=True)
    for batch_idx, batch_data in enumerate(train_loader):
        batch_data = batch_data.to(device)
        data, target = batch_data[:,:,:batch_data.shape[2]//2],batch_data[:,:,batch_data.shape[2]//2:]
        laser_parameter = data[:,square_size,:5].float().reshape(-1)
        data = data[:,:square_size,:]
        target = target[:,:square_size,:]
        data_v = torch.autograd.Variable(data)
        target_v = torch.autograd.Variable(target)
        optimizer.zero_grad()
        output = model(data_v,laser_parameter)
        loss = loss_function(output,target)
        loss.backward()
        optimizer.step()
        if(batch_idx+1)%100 == 0:
            logging.info('Train Epoch: {} [{} ({:.0f}%)]\tLoss: {:.6f}'.format
                         (
                           epoch, batch_idx * len(data),
                           100. * batch_idx / len(train_loader),
                           loss.item()/batch_size))
            
def test(model, device, test_data, test_result, epoch, square_size, order, save=True):
    global test_loss_min
    model.eval()
    test_loss = 0
    practice_distance = 0
    test_loader = torch.utils.data.DataLoader(np.concatenate([test_data,test_result],axis=2),batch_size=1,
                                              shuffle = True)
    with torch.no_grad():
        for batch_data in test_loader:
            batch_data = batch_data.to(device)
            data, target = batch_data[:,:,:batch_data.shape[2]//2],batch_data[:,:,batch_data.shape[2]//2:]
            laser_parameter = data[:,square_size,:5].float().reshape(-1)
            data = data[:,:square_size,:]
            target = target[:,:square_size,:]
            output = model(data,laser_parameter)
            loss_cur, practice_distance_cur = loss_function(output,target,test=True)
            test_loss += loss_cur
            practice_distance += practice_distance_cur
    test_loss /= len(test_data)
    practice_distance /= len(test_data)
    logging.info('EPOCH:{}    test test: Average loss: {:.4f}'.format(epoch,test_loss))
    logging.info('EPOCH:{}    test test: Practice distance: {:.4f}'.format(epoch,practice_distance))
    if test_loss < test_loss_min and save == True:
        test_loss_min = test_loss
        print("saving model ...")
        torch.save(model.state_dict(),'laser_model_' + str(order) + '.pt')

def nan_average(samples):
    df = DataFrame(samples)
    df = df.fillna(method='ffill')
    df = df.fillna(method='bfill')
    return df.values

def main(args):
    #Write logs to checkpoint and console
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    set_logger(args)
    
    before_exposure = []
    after_exposure = []

    for i in range (1, args.board_num+1):
        file1 = np.loadtxt(args.data_path + 'board'+str(i)+'_1.txt', usecols=2)
        before_exposure.append( nan_average(np.reshape(file1,(args.board_size,-1))))
        file1 = np.loadtxt(args.data_path + 'board'+str(i)+'_2.txt', usecols=2)
        after_exposure.append( nan_average(np.reshape(file1,(args.board_size,-1))))
        
    before_exposure = np.concatenate([before_exposure])
    after_exposure = np.concatenate([after_exposure])
    
    #Normalization:range(0,1)
    global norm_min, norm_max
    norm_min = np.min(np.concatenate([before_exposure,after_exposure]))
    norm_max = np.max(np.concatenate([before_exposure,after_exposure]))
    before_exposure = (before_exposure - norm_min ) / (norm_max - norm_min)
    after_exposure = (after_exposure - norm_min ) / (norm_max -norm_min) 
    
    logging.info('Data Path: %S' % args.data_path)
    logging.info('Square size %d' % args.square_size)
    logging.info('Step: %d' % args.step)
    logging.info('Board Total number: %d' % args.board_num)
    logging.info('Learning Rate: %f' % args.learning_rate)
    logging.info('Batch size: %f' % args.batch_size)
    
    file1 = np.loadtxt(args.data_path + "para.txt", delimiter=',')
    args.laser_parameter = file1[:args.board_num,:]
    
    before_exposure_samples = []
    after_exposure_samples = []
    indices = np.arange(0,int(args.board_size-args.step),int(args.step))
    for board in range(0,args.board_num):
        for i in indices:
            if i+args.square_size>=args.board_size:
                continue
            for j in indices:
                if j+args.square_size>=args.board_size:
                    continue
                temp1 = np.pad(args.laser_parameter[board,:],(0,args.square_size-5),
                               'constant', constant_value=0)
                temp = np.vstack([before_exposure[board,i:i+args.square_size,j:j+args.square_size],temp1])
                before_exposure_samples.append(temp)
                after_exposure_samples.append(np.vstack([after_exposure[board,i:i+args.square_size,j:j+args.square_size],temp1]))
                
    before_exposure_samples = np.concatenate([before_exposure_samples])
    after_exposure_samples = np.concatenate([after_exposure_samples])
    
    train_data, test_data, train_result, test_result = train_test_split(before_exposure_samples,
                                                                        after_exposure_samples,
                                                                        test_size = 0.33)
    logging.info('Total sample: %d' % len(before_exposure_samples))
    logging.info('train: %d' % len(train_data))
    logging.info('test: %d' % len(test_data))
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    global test_loss_min
    test_loss_min = 300
    model = ConvNet().to(DEVICE)
    
    if args.test==True:
        model.load_state_dict(torch.load('laser_model_' + str(args.order)+'.pt'))
        test(model,DEVICE,test_data,test_result,0,args.square_size,args.order,save=False)
    else:
        optimizer = torch.optim.Adam(model.parameters(),lr = args.learning_rate)
        
        for epoch in range(1, args.epoch):
            if epoch == 30:
                optimizer = torch.optim.Adam(model.parameters(),lr = args.learning_rate/10)
            if epoch == 60:   
                optimizer = torch.optim.Adam(model.parameters(),lr = args.learning_rate/100)
            train(model,DEVICE,train_data,train_result,optimizer,epoch,args.batch_size,args.square_size)
            test(model,DEVICE,test_data,test_result,epoch,args.square_size,args.order)


if __name__ == '__main__':
    main(parse_args())      
        
        
        
        
        
    
    