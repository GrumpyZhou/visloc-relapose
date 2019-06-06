import torch
import torch.nn as nn
from collections import OrderedDict

class BaseNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.device = config.device

    @property
    def num_params(self):
        return sum([p.numel() for p in self.parameters()])

      
    def forward(self, *inputs):
        """Defines the computation performed at every call.
        Inherited from Superclass torch.nn.Module.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError
    
    def get_inputs_(self, batch, **kwargs):
        """Define how to parse the data batch for the network training/testing.
        This allows the flexibility for different training data formats in different applications.
        Should be overridden by all subclasses.
        """
        raise NotImplementedError
        
    def init_weights_(self):
        """Define how to initialize the weights of the network.
        Should be overridden by all subclasses, since it 
        normally differs according to the network models.
        """
        raise NotImplementedError
    
    def loss_(self, batch):        
        """Define how to calculate the loss  for the network.
        Should be overridden by all subclasses, since different 
        applications or network models may have different types 
        of targets and the corresponding criterions to evaluate 
        the predictions.
        """
        raise NotImplementedError
     
    def set_optimizer_(self, config):
        if not config.training:
            return
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=config.lr_init, 
                                          weight_decay=config.weight_decay)
        if config.optimizer_dict:
            self.optimizer.load_state_dict(config.optimizer_dict)
        if config.lr_decay_factor < 1 and config.lr_decay_step > 0:
            self.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 
                                                                step_size=config.lr_decay_step, 
                                                                gamma=config.lr_decay_factor, 
                                                                last_epoch=config.start_epoch-1)
        else:
            self.lr_scheduler = None    
        
    def predict_(self, batch):
        return self.forward(*self.get_inputs_(batch, with_label=False))
    
    def optim_step_(self, batch):
        self.optimizer.zero_grad()
        loss = self.loss_(batch)
        loss.backward()
        self.optimizer.step()
        return loss
    
    def train_epoch(self, data_loader, epoch):
        if self.lr_scheduler:
            self.lr_scheduler.step()
        for i, batch in enumerate(data_loader):
            loss = self.optim_step_(batch)
        return loss
    
    def save_weights_(self, sav_path):
        torch.save(self.state_dict(), sav_path)
    
    def print_(self):
        for k, v in self.state_dict().items():
            print(k, v.size())

    def kaiming_normal_init_func_(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.kaiming_normal_(m.weight.data, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:  # Incase bias is turned off
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.constant_(m.weight.data, 1.0)
            nn.init.constant_(m.bias.data, 0.0)
                
    def xavier_init_func_(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            nn.init.xavier_uniform_(m.weight.data)
            nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    def normal_init_func_(self, m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0.0)

    def fixed_weighting_loss(self, loss_pos, loss_rot, beta=1.0):
        '''The weighted loss function with beta as a hyperparameter to balance the positional loss and the rotational loss'''
        return loss_pos + beta * loss_rot

    def learned_weighting_loss(self, loss_pos, loss_rot, sx, sq):
        '''The weighted loss function that learns variables sx and sy to balance the positional loss and the rotational loss'''
        return (-1 * sx).exp() * loss_pos + sx + (-1 * sq).exp() * loss_rot + sq

