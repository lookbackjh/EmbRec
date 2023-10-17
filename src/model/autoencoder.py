from typing import Any
from pytorch_lightning.utilities.types import OptimizerLRScheduler
import torch
import torch.nn as nn 
# lightning
import pytorch_lightning as pl

class AutoEncoder(pl.LightningModule):

    def __init__(self, input_dim, output_dim, args):
        super(AutoEncoder, self).__init__()
        #here, input can be either user or item
        self.input_dim = input_dim #which is recommended to be one-hot encoded 
        self.output_dim = output_dim #a vector of each user or item
        self.args=args


        self.encoder=nn.Sequential(
            nn.Linear(self.input_dim,100),


            nn.ReLU(),
            nn.Linear(100, self.args.k),
            nn.ReLU(),
        )
        self.decoder=nn.Sequential(
            nn.Linear(self.args.k, 100),
            nn.ReLU(),
            nn.Linear(100, self.output_dim),

        )

    def l1_regularization(self, x):
        for params in self.parameters():
            l1_reg = torch.norm(params, 2)
        return l1_reg


    def encode(self, x):
        x=self.encoder(x)
        return x
    

    def forward(self, x):
        x=self.encoder(x)
        x=self.decoder(x)
        return x

        #loss function
    def bceloss(self, y_hat, y):
        loss = nn.BCEWithLogitsLoss()

        return loss(y_hat, y)

    def training_step(self, batch, batch_idx):
        x,y=batch
        
        y_hat=self.forward(x)
        #batch size * item_number

        loss=self.bceloss(y_hat, y)
        reg=self.l1_regularization(x)

        total_loss=loss+0.01*reg

        self.log('train_loss', total_loss, on_epoch=True, prog_bar=True, logger=True)
        return total_loss
    
    #ADAM optimizer
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.args.auto_lr, weight_decay=self.args.weight_decay)
        return optimizer
    






