from typing import Any
import torch
import torch.nn as nn
from src.model.layers import MLP,FeatureEmbedding,FM_Linear,FM_Interaction
#lightning
import pytorch_lightning as pl

class FactorizationMachine(pl.LightningModule):
    def __init__(self, args, field_dims):
        super(FactorizationMachine, self).__init__()
        self.embedding=FeatureEmbedding(args,field_dims)
        self.linear=FM_Linear(args,field_dims)
        self.interaction=FM_Interaction(args,field_dims)
        self.bceloss=nn.BCEWithLogitsLoss() # since bcewith logits is used, we don't need to add sigmoid layer in the end
        self.lr=args.lr

    def loss(self, y_pred, y_true,c_values):
        # calculate weighted mse with l2 regularization
        #mse = (y_pred - y_true.float()) ** 2
        bce = self.bceloss(y_pred, y_true.float())
        weighted_bce = c_values * bce
        #l2_reg = torch.norm(self.w)**2 + torch.norm(self.v)**2
        return torch.mean(weighted_bce) 
    
    def forward(self, x,x_cont):
        # FM part loss with interaction terms
        # x: batch_size * num_features
        lin_term = self.linear(x,x_cont)
        embedding=self.embedding(x)

        inter_term = self.interaction(embedding,x_cont)
        x= lin_term + inter_term
        x=x.squeeze(1)
        return x

    
    def training_step(self, batch, batch_idx):
        x,x_cont,y,c_values=batch
        y_pred=self.forward(x,x_cont)
        loss_y=self.loss(y_pred, y,c_values)
        self.log('train_loss', loss_y, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss_y
    
    def configure_optimizers(self) ->Any:

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    

    





    
