import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any

import torch
from src.model.fm import FactorizationMachine
from src.model.layers import FeatureEmbedding, FeatureEmbedding, FM_Linear, MLP



class DeepFM(pl.LightningModule):
    def __init__(self, args,field_dims):
        super(DeepFM, self).__init__()
        self.linear = FM_Linear(args,field_dims)
        self.fm = FactorizationMachine(args, field_dims)
        self.embedding = FeatureEmbedding(args, field_dims)
        if args.embedding_type=='SVD':
            self.embed_output_dim = (len(field_dims) +1)* args.emb_dim
        else:
            self. embed_output_dim = len(field_dims) * args.emb_dim
        #self.embed_output_dim = (len(field_dims) +1)* args.emb_dim
        self.mlp = MLP(args, self.embed_output_dim)
        self.bceloss=nn.BCEWithLogitsLoss() # since bcewith logits is used, we don't need to add sigmoid layer in the end
        self.lr=args.lr
        self.args=args

    def mse(self, y_pred, y_true):
        return self.bceloss(y_pred, y_true.float())


    def deep_part(self, x):
        return self.mlp(x)
    

    def l2norm(self):
        
        for param in self.model.parameters():
            param.data = param.data / torch.norm(param.data, 2)
        

    def loss(self, y_pred, y_true, c_values):
        mse =self.bceloss(y_pred, y_true.float())
        #bce=self.bceloss(y_pred,y_true.float())

        weighted_bce = c_values * mse
        #l2_reg = torch.norm(self.w) + torch.norm(self.v) # L2 regularization

        loss_y=torch.mean(weighted_bce) 
        
        return loss_y
    

    def forward(self, x,x_cont):
        # FM part, here, x_hat means another arbritary input of data, for combining the results. 
        fm_part=self.fm(x, x_cont)
        embed_x=self.embedding(x)
        if self.args.embedding_type=='SVD':
            x_cont=x_cont.unsqueeze(1)
            new_x=torch.cat((embed_x,x_cont),1)
            new_x=new_x.view(-1, self.embed_output_dim)
        else:
            new_x=embed_x.view(-1, self.embed_output_dim)

        deep_part=self.mlp(new_x)
        #x=x.float()
        
        # Deep part

        #deep_out=self.sig(deep_out)
        y_pred=fm_part+deep_part.squeeze()
       
        return y_pred

    def training_step(self, batch, batch_idx):
        x,x_cont,y,c_values=batch
        y_pred=self.forward(x,x_cont)
        loss_y=self.loss(y_pred, y,c_values)
        self.log('train_loss', loss_y, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss_y
    
    def configure_optimizers(self) ->Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    









