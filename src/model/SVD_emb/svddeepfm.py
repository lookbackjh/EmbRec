import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any

import torch
from src.model.SVD_emb.svdfm import FactorizationMachineSVD
from src.model.SVD_emb.layers import FeatureEmbedding, FeatureEmbedding, FM_Linear, MLP
#from src.util.scaler import StandardScaler



class DeepFMSVD(pl.LightningModule):
    def __init__(self, args,field_dims):
        super(DeepFMSVD, self).__init__()
        self.linear = FM_Linear(args,field_dims)
        self.fm = FactorizationMachineSVD(args, field_dims)
        self.embedding = FeatureEmbedding(args, field_dims)

        self.embed_output_dim = (len(field_dims))* args.emb_dim + 2*args.emb_dim+(args.cont_dims-2*args.num_eigenvector)*args.emb_dim

        #self.embed_output_dim = (len(field_dims) +1)* args.emb_dim
        self.mlp = MLP(args, self.embed_output_dim)
        self.bceloss=nn.BCEWithLogitsLoss() # since bcewith logits is used, we don't need to add sigmoid layer in the end
        self.lr=args.lr
        self.args=args
        # self.simplemlp=nn.Sequential(
        #         nn.Linear(args.cont_dims,args.num_eigenvector*2),
        #         nn.ReLU(),
        #         )
        self.field_dims=field_dims
        self.sig=nn.Sigmoid()
        self.lastlinear=nn.Linear(3,1)

    def l2norm(self):

        reg=0
        for param in self.linear.parameters():
            reg+=torch.norm(param)**2
        for param in self.embedding.parameters():
            reg+=torch.norm(param)**2
        for param in self.mlp.parameters():
            reg+=torch.norm(param)**2
        return reg*self.args.weight_decay


    def mse(self, y_pred, y_true):
        return self.bceloss(y_pred, y_true.float())


    def deep_part(self, x):
        return self.mlp(x)
    

        

    def loss(self, y_pred, y_true, c_values):
        mse =self.bceloss(y_pred, y_true.float())
        #bce=self.bceloss(y_pred,y_true.float())

        weighted_bce = c_values * mse
        #l2_reg = torch.norm(self.w) + torch.norm(self.v) # L2 regularization

        loss_y=weighted_bce.mean() #+ self.args.weight_decay * l2_reg
        
        loss_y+=self.l2norm()

        return loss_y
    

    def forward(self, x,embed_x,svd_emb,x_cont):
        # FM part, here, x_hat means another arbritary input of data, for combining the results. 
        
        #embed_x=self.embedding(x)
        fm_part,cont_emb,lin_term,inter_term=self.fm(x, embed_x,svd_emb,x_cont)
        user_emb=svd_emb[:,:self.args.num_eigenvector]
        item_emb=svd_emb[:,self.args.num_eigenvector:]


        #embed_x.shape: batch_size * num_features * embedding_dim   
        


        embed_x=torch.cat((embed_x,cont_emb),1)
        feature_number=embed_x.shape[1]
        
        # want embed_x to be batch_size * (num_features*embedding_dim)
        embed_x=embed_x.reshape(-1,feature_number*self.args.emb_dim)
        
        #embed_x=embed_x.view(-1,feature_number*self.args.emb_dim)
        new_x=torch.cat((embed_x,user_emb),1)
        bnew_x=torch.cat((new_x,item_emb),1)
        deep_part=self.mlp(bnew_x)
        #x=x.float()
        
        # Deep part

        #deep_out=self.sig(deep_out)
        #std3=StandardScaler()
        lin_term=self.sig(lin_term)
        inter_term=self.sig(inter_term)
        deep_part=self.sig(deep_part)

        outs=torch.cat((lin_term,inter_term ),1)
        outs=torch.cat((outs,deep_part),1)
        y_pred=self.lastlinear(outs).squeeze(1)

        #y_pred=fm_part+deep_part.squeeze()
       
        #sig_y_pred=self.sig(y_pred)

        return y_pred

    def training_step(self, batch, batch_idx):
        x,svd_emb,ui,x_cont,y,c_values=batch


        embed_x=self.embedding(x)
        y_pred=self.forward(x,embed_x,svd_emb,x_cont)
        loss_y=self.loss(y_pred, y,c_values)
        self.log('train_loss', loss_y, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss_y
    
    def configure_optimizers(self) ->Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    









