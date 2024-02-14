from typing import Any
import torch
import torch.nn as nn
from src.model.SVD_emb.layers import MLP,FeatureEmbedding,FM_Linear,FM_Interaction
#from src.util.scaler import StandardScaler

#lightning
import pytorch_lightning as pl

class FactorizationMachineSVD(pl.LightningModule):
    def __init__(self, args, field_dims):
        super(FactorizationMachineSVD, self).__init__()

        if args.model_type=='fm':
            self.embedding=FeatureEmbedding(args,field_dims)
        self.linear=FM_Linear(args,field_dims)
        self.interaction=FM_Interaction(args,field_dims)
        self.bceloss=nn.BCEWithLogitsLoss() # since bcewith logits is used, we don't need to add sigmoid layer in the end
        self.lr=args.lr
        self.args=args
        self.sig=nn.Sigmoid()
        self.last_linear=nn.Linear(2,1)


    def l2norm(self):
        reg=0
        for param in self.linear.parameters():
            reg+=torch.norm(param)**2
        for param in self.embedding.parameters():
            reg+=torch.norm(param)**2
        for param in self.interaction.parameters():
            reg+=torch.norm(param)**2
        return reg*self.args.weight_decay

    def loss(self, y_pred, y_true,c_values):
        # calculate weighted mse with l2 regularization
        #mse = (y_pred - y_true.float()) ** 2
        bce = self.bceloss(y_pred, y_true.float())
        weighted_bce = c_values * bce
        #l2_reg = torch.norm(self.w)**2 + torch.norm(self.v)**2
        loss_y = weighted_bce.mean() +self.l2norm()#+ self.args.weight_decay * l2_reg

        return loss_y 
    
    def forward(self, x,emb_x,svd_emb,x_cont):
        # FM part loss with interaction terms
        # x: batch_size * num_features
        lin_term = self.linear(x,svd_emb,x_cont)
        
        #embedding=self.embedding(x)

        inter_term,cont_emb = self.interaction(emb_x,svd_emb,x_cont)




        #x= lin_term*0.0001+ inter_term*0.0001


        # want to normalize lin_term and inter_term to be in the same scale
        # so that the weights can be comparable
        lin_term_sig=self.sig(lin_term)
        inter_term_sig=self.sig(inter_term)
        outs=torch.cat((lin_term_sig,inter_term_sig),1)
        x=self.last_linear(outs)
        
        x=x.squeeze(1)

        return x, cont_emb, lin_term, inter_term

    
    def training_step(self, batch, batch_idx):
        x,svd_emb,ui,x_cont,y,c_values=batch


        embed_x=self.embedding(x)
        y_pred,_,_,_=self.forward(x,embed_x,svd_emb,x_cont)
        
        loss_y=self.loss(y_pred, y,c_values)
        self.log('train_loss', loss_y, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss_y
    
    def configure_optimizers(self) ->Any:

        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    

    





    
