import torch
import torch.nn as nn
import pytorch_lightning as pl
from typing import Any

class DeepFM(pl.LightningModule):
    def __init__(self, num_features, emb_num_features,num_factors, args):
        super(DeepFM, self).__init__()
        self.num_features = num_features
        self.num_factors = num_factors
        self.emb_num_features=emb_num_features
        self.weight_decay = args.weight_decay
        self.lr=args.lr
        self.args=args
        
        # embedding part
        self.embedding=nn.Embedding(self.num_features,args.emb_dim)

        self.linear=torch.nn.Embedding(self.num_features,1)
        # FM part
        self.w = nn.Parameter(torch.randn(num_features))
        self.bias=nn.Parameter(torch.randn(1))
        self.v = nn.Parameter(torch.randn(args.emb_dim, num_factors))
        
        # Deep part
        input_size = args.emb_dim*num_features  # Adjust this line to match the shape of your input data
        self.deep_layers = nn.ModuleList()
        for i in range(args.num_deep_layers):
            self.deep_layers.append(nn.Linear(input_size, args.deep_layer_size))
            self.deep_layers.append(nn.ReLU())
            self.deep_layers.append(nn.Dropout(p=0.2))
            input_size = args.deep_layer_size
        self.deep_output_layer = nn.Linear(input_size, 1)
        self.bceloss=nn.BCEWithLogitsLoss() # since bcewith logits is used, we don't need to add sigmoid layer in the end

    def mse(self, y_pred, y_true):
        return self.bceloss(y_pred, y_true.float())


    def deep_part(self, x):
        deep_x = x
        for layer in self.deep_layers:
            deep_x = layer(deep_x)
        deep_out = self.deep_output_layer(deep_x)
        return deep_out
    

    def l2norm(self):
        
        for param in self.model.parameters():
            param.data = param.data / torch.norm(param.data, 2)

        
        

    def loss(self, y_pred, y_true, c_values):
        mse =self.bceloss(y_pred, y_true.float())
        #bce=self.bceloss(y_pred,y_true.float())

        weighted_bce = c_values * mse
        l2_reg = torch.norm(self.w) + torch.norm(self.v) # L2 regularization

        loss_y=torch.mean(weighted_bce) + self.weight_decay * l2_reg
        
        return loss_y
    
    def fm_part(self, x,emb_x):
        #linear_terms = torch.matmul(x, self.w)+self.bias
        linear_terms=torch.sum(self.linear(x),dim=1)+self.bias


        square_of_sum = torch.sum((emb_x), dim=1) ** 2
        sum_of_square = torch.sum((emb_x) ** 2, dim=1)
        ix=square_of_sum-sum_of_square
        interactions = 0.5 * torch.sum(ix, dim=1, keepdim=True)
        return linear_terms.squeeze() + interactions.squeeze()

    def forward(self, x,x_hat):
        # FM part, here, x_hat means another arbritary input of data, for combining the results. 
        emb_x=self.embedding(x)
        
        
        fm_part=self.fm_part(x,emb_x)
        
        deep_part=self.deep_part(emb_x.view(-1, self.args.emb_dim*self.num_features))
        #x=x.float()
        
        # Deep part

        #deep_out=self.sig(deep_out)
        y_pred=fm_part+deep_part.squeeze()
       
        return y_pred

    def training_step(self, batch, batch_idx):
        x,y,c_values,emb_x=batch
        y_pred=self.forward(x,emb_x)
        loss_y=self.loss(y_pred, y,c_values)
        self.log('train_loss', loss_y, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss_y
    
    def configure_optimizers(self) ->Any:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
    
    









