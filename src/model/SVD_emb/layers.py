import torch
import torch.nn as nn
import numpy as np

class MLP(nn.Module):

    def __init__(self, args,input_size):
        super(MLP, self).__init__()
        self.args=args
        self.deep_layers = nn.ModuleList()
        for i in range(args.num_deep_layers):
            self.deep_layers.append(nn.Linear(input_size, args.deep_layer_size))
            self.deep_layers.append(nn.ReLU())
            self.deep_layers.append(nn.Dropout(p=0.2))
            input_size = args.deep_layer_size
        self.deep_output_layer = nn.Linear(input_size, 1)

    def forward(self, x):
        # input x : batch_size * (num_features* num_embedding)
        deep_x = x
        for layer in self.deep_layers:
            deep_x = layer(deep_x)
        x = self.deep_output_layer(deep_x)
        return x

class FeatureEmbedding(nn.Module):

    def __init__(self,args,field_dims):
        super(FeatureEmbedding, self).__init__()
        self.embedding=nn.Embedding(sum(field_dims+1),args.emb_dim)
        self.field_dims=field_dims
        # for adding offset for each feature for example, movie id starts from 0, user id starts from 1000
        # as the features should be embedded column-wise this operatation easily makes it possible
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)

    def forward(self, x):
        # input x: batch_size * num_features
        x = x + x.new_tensor(self.offsets).unsqueeze(0)  # this is for adding offset for each feature for example, movie id starts from 0, user id starts from 1000

        x=self.embedding(x)


        return x


class FM_Linear(nn.Module):

    def __init__(self,args,field_dims):
        super(FM_Linear, self).__init__()
        self.linear=torch.nn.Embedding(sum(field_dims)+1,1)
        self.bias=nn.Parameter(torch.randn(1))
        self.w=nn.Parameter(torch.randn(args.cont_dims-args.num_eigenvector*2))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        self.args=args
    
    def forward(self,x, emb_x,x_cont):
        # input x: batch_size * num_features
        x=x+x.new_tensor(self.offsets).unsqueeze(0)
        
        linear_term=self.linear(x)
        # linear_term: batch_size * num_features * 1


        # add continuous features
        cont_linear = torch.matmul(x_cont, self.w).reshape(-1,1)

        x = torch.sum(linear_term,dim=1)+self.bias

        emb_x=emb_x[:,0].reshape(-1,1)
        # if self.args.embedding_type=='SVD':
        #     x=x+cont_linear
        # else:
        x=x+cont_linear+emb_x 

        return x

class FM_Interaction(nn.Module):

    def __init__(self,args,input_size):
        super(FM_Interaction, self).__init__()
        self.args=args
        #self.v=nn.Parameter(torch.randn(args.num_eigenvector*2,16))
        self.v = nn.Parameter(torch.randn(args.cont_dims-args.num_eigenvector*2, args.emb_dim))
    
    def forward(self, emb_x,svd_emb,x_cont):
        # input x: batch_size * num_features * num_embedding
        #cont_linear=torch.matmul(x_cont,self.v)
        x_comb=emb_x
        x_cont=x_cont.unsqueeze(1)
        user_emb=svd_emb[:,:self.args.num_eigenvector].unsqueeze(1)
        item_emb=svd_emb[:,self.args.num_eigenvector:].unsqueeze(1)
        x_comb=torch.cat((x_comb,user_emb),1)
        x_comb=torch.cat((x_comb,item_emb),1)

        sum_square=torch.sum(x_comb,1)**2
        square_sum=torch.sum(x_comb**2,1)
        interaction=0.5*torch.sum(sum_square-square_sum,1,keepdim=True)

        cont_interactions = 0.5 * torch.sum(
            torch.matmul(x_cont, self.v) ** 2 - torch.matmul(x_cont ** 2, self.v ** 2),
            dim=1,
            keepdim=True
        )

        #square_sum_emb=torch.concat((square_sum,x_cont**2),1)

        new_interaction=interaction+cont_interactions
        
        cont_emb=torch.matmul(x_cont,self.v)

        return interaction, cont_emb




