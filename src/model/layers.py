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
        self.w=nn.Parameter(torch.randn(args.cont_dims))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.int64)
        self.args=args
    
    def forward(self, x,x_cont):
        # input x: batch_size * num_features
        x=x+x.new_tensor(self.offsets).unsqueeze(0)
        linear_term=self.linear(x)
        # add continuous features
        cont_linear = torch.matmul(x_cont, self.w).reshape(-1,1)

        x = torch.sum(linear_term,dim=1)+self.bias
        # if self.args.embedding_type=='SVD':
        #     x=x+cont_linear
        # else:
        x=x+cont_linear 

        return x

class FM_Interaction(nn.Module):

    def __init__(self,args,input_size):
        super(FM_Interaction, self).__init__()
        self.args=args
        #self.v=nn.Parameter(torch.randn(args.num_eigenvector*2,16))
        self.v = nn.Parameter(torch.randn(args.cont_dims,args.num_eigenvector*2))
    
    def forward(self, x,x_cont):
        # input x: batch_size * num_features * num_embedding
        #cont_linear=torch.matmul(x_cont,self.v)
        x_comb=x
        x_cont=x_cont.unsqueeze(1)

        if self.args.embedding_type=='SVD':
            linear=torch.sum(x_comb,1)**2
            square_sum=torch.sum(x_comb**2,1)

        else:
            linear=torch.sum(x_comb,1)**2
            square_sum=torch.sum(x_comb**2,1)

        #square_sum_emb=torch.concat((square_sum,x_cont**2),1)


        cont_linear=torch.sum(torch.matmul(x_cont,self.v)**2,dim=1)
        #cont_linear=torch.matmul(x_cont,self.v)**2
        new_linear=torch.cat((linear,cont_linear),1)

        cont_interaction=torch.sum(torch.matmul(x_cont**2,self.v**2),1,keepdim=True)
        new_interaction= torch.cat((square_sum,cont_interaction.squeeze(1)),1)


        interaction=0.5*torch.sum(new_linear-new_interaction,1,keepdim=True)
        
        
        #interaction=interaction+0.5*torch.sum(new_linear,1,keepdim=True)
        # interaction = 0.5 * torch.sum(

        # cont_interactions = 0.5 * torch.sum(
        #     torch.matmul(x_cont, self.v) ** 2 - torch.matmul(x_cont ** 2, self.v ** 2),
        #     dim=1,
        #     keepdim=True
        # )
        # interaction = interaction + cont_interactions
    

        return interaction




