import torch
import torch.nn as nn

class MLP(nn.Module):

    def __init__(self, args,input_size):
        super(MLP, self).__init__()
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
        self.embedding=nn.Embedding(sum(field_dims),args.emb_dim)

    def forward(self, x):
        # input x: batch_size * num_features
        x = self.embedding(x)
        return x


class FM_Linear(nn.Module):

    def __init__(self,args,field_dims):
        super(FM_Linear, self).__init__()
        self.linear=torch.nn.Embedding(sum(field_dims),1)
        self.bias=nn.Parameter(torch.randn(1))
        self.w=nn.Parameter(torch.randn(args.num_eigenvector*2))
        self.args=args
    
    def forward(self, x,x_cont):
        # input x: batch_size * num_features
        linear_term=self.linear(x)
        # add continuous features
        cont_linear = torch.matmul(x_cont, self.w).reshape(-1,1)

        x = torch.sum(linear_term,dim=1)+self.bias
        if self.args.embedding_type=='SVD':
            x=x+cont_linear
        else:
            x=x 

        return x

class FM_Interaction(nn.Module):

    def __init__(self,args,input_size):
        super(FM_Interaction, self).__init__()
        self.args=args
        #self.v=nn.Parameter(torch.randn(args.num_eigenvector*2,16))
    
    def forward(self, x,x_cont):
        # input x: batch_size * num_features * num_embedding
        #cont_linear=torch.matmul(x_cont,self.v)
        x_cont=x_cont.unsqueeze(1)
        # must be sure that embedding layer should be same. 
        x_comb=torch.cat((x,x_cont),1)
        # x_comb: batch_size * (num_features+1) * num_embedding


        if self.args.embedding_type=='SVD':
            linear=torch.sum(x_comb,1)
            square_sum=torch.sum(x_comb**2,1)
            
        else:
            linear=torch.sum(x,1)
            square_sum=torch.sum(x**2,1)

        #square_sum_emb=torch.concat((square_sum,x_cont**2),1)

        interaction=0.5*(linear**2-square_sum)
        interaction=torch.sum(interaction,1,keepdim=True)

    

        return interaction




