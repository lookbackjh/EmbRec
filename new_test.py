import torch
from torch.utils.data import Dataset
from src.util.negativesampler import NegativeSampler
from src.data.movielens100k import Movielens100k
import argparse
from src.data.customdataloader import CustomDataLoader
from torch.utils.data import DataLoader
from src.data.realdata import RealData
from src.model.fm import FactorizationMachine
from src.customtest import Emb_Test
from sklearn.preprocessing import LabelEncoder
from src.model.deepfm import DeepFM
from src.model.SVD import SVD
from src.data.custompreprocess import CustomOneHot
from src.model.layers import MLP
import time
import numpy as np
#copy
from copy import deepcopy
parser = argparse.ArgumentParser()
parser.add_argument('--num_factors', type=int, default=15, help='Number of factors for FM')
parser.add_argument('--lr', type=float, default=0.005, help='Learning rate for fm training')
parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay(for both FM and autoencoder)')
parser.add_argument('--num_epochs_ae', type=int, default=100,    help='Number of epochs')
parser.add_argument('--num_epochs_training', type=int, default=50,    help='Number of epochs')

parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
parser.add_argument('--ae_batch_size', type=int, default=256, help='Batch size for autoencoder')

parser.add_argument('--num_workers', type=int, default=20, help='Number of workers for dataloader')
parser.add_argument('--num_deep_layers', type=int, default=2, help='Number of deep layers')
parser.add_argument('--deep_layer_size', type=int, default=128, help='Size of deep layers')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--save_model', type=bool, default=False)


parser.add_argument('--emb_dim', type=int, default=16, help='embedding dimension for DeepFM')
parser.add_argument('--num_embedding', type=int, default=200, help='Number of embedding for autoencoder') 
parser.add_argument('--embedding_type', type=str, default='SVD', help='AE or SVD or original')
parser.add_argument('--model_type', type=str, default='deepfm', help='fm or deepfm')
parser.add_argument('--topk', type=int, default=5, help='top k items to recommend')
parser.add_argument('--fold', type=int, default=1, help='fold number')
parser.add_argument('--isuniform', type=bool, default=True, help='isuniform')
parser.add_argument('--ratio_negative', type=int, default=0.2, help='ratio_negative')
parser.add_argument('--auto_lr', type=float, default=0.01, help='autoencoder learning rate')
parser.add_argument('--k', type=int, default=10, help='autoencoder k')
parser.add_argument('--num_eigenvector', type=int, default=8,help='Number of eigenvectors for SVD')
parser.add_argument('--datatype', type=str, default="ml100k",help='ml100k or ml1m or goodbook or googlebook')
args = parser.parse_args("")




def getdata(args):
    
    # get any dataset
    dataset=RealData(args)
    train_df, test, item_info, user_info, ui_matrix =dataset.get_data()
    train=train_df.copy(deep=True)

    # do negative sampling and merge with item_info and user_info, negative sampling based on c
    ns=NegativeSampler(args,train,item_info,user_info)
    nssampled=ns.negativesample(args.isuniform)
    target=nssampled['target'].to_numpy()
    c=nssampled['c'].to_numpy()
    nssampled.drop(['target','c'],axis=1,inplace= True)
    nssampled=nssampled.merge(item_info,on='item_id',how='left')
    nssampled=nssampled.merge(user_info,on='user_id',how='left')

    user_embedding,item_embedding= SVD(args).get_embedding(ui_matrix)

    cat_columns=nssampled.columns


    merger=CustomOneHot(args,nssampled,item_info,user_info)
    new_train_df=merger.embedding_merge(user_embedding=user_embedding,item_embedding=item_embedding)

    cont_train_df=new_train_df.drop(cat_columns,axis=1)
    #labelencoder

    les={}

    
    for col in cat_columns:
        le=LabelEncoder()
        new_train_df[col]=le.fit_transform(new_train_df[col])
        les[col]=le


    if args.embedding_type=='SVD':
        items=new_train_df[cat_columns].drop(['user_id','item_id'],axis=1).to_numpy()[:].astype('int')
    else: 
        items=new_train_df[cat_columns].to_numpy()[:].astype('int')
    
    cons=cont_train_df.to_numpy()[:].astype('float32')
    
    field_dims=np.max(items,axis=0)+1



    return items,cons,target,c,field_dims,les,item_info,user_info,train_df,test,user_embedding,item_embedding


def trainer(args,items,cons,target,c,field_dims):
    fm=DeepFM(args,field_dims)
    Dataset=CustomDataLoader(items,cons,target,c)
    #dataloaders

    dataloader=DataLoader(Dataset,batch_size=1024,shuffle=True,num_workers=20)

    import pytorch_lightning as pl

    #fm=DeepFM(args,field_dims)
    trainer=pl.Trainer(max_epochs=args.num_epochs_training)
    trainer.fit(fm,dataloader)
    return fm

if __name__=='__main__':
    args = parser.parse_args("")
    svdresults=[]
    originalresults=[]
    embedding_type=['SVD','original']

    for embedding in embedding_type:
        args.embedding_type=embedding
        for i in range(1,6):
            args.fold=i
            items,cons,target,c,field_dims,le,item_info,user_info,train_df,test_df,user_embedding,item_embedding=getdata(args)
            model=trainer(args,items,cons,target,c,field_dims)
            tester=Emb_Test(args,model,train_df,test_df,le,item_info,user_info,user_embedding,item_embedding)
            result=tester.test()
            if embedding=='SVD':
                svdresults.append(result)
            else:
                originalresults.append(result)
    

    

    for i in range(5):
        print("fold ",i+1," SVD result: ",svdresults[i])
        print("fold ",i+1," original result: ",originalresults[i])