import torch
from torch.utils.data import Dataset
from src.util.negativesampler import NegativeSampler
import argparse
from src.data.customdataloader import CustomDataLoader
from torch.utils.data import DataLoader
from src.data.datawrapper import DataWrapper
from src.model.fm import FactorizationMachine
from src.customtest import Emb_Test
from sklearn.preprocessing import LabelEncoder
from src.model.deepfm import DeepFM
from src.model.SVD import SVD
from src.data.custompreprocess import CustomOneHot
import time
import numpy as np
#copy
from copy import deepcopy
parser = argparse.ArgumentParser()

parser.add_argument('--train_ratio', type=float, default=0.7, help='training ratio for movielens1m')


parser.add_argument('--num_factors', type=int, default=15, help='Number of factors for FM')
parser.add_argument('--lr', type=float, default=0.005, help='Learning rate for fm training')
parser.add_argument('--weight_decay', type=float, default=0.001, help='Weight decay(for both FM and autoencoder)')
parser.add_argument('--num_epochs_ae', type=int, default=300,    help='Number of epochs')
parser.add_argument('--num_epochs_training', type=int, default=40,    help='Number of epochs')

parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
parser.add_argument('--ae_batch_size', type=int, default=256, help='Batch size for autoencoder')

parser.add_argument('--num_workers', type=int, default=10, help='Number of workers for dataloader')
parser.add_argument('--num_deep_layers', type=int, default=2, help='Number of deep layers')
parser.add_argument('--deep_layer_size', type=int, default=128, help='Size of deep layers')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--save_model', type=bool, default=False)


parser.add_argument('--emb_dim', type=int, default=128, help='embedding dimension for DeepFM')
parser.add_argument('--num_embedding', type=int, default=200, help='Number of embedding for autoencoder') 
parser.add_argument('--embedding_type', type=str, default='original', help='AE or SVD or original')
parser.add_argument('--model_type', type=str, default='deepfm', help='fm or deepfm')
parser.add_argument('--topk', type=int, default=5, help='top k items to recommend')
parser.add_argument('--fold', type=int, default=1, help='fold number')
parser.add_argument('--isuniform', type=bool, default=False, help='isuniform')
parser.add_argument('--ratio_negative', type=int, default=0.2, help='ratio_negative')
parser.add_argument('--auto_lr', type=float, default=0.01, help='autoencoder learning rate')
parser.add_argument('--k', type=int, default=10, help='autoencoder k')
parser.add_argument('--num_eigenvector', type=int, default=64,help='Number of eigenvectors for SVD')
parser.add_argument('--datatype', type=str, default="goodbook",help='ml100k or ml1m or shopping or goodbook or frappe')
parser.add_argument('--c_zeros', type=int, default=5,help='c_zero for negative sampling')
parser.add_argument('--cont_dims', type=int, default=0,help='continuous dimension(that changes for each dataset))')


args = parser.parse_args("")




def getdata(args):
    
    # get any dataset
    dataset=DataWrapper(args)
    train_df, test, item_info, user_info, ui_matrix =dataset.get_data()
    train=train_df.copy(deep=True)
    #cat_columns,cont_columns=dataset.get_columns()

    # do negative sampling and merge with item_info and user_info, negative sampling based on c
    ns=NegativeSampler(args,train,item_info,user_info)
    nssampled=ns.negativesample(args.isuniform)


    target=nssampled['target'].to_numpy()
    c=nssampled['c'].to_numpy()
    nssampled.drop(['target','c'],axis=1,inplace= True)

    nssampled=nssampled.merge(item_info,on='item_id',how='left')
    nssampled=nssampled.merge(user_info,on='user_id',how='left')

    user_embedding,item_embedding= SVD(args).get_embedding(ui_matrix)

    # if dataset is movielens or frappe  there is no continuous column
    cat_columns=nssampled.columns

    #if args.datatype=='goodbook' :
    cat_columns,cont_cols=dataset.get_col_type()
    
    
    merger=CustomOneHot(args,nssampled,item_info,user_info)
    new_train_df,user_embedding_df,item_embedding_df=merger.embedding_merge(user_embedding=user_embedding,item_embedding=item_embedding)

    cont_train_df=new_train_df.drop(cat_columns,axis=1)
    #labelencoder

    les={}

    total_columns=new_train_df.columns
    #select categorical columns without hurting the order of original
    cat_columns=[col for col in total_columns if col  in cat_columns]

    if args.embedding_type=='SVD':
        for col in cat_columns:
            le=LabelEncoder()
            
            if col=='user_id' or col=='item_id':
                le.fit(new_train_df[col])
            else:
                new_train_df[col]=le.fit_transform(new_train_df[col])
            les[col]=le
    else:
        for col in cat_columns:
            le=LabelEncoder()
            new_train_df[col]=le.fit_transform(new_train_df[col])
            les[col]=le

    if args.embedding_type=='SVD':
        items=new_train_df[cat_columns].drop(['user_id','item_id'],axis=1).to_numpy()[:].astype('int')
    else: 
        items=new_train_df[cat_columns].to_numpy()[:].astype('int')
    


    if args.embedding_type=='original':
        cont_train_df=cont_train_df[cont_cols]
        args.cont_dims=len(cont_cols)

    else:
        cont_cols=cont_cols+user_embedding_df.columns.tolist()+item_embedding_df.columns.tolist()
        #delete user_id, item_id from cont_cols
        cont_cols.remove('user_id')
        cont_cols.remove('item_id')

        cont_train_df=cont_train_df[cont_cols]    
        args.cont_dims=len(cont_cols)
        cat_columns.remove('user_id')
        cat_columns.remove('item_id')
    
    
    cons=cont_train_df.to_numpy()[:].astype('float32')
    field_dims=np.max(items,axis=0)+1



    return items,cons,target,c,field_dims,les,item_info,user_info,new_train_df,test,user_embedding,item_embedding, cat_columns,cont_cols,train_df


def trainer(args,items,cons,target,c,field_dims):
    if args.model_type=='fm':
        fm=FactorizationMachine(args,field_dims)
    else:

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
    model_type=['deepfm','fm']
    svd_test_time=[]
    original_test_time=[]
    svd_train_time=[]
    original_train_time=[]
    results={}
    
    for md in model_type:
        args.model_type=md
        for embedding in embedding_type:
            args.embedding_type=embedding
           
            items,cons,target,c,field_dims,le,item_info,user_info,train_df,test_df,user_embedding,item_embedding,cat_cols,cont_cols,train_org=getdata(args)
            start_training_time=time.time()
            model=trainer(args,items,cons,target,c,field_dims)
            end_training_time=time.time()
            
            start_test_time=time.time()
            tester=Emb_Test(args,model,train_df,test_df,le,item_info,user_info,user_embedding,item_embedding,cat_cols,cont_cols,train_org)


            result=tester.test()
            end_test_time=time.time()
            results[md+embedding]=result


            
    
    print(args.isuniform)
    print(results)
