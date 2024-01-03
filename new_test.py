import torch
from torch.utils.data import Dataset
from src.util.negativesampler import NegativeSampler
import argparse
from src.data.customdataloader import CustomDataLoader
from src.data.SVDdataloader import SVDDataloader
from torch.utils.data import DataLoader
from src.data.datawrapper import DataWrapper
from src.model.original.fm import FactorizationMachine
from src.model.SVD_emb.svdfm import FactorizationMachineSVD
from src.model.SVD_emb.svddeepfm import DeepFMSVD   
from src.customtest import Emb_Test
from sklearn.preprocessing import LabelEncoder
from src.model.original.deepfm import DeepFM
from src.model.SVD import SVD
from src.data.custompreprocess import CustomOneHot
import time
import numpy as np
#copy
import json
from copy import deepcopy
from src.util.preprocessor import Preprocessor
parser = argparse.ArgumentParser()

parser.add_argument('--train_ratio', type=float, default=0.7, help='training ratio for any dataset')


#parser.add_argument('--num_factors', type=int, default=15, help='Number of factors for FM')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for fm training')
parser.add_argument('--weight_decay', type=float, default=0.00001, help='Weight decay(for both FM and autoencoder)')
#parser.add_argument('--num_epochs_ae', type=int, default=300,    help='Number of epochs')
parser.add_argument('--num_epochs_training', type=int, default=1,    help='Number of epochs')

parser.add_argument('--batch_size', type=int, default=4096, help='Batch size')
#parser.add_argument('--ae_batch_size', type=int, default=256, help='Batch size for autoencoder')

parser.add_argument('--num_workers', type=int, default=10, help='Number of workers for dataloader')
parser.add_argument('--num_deep_layers', type=int, default=2, help='Number of deep layers')
parser.add_argument('--deep_layer_size', type=int, default=128, help='Size of deep layers')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--save_model', type=bool, default=False)


parser.add_argument('--emb_dim', type=int, default=16, help='embedding dimension for DeepFM')
#parser.add_argument('--num_embedding', type=int, default=200, help='Number of embedding for autoencoder') 
parser.add_argument('--embedding_type', type=str, default='original', help='AE or SVD or original')
parser.add_argument('--model_type', type=str, default='deepfm', help='fm or deepfm')
parser.add_argument('--topk', type=int, default=5, help='top k items to recommend')
parser.add_argument('--fold', type=int, default=1, help='fold number for folded dataset')
parser.add_argument('--isuniform', type=bool, default=False, help='true if uniform false if not')
parser.add_argument('--ratio_negative', type=int, default=0.2, help='negative sampling ratio rate for each user')
#parser.add_argument('--auto_lr', type=float, default=0.01, help='autoencoder learning rate')
#parser.add_argument('--k', type=int, default=10, help='autoencoder k')
parser.add_argument('--num_eigenvector', type=int, default=16,help='Number of eigenvectors for SVD ,note that this must be same as emb_dim')
parser.add_argument('--datatype', type=str, default="goodbook",help='ml100k or ml1m or shopping or goodbook or frappe')
parser.add_argument('--c_zeros', type=int, default=5,help='c_zero for negative sampling')
parser.add_argument('--cont_dims', type=int, default=0,help='continuous dimension(that changes for each dataset))')
parser.add_argument('--shopping_file_num', type=int, default=147,help='name of shopping file choose from 147 or  148 or 149')


args = parser.parse_args("")




def getdata(args):
    
    # get any dataset
    dataset=DataWrapper(args)

    train_df, test, item_info, user_info, ui_matrix =dataset.get_data()
    cat_columns,cont_cols=dataset.get_col_type()
    #those are basic dataframes that we can get from various datasets
    preprocessor=Preprocessor(args,train_df,test,user_info,item_info,ui_matrix,cat_columns,cont_cols)
    #preprocessor is a class that preprocesses dataframes and returns train_df, test_df, item_info, user_info, useritem_matrix, cat_columns, cont_columns, label_encoders, user_embedding, item_embedding
    return preprocessor


def trainer(args,data:Preprocessor):
    data.label_encode()
    items,cons=data.get_catcont_train()
    target,c=data.get_target_c()
    field_dims=data.get_field_dims()
    uidf=data.uidf.values


    if args.model_type=='fm' and args.embedding_type=='original':
        model=FactorizationMachine(args,field_dims)
        Dataset=CustomDataLoader(items,cons,target,c)
    elif args.model_type=='deepfm' and args.embedding_type=='original':
        model=DeepFM(args,field_dims)
        Dataset=CustomDataLoader(items,cons,target,c)
    elif args.model_type=='fm' and args.embedding_type=='SVD':
        model=FactorizationMachineSVD(args,field_dims)
        embs=cons[:,-32:]
        cons=cons[:,:-32]
        Dataset=SVDDataloader(items,embs,uidf,cons,target,c)
    elif args.model_type=='deepfm' and args.embedding_type=='SVD':
        model=DeepFMSVD(args,field_dims)
        embs=cons[:,-32:]
        cons=cons[:,:-32]
        Dataset=SVDDataloader(items,embs,uidf,cons,target,c)
    else:
        raise NotImplementedError
    
    
    #dataloaders

    dataloader=DataLoader(Dataset,batch_size=args.batch_size,shuffle=True,num_workers=20)

    import pytorch_lightning as pl

    #fm=DeepFM(args,field_dims)
    import time
    start=time.time()
    trainer=pl.Trainer(max_epochs=args.num_epochs_training)
    trainer.fit(model,dataloader)
    end=time.time()
    return model, end-start

if __name__=='__main__':
    args = parser.parse_args("")
    svdresults=[]
    originalresults=[]
    results={}

    #data_types=['goodbook']
    embedding_type=['SVD','original']
    model_type=['fm','deepfm']
    #shopping_file_num=[147,148,149]
    folds=[1,2,3,4,5]
    isuniform=[True]
    times=[]
    for uni in isuniform:
        args.isuniform=uni
        data_info=getdata(args)
        for md in model_type:
            args.model_type=md
            for embedding in embedding_type:
                args.embedding_type=embedding
            
                print('model type is',md)
                print('embedding type is',embedding)
                model,timeee=trainer(args,data_info)
                test_time=time.time()
                tester=Emb_Test(args,model,data_info)
                if args.embedding_type=='SVD':
                    result=tester.svdtest()
                else:
                    result=tester.test()
                end_test_time=time.time()
                results[md+embedding]=result
                    #results[md+embedding]=result
                times.append(end_test_time-test_time)
            

        dataset_name=args.datatype
        num_eigenvector=args.num_eigenvector
        print('time is',times)
        # json_name=dataset_name+'_'+'eigen_'+str(num_eigenvector)+'_'+'uniform'+str(args.isuniform)+'ver345.json'
        # # want to save in results folder
        # #folder
        # foldername='n_results/'+dataset_name+'/'
        # if dataset_name=='shopping':
        #     json_name=dataset_name+'_'+str(args.shopping_file_num)+'_'+'eigen_'+str(num_eigenvector)+'_'+'uniform'+str(args.isuniform)+'.json'
        # if dataset_name=='ml100k':
        #     json_name=dataset_name+'_folds'+str(args.fold)+'eigen_'+str(num_eigenvector)+'_'+'uniform'+str(args.isuniform)+'.json'
        # with open(foldername+json_name, 'w') as fp:
        #     json.dump(results, fp)

