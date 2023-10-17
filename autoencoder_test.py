import torch
import pytorch_lightning as pl
from src.model.deepfm import DeepFM
from src.data.customdataloader import CustomDataLoader
import pandas as pd
from src.data.fm_preprocess import FM_Preprocessing
from src.model.deepfm import DeepFM
from src.model.fm import FactorizationMachine
import argparse
from src.data.negativesampler import NegativeSampler
from src.data.custompreprocess import CustomOneHot
from src.data.movielensdata import MovielensData
from src.tester import Tester
from src.model.autoencoder import AutoEncoder
from src.data.autoencoderdataloader import AutoEncoderDataLoader
from src.model.SVD import SVD

def parser():

    parser = argparse.ArgumentParser()
    parser.add_argument('--num_factors', type=int, default=15, help='Number of factors for FM')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate for fm training')
    parser.add_argument('--weight_decay', type=float, default=0.1, help='Weight decay(for both FM and autoencoder)')
    parser.add_argument('--num_epochs_ae', type=int, default=100,    help='Number of epochs')
    parser.add_argument('--num_epochs_training', type=int, default=200,    help='Number of epochs')

    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size')
    parser.add_argument('--ae_batch_size', type=int, default=256, help='Batch size for autoencoder')
    
    parser.add_argument('--num_workers', type=int, default=20, help='Number of workers for dataloader')
    parser.add_argument('--num_deep_layers', type=int, default=2, help='Number of deep layers')
    parser.add_argument('--deep_layer_size', type=int, default=128, help='Size of deep layers')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_model', type=bool, default=False)
   

    parser.add_argument('--emb_dim', type=int, default=16, help='embedding dimension for DeepFM')
    parser.add_argument('--num_embedding', type=int, default=200, help='Number of embedding for autoencoder') 
    parser.add_argument('--embedding_type', type=str, default='original', help='AE or SVD or original')
    parser.add_argument('--model_type', type=str, default='deepfm', help='fm or deepfm')
    parser.add_argument('--topk', type=int, default=5, help='top k items to recommend')
    parser.add_argument('--fold', type=int, default=1, help='fold number')
    parser.add_argument('--isuniform', type=bool, default=True, help='isuniform')
    parser.add_argument('--ratio_negative', type=int, default=0.2, help='ratio_negative')
    parser.add_argument('--auto_lr', type=float, default=0.01, help='autoencoder learning rate')
    parser.add_argument('--k', type=int, default=10, help='autoencoder k')
    parser.add_argument('--num_eigenvector', type=int, default=10,help='Number of eigenvectors for SVD')
    args = parser.parse_args("")
    return args



def getdata(args):

    movielens=MovielensData('dataset/ml-100k','u.data',fold=args.fold)
    train,test=movielens.data_getter()
    movie_info=movielens.movie_getter()
    user_info=movielens.user_getter()

    matrix=train.copy()

    ## 2. Negative Sampling
    ns=NegativeSampler(args,train)
    nssampled=ns.negativesample(args.isuniform)
    #print(nssampled)
    
    ## 3. learn user and movie embedding


    ## 4. one hot encoding
    custom_object=CustomOneHot(args,nssampled,movie_info,user_info)
    #train=onehot.infomerge()
    #train=onehot.embedding_merge(user_embedding=user_embedding,movie_embedding=movie_embedding)
    #print(train)
    return train, test,movie_info,user_info,matrix,custom_object

def learn_encoder(args,matrix):
    # matrix is original data
    # from training data, 
    user_item_matrix = matrix.pivot_table(index='user_id', columns='movie_id', values='rating')
    user_item_matrix.fillna(0, inplace=True)
    user_item_matrix[user_item_matrix>1]=1
    user_item_matrix=user_item_matrix.astype(float)

    user_num=user_item_matrix.shape[0]
    movie_num=user_item_matrix.shape[1]
    # diagonal matrix for user

    user_x=torch.eye(user_num)
    user_y=torch.from_numpy(user_item_matrix.values)
    # y to float32
    user_y=user_y.type(torch.FloatTensor)


    # training for user autoencoder
    user_aedataloader=AutoEncoderDataLoader(user_x,user_y)
    user_aedataset=torch.utils.data.DataLoader(user_aedataloader,batch_size=args.ae_batch_size,shuffle=True,num_workers=args.num_workers)

    user_autoencoder=AutoEncoder(user_num,movie_num,args)

    user_autoencoder_trainer=pl.Trainer(max_epochs=args.num_epochs_ae)
    user_autoencoder_trainer.fit(user_autoencoder,user_aedataset)

    # training for movie autoencoder
    movie_x=torch.eye(movie_num)
    movie_y=torch.from_numpy(user_item_matrix.values.T)
    # y to float32
    movie_y=movie_y.type(torch.FloatTensor)

    movie_aedataloader=AutoEncoderDataLoader(movie_x,movie_y)
    movie_aedataset=torch.utils.data.DataLoader(movie_aedataloader,batch_size=args.ae_batch_size,shuffle=True,num_workers=args.num_workers)

    movie_autoencoder=AutoEncoder(movie_num,user_num,args)

    movie_autoencoder_trainer=pl.Trainer(max_epochs=args.num_epochs_ae)
    movie_autoencoder_trainer.fit(movie_autoencoder,movie_aedataset)


    return user_autoencoder,movie_autoencoder, user_x,movie_x


def trainer(args,train_df,emb_train_df):
    # trainer for each fold
    
    #print(train_df)
    #print(train_df)
    train_preprocess = FM_Preprocessing(args,train_df)

    emb_preprocess=FM_Preprocessing(args,emb_train_df)

    
    train_X_tensor=train_preprocess.X_tensor
    train_y_tensor=train_preprocess.y_tensor

    train_c_values_tensor=train_preprocess.c_values_tensor
    
    emb_X_tensor=emb_preprocess.X_tensor


    train_dataset=CustomDataLoader(train_X_tensor,train_y_tensor,train_c_values_tensor,emb_X_tensor)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    
    naive_num_features=train_X_tensor.shape[1]
    emb_num_features=emb_X_tensor.shape[1]    

    if args.model_type=='fm':
        model=FactorizationMachine(train_preprocess.num_features,args.num_factors,args)
    elif args.model_type=='deepfm':
        model=DeepFM(naive_num_features,emb_num_features,args.num_factors,args)

    #model=DeepFM(preprocess.num_features,args.num_factors,args)
    pl.trainer.Trainer(max_epochs=args.num_epochs_training).fit(model,train_dataloader)
    return model,train_df




if __name__ == '__main__':
    args=parser()
    results=[]
    for i in range(1,6):
        args.fold=i
        train_df ,test,movie_info,user_info,matrix,custom_object=getdata(args)
    
        
        # train_df is original version of train_df
        train_df=custom_object.original_merge()
        print("fold ",i," data loaded")
        model,train_preprocess=trainer(args,train_df,train_df)
        tester=Tester(args,model,train_df,test,movie_info,user_info)
        result=tester.test()
        results.append(result)

    for i in range(5):
        print("fold ",i+1," result:",end=" ")
        print(results[i])

    # types=['SVD','AE']
    # svdresults=[]
    # aeresults=[]
    # originalresults=[]
    # for t in types:
    #     args.embedding_type=t

    #     for i in range(1,6):
    #         args.fold=i
    #         train_df ,test,movie_info,user_info,matrix,custom_object=getdata(args)
            
    #         if args.embedding_type=='SVD':
                
    #             svd=SVD(args)
    #             user_embedding,movie_embedding=svd.get_embedding(matrix)

    #             # emb_train df is embedded version of train_df
    #             emb_train_df=custom_object.embedding_merge(user_embedding=user_embedding,movie_embedding=movie_embedding)
    #             print("fold ",i," data loaded")
    #             # train_df is original version of train_df
    #             train_df=custom_object.original_merge()
    #             model,train_preprocess=trainer(args,emb_train_df,emb_train_df)
                
                
                
                
    #             tester=Tester(args,model,train_df,test,movie_info,user_info)
    #             result=tester.test(user_embedding=user_embedding,movie_embedding=movie_embedding)
    #             svdresults.append(result)





    #         elif args.embedding_type=='AE':
    #             user_encoder, movie_encoder,user_x,movie_x=learn_encoder(args,matrix)
    #             user_encoder.eval()
    #             movie_encoder.eval()
    #             user_embedding=user_encoder.encode(user_x).detach().numpy()
    #             movie_embedding=movie_encoder.encode(movie_x).detach().numpy()
    #             emb_train_df=custom_object.embedding_merge(user_embedding=user_embedding,movie_embedding=movie_embedding)
                
    #             print("fold ",i," data loaded")
    #             train_df=custom_object.original_merge()
    #             model,train_preprocess=trainer(args,emb_train_df,emb_train_df)

                
    #             tester=Tester(args,model,train_df,test,movie_info,user_info)
    #             result=tester.test(user_embedding=user_embedding,movie_embedding=movie_embedding)
    #             aeresults.append(result)


    
    # for key in ['AE','SVD']:
    #     print(key," results embedding only for the deep part")
    #     for i in range(5):
    #         print("fold ",i+1," result:",end=" ")
    #         if key=='AE':
    #             print(aeresults[i])
    #         elif key=='SVD':
    #             print(svdresults[i])
    #         else:
    #             print(originalresults[i])
    # # drop all columns including user_id and movie_id in str columnname
    
    # #train_df=train_df.loc[:,~train_df.columns.str.contains('user_id')]
    # #train_df=train_df.loc[:,~train_df.columns.str.contains('movie')]
    # print(train_df)    



