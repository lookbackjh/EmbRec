import pandas as pd
import numpy as np
from src.data.custompreprocess import CustomOneHot
from src.data.fm_preprocess import FM_Preprocessing
import tqdm
import torch
import copy

class Emb_Test:

    def __init__(self, args,model,train_df,test_df, le, movie_df, user_df) -> None:

        self.args = args
        self.train_df = train_df
        self.test_org = test_df
        self.movie_df = movie_df
        self.user_df = user_df
        self.model=model
        self.le=le  # le is labelencoder
        #self.original_df=pd.read_csv('dataset/ml-100k/u'+str(args.fold)+'.base',sep='\t',header=None, names=['user_id','movie_id','rating','timestamp'],encoding='latin-1')


    def test_data_generator(self):
        # want to make a dataframe that has user_id, movie_id and for every user_id, movie_id pair 
        movie_ids=sorted(self.train_df['movie_id'].unique())
        user_ids=sorted(self.train_df['user_id'].unique())



        # make a dataframe that has all the user_id, movie_id pairs
        npuser_movie=np.zeros((len(user_ids)*len(movie_ids),4))
        npuser_movie=npuser_movie.astype(int)
        npuser_movie[:,0]=np.repeat(user_ids,len(movie_ids))
        npuser_movie[:,1]=np.tile(movie_ids,len(user_ids))
        #2nd column is target
        npuser_movie[:,2]=1
        # 3rd column is c
        npuser_movie[:,3]=1

        user_movie=pd.DataFrame(npuser_movie,columns=['user_id','movie_id','target','c'])

        #c=CustomOneHot(self.args,user_movie,self.movie_df,self.user_df)
        user_list=user_movie['user_id']
        movie_list=user_movie['movie_id']
        #user_movie=c.movieonehot()
        user_movie['user_id']=user_movie['user_id'].astype(int)
        user_movie['movie_id']=user_movie['movie_id'].astype(int)
        target=user_movie['target'].astype(int)
        c=user_movie['c'].astype(int)
        user_movie.drop(['target','c'],axis=1,inplace=True)

        movieinfoadded=pd.merge(user_movie,self.movie_df,on='movie_id',how='left')

        userinfoadded=pd.merge(movieinfoadded,self.user_df,on='user_id',how='left')

        for col in userinfoadded.columns:
            userinfoadded[col].astype(object)
            userinfoadded[col]=self.le[col].fit_transform(userinfoadded[col])
        
        return userinfoadded,user_list,movie_ids

    def get_metric(self,pred,real):
        # pred is a list of top 5 recommended product code
        #(len(set(self.recommended_products).intersection(set(actual)))/len(self.recommended_products))
        precision=len(set(pred).intersection(set(real)))/len(pred)
        return precision
    
    def test(self,user_embedding=None,movie_embedding=None):
        
        test_df,user_list,movie_list=self.test_data_generator()
        user_list=user_list.astype(int).unique().tolist()
        #movie_list=movie_list.tolist()
        self.model.eval()
        precisions=[]
        for customerid in tqdm.tqdm(user_list[:]):

            #if self.args.embedding_type=='original':
            cur_customer_id='user_id_'+str(customerid)
            temp=test_df[self.le['user_id'].inverse_transform(test_df['user_id'])==customerid]
            X_org=temp.values
            X_tensor_org= torch.tensor(X_org, dtype=torch.int64)



            result=self.model.forward(X_tensor_org)
            topidx=torch.argsort(result,descending=True)[:]
            #swith tensor to list
            topidx=topidx.tolist()


            print("customer id: ",customerid, end=" ")
            ml=self.le['movie_id'].inverse_transform(temp['movie_id'].unique())
            ml=np.array(ml)
            # reorder movie_list
            ml=ml[topidx]
            cur_userslist=np.array(self.train_df[(self.train_df['user_id'])==customerid]['movie_id'].unique())
            # erase the things in ml that are in cur_userslist without changing the order
            real_rec=np.setdiff1d(ml,cur_userslist,assume_unique=True)
            
            print("top {} recommended product code: ".format(self.args.topk),real_rec[:5])

            cur_user_test=np.array(self.test_org[(self.test_org['user_id'])==customerid])
            cur_user_test=cur_user_test[:,1]
            cur_user_test=np.unique(cur_user_test)
            cur_user_test=cur_user_test.tolist()
            if(len(cur_user_test)==0 or len(cur_user_test)<self.args.topk):
                continue
            print("real product code: ",cur_user_test[:])
            real_rec=real_rec.tolist()

            precision=self.get_metric(real_rec[:self.args.topk],cur_user_test)
            precisions.append(precision)
  
            print("precision: ",precision)
        print("average precision: ",np.mean(precisions))
        return np.mean(precisions)

        