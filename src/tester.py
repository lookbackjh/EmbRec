import pandas as pd
import numpy as np
from src.data.custompreprocess import CustomOneHot
from src.data.fm_preprocess import FM_Preprocessing
import tqdm
import torch
import copy

class Tester:

    def __init__(self, args,model,train_df, test_df, movie_df, user_df) -> None:

        self.args = args
        self.train_df = train_df
        self.test_df = test_df
        self.movie_df = movie_df
        self.user_df = user_df
        self.model=model
        self.original_df=pd.read_csv('dataset/ml-100k/u'+str(args.fold)+'.base',sep='\t',header=None, names=['user_id','movie_id','rating','timestamp'],encoding='latin-1')

    def embedding_test_data_generator(self,user_embedding,movie_embedding):

        movie_ids=sorted(self.train_df['movie_id'].unique())
        user_ids=sorted(self.train_df['user_id'].unique())

        #dictionary that maps movie_id to movie_frequency
        # movie_frequency={}
        # for i in movie_ids:
        #     movie_frequency[i]=len(self.original_df[self.original_df['movie_id']==i])
        
        # #dictionary that maps user_id to user_frequency
        # user_frequency={}
        # for i in user_ids:
        #     user_frequency[i]=len(self.original_df[self.original_df['user_id']==i])


        # make a dataframe that has all the user_id, movie_id pairs
        npuser_movie=np.zeros((len(user_ids)*len(movie_ids),4))
        npuser_movie=npuser_movie.astype(int)
        npuser_movie[:,0]=np.repeat(user_ids,len(movie_ids))
        npuser_movie[:,1]=np.tile(movie_ids,len(user_ids))
        #2nd column is target
        npuser_movie[:,2]=1
        # 3rd column is c
        npuser_movie[:,3]=1
        # 4th column is user_frequency

        user_movie=pd.DataFrame(npuser_movie,columns=['user_id','movie_id','target','c'])


        c=CustomOneHot(self.args,user_movie,self.movie_df,self.user_df)
        user_list=user_movie['user_id']
        movie_list=user_movie['movie_id']
        user_movie=c.embedding_merge(user_embedding=user_embedding,movie_embedding=movie_embedding)
        user_movie['user_id']=user_movie['user_id'].astype(int)
        user_movie['movie_id']=user_movie['movie_id'].astype(int)


        # movieinfoadded=pd.merge(user_movie,self.movie_df,on='movie_id',how='left')

        # userinfoadded=pd.merge(movieinfoadded,self.user_df,on='user_id',how='left')
        
        #drop movie_id
        user_movie.drop(['movie_id'],axis=1,inplace=True)
        #userinfoadded.drop(['user_id'],axis=1,inplace=True)

        #pd.getdummies on user_id
        #user_df=pd.get_dummies(columns=['user_id'],data=userinfoadded)
        #print(user_df)
        #user_df.drop(['user_frequency'],axis=1,inplace=True)
        #user_df.drop(['movie_frequency'],axis=1,inplace=True)

        return user_movie,user_list,movie_ids
    



    def test_data_generator(self):
        # want to make a dataframe that has user_id, movie_id and for every user_id, movie_id pair 
        movie_ids=sorted(self.train_df['movie_id'].unique())
        user_ids=sorted(self.train_df['user_id'].unique())

        #dictionary that maps movie_id to movie_frequency
        movie_frequency={}
        for i in movie_ids:
            movie_frequency[i]=len(self.original_df[self.original_df['movie_id']==i])
        
        #dictionary that maps user_id to user_frequency
        user_frequency={}
        for i in user_ids:
            user_frequency[i]=len(self.original_df[self.original_df['user_id']==i])


        # make a dataframe that has all the user_id, movie_id pairs
        npuser_movie=np.zeros((len(user_ids)*len(movie_ids),6))
        npuser_movie=npuser_movie.astype(int)
        npuser_movie[:,0]=np.repeat(user_ids,len(movie_ids))
        npuser_movie[:,1]=np.tile(movie_ids,len(user_ids))
        #2nd column is target
        npuser_movie[:,2]=1
        # 3rd column is c
        npuser_movie[:,3]=1
        # 4th column is user_frequency
        npuser_movie[:,4]=[user_frequency[i] for i in npuser_movie[:,0]]
        # 5th column is movie_frequency
        npuser_movie[:,5]=[movie_frequency[i] for i in npuser_movie[:,1]]

        user_movie=pd.DataFrame(npuser_movie,columns=['user_id','movie_id','target','c','user_frequency','movie_frequency'])

        c=CustomOneHot(self.args,user_movie,self.movie_df,self.user_df)
        user_list=user_movie['user_id']
        movie_list=user_movie['movie_id']
        user_movie=c.movieonehot()
        user_movie['user_id']=user_movie['user_id'].astype(int)
        user_movie['movie_id']=user_movie['movie_id'].astype(int)


        movieinfoadded=pd.merge(user_movie,self.movie_df,on='movie_id',how='left')

        userinfoadded=pd.merge(movieinfoadded,self.user_df,on='user_id',how='left')
        
        #drop movie_id
        userinfoadded.drop(['movie_id'],axis=1,inplace=True)

        #pd.getdummies on user_id
        user_df=pd.get_dummies(columns=['user_id'],data=userinfoadded)
        user_df.drop(['user_frequency'],axis=1,inplace=True)
        user_df.drop(['movie_frequency'],axis=1,inplace=True)   
    

        print(user_df)
        
        return user_df,user_list,movie_ids

    def get_metric(self,pred,real):
        # f1 score = 2 * (precision * recall) / (precision + recall)
        #(len(set(self.recommended_products).intersection(set(actual)))/len(self.recommended_products))
        precision=len(set(pred).intersection(set(real)))/len(pred)
        return precision
    
    def test(self,user_embedding=None,movie_embedding=None):

        original_df,user_list,movie_list=self.test_data_generator()
        user_list=user_list.astype(int).unique().tolist()
        self.model.eval()
        precisions=[]
        for customerid in tqdm.tqdm(user_list[:]):
            #if self.args.embedding_type=='original':
            cur_customer_id='user_id_'+str(customerid)
            temp=original_df[original_df[cur_customer_id]==1]
            X_org=temp.drop(['c','target'],axis=1).values
            y=temp['target'].values
            X_org=X_org.astype(float)
            y=y.astype(float)
            X_tensor_org= torch.tensor(X_org, dtype=torch.int64)
            y_tensor = torch.tensor(y, dtype=torch.float32).view(-1)
            result=self.model.forward(X_tensor_org,X_tensor_org)
            # get top k indices
            topidx=torch.argsort(result,descending=True)[:]
            topidx=topidx.tolist()
            # recommendation for customerid
            print("customer id: ",customerid, end=" ")
            ml=copy.deepcopy(movie_list)    
            ml=np.array(ml)
            # reorder movie_list
            ml=ml[topidx]
            #print(ml)
            cur_userslist=np.array(self.original_df[self.original_df['user_id']==customerid]['movie_id'].unique())

            # erase the things in ml that are in cur_userslist without changing the order
            real_rec=np.setdiff1d(ml,cur_userslist,assume_unique=True)
            
            print("top {} recommended product code: ".format(self.args.topk),real_rec[:5])

            cur_user_test=np.array(self.test_df[self.test_df['user_id']==customerid])
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

        