import pandas as pd
import numpy as np
from src.data.custompreprocess import CustomOneHot
from src.data.fm_preprocess import FM_Preprocessing
import tqdm
import torch
import copy

class Emb_Test:

    def __init__(self, args,model,train_df,test_df, le, movie_df, user_df,user_embedding,item_embedding) -> None:

        self.args = args
        self.train_df = train_df
        self.test_org = test_df
        self.movie_df = movie_df
        self.user_df = user_df
        self.model=model
        self.le=le  # le is labelencoder
        self.user_embedding=user_embedding
        self.item_embedding=item_embedding    
        #self.original_df=pd.read_csv('dataset/ml-100k/u'+str(args.fold)+'.base',sep='\t',header=None, names=['user_id','item_id','rating','timestamp'],encoding='latin-1')


    def test_data_generator(self):
        # want to make a dataframe that has user_id, item_id and for every user_id, item_id pair 
        item_ids=sorted(self.train_df['item_id'].unique())
        user_ids=sorted(self.train_df['user_id'].unique())



        # make a dataframe that has all the user_id, item_id pairs
        npuser_movie=np.zeros((len(user_ids)*len(item_ids),4))
        npuser_movie=npuser_movie.astype(int)
        npuser_movie[:,0]=np.repeat(user_ids,len(item_ids))
        npuser_movie[:,1]=np.tile(item_ids,len(user_ids))
        #2nd column is target
        npuser_movie[:,2]=1
        # 3rd column is c
        npuser_movie[:,3]=1

        user_movie=pd.DataFrame(npuser_movie,columns=['user_id','item_id','target','c'])

        #c=CustomOneHot(self.args,user_movie,self.movie_df,self.user_df)
        user_list=user_movie['user_id']
        movie_list=user_movie['item_id']
        #user_movie=c.movieonehot()
        user_movie['user_id']=user_movie['user_id'].astype(int)
        user_movie['item_id']=user_movie['item_id'].astype(int)
        target=user_movie['target'].astype(int)
        c=user_movie['c'].astype(int)
        user_movie.drop(['target','c'],axis=1,inplace=True)

        movieinfoadded=pd.merge(user_movie,self.movie_df,on='item_id',how='left')

        userinfoadded=pd.merge(movieinfoadded,self.user_df,on='user_id',how='left')

        for col in userinfoadded.columns:
            userinfoadded[col].astype(object)
            userinfoadded[col]=self.le[col].fit_transform(userinfoadded[col])

        cat_cols=userinfoadded.columns
        user_embedding_df=pd.DataFrame()
        item_embedding_df=pd.DataFrame()

        user_embedding_df['user_id']=sorted(userinfoadded['user_id'].unique())

        item_embedding_df['item_id']=sorted(userinfoadded['item_id'].unique())

        for i in range(self.user_embedding.shape[1]):
            user_embedding_df['user_embedding_'+str(i)]=self.user_embedding[:,i]

        for i in range(self.item_embedding.shape[1]):
            item_embedding_df['item_embedding_'+str(i)]=self.item_embedding[:,i]
        
        
        #없는건 0으로 처리해줘야함. 

        movie_emb_included_df=pd.merge(userinfoadded.set_index('item_id'), item_embedding_df,on='item_id',how='left')
        user_emb_included_df=pd.merge(movie_emb_included_df.set_index('user_id'),user_embedding_df, on='user_id',how='left')
        
        # cat_df=user_emb_included_df[cat_cols]
        # cat_df=cat_df.astype(int)

        # cont_df=user_emb_included_df.drop(cat_cols,axis=1)
        # cont_df=cont_df.astype(float)

        return user_emb_included_df,user_list,item_ids,cat_cols

    def get_metric(self,pred,real):
        # pred is a list of top 5 recommended product code
        #(len(set(self.recommended_products).intersection(set(actual)))/len(self.recommended_products))
        precision=len(set(pred).intersection(set(real)))/len(pred)
        return precision
    
    def test(self,user_embedding=None,movie_embedding=None):
        
        
        train_movie=self.train_df['item_id'].unique()
        test_movie=self.test_org['item_id'].unique()
        diff=np.setdiff1d(test_movie,train_movie)


        test_df,user_list,item_ids,catcols=self.test_data_generator()
        user_list=user_list.astype(int).unique().tolist()
        #movie_list=movie_list.tolist()
        self.model.eval()
        precisions=[]
        for customerid in tqdm.tqdm(user_list[:]):

            #if self.args.embedding_type=='original':
            cur_customer_id='user_id_'+str(customerid)
            temp=test_df[self.le['user_id'].inverse_transform(test_df['user_id'])==customerid]
            

            if self.args.embedding_type=='SVD':
                X_cat=temp[catcols].drop(['user_id','item_id'],axis=1).values
            else:
                X_cat=temp[catcols].values

            
            X_cat=torch.tensor(X_cat, dtype=torch.int64)
            X_cont=temp.drop(catcols,axis=1).values
            X_cont=torch.tensor(X_cont, dtype=torch.float32)
            
            # X_org=temp.values
            # X_tensor_org= torch.tensor(X_org, dtype=torch.int64)



            result=self.model.forward(X_cat,X_cont)
            topidx=torch.argsort(result,descending=True)[:]
            #swith tensor to list
            topidx=topidx.tolist()


            print("customer id: ",customerid, end=" ")
            ml=self.le['item_id'].inverse_transform(temp['item_id'].unique())
            ml=np.array(ml)
            # reorder movie_list
            ml=ml[topidx]
            cur_userslist=np.array(self.train_df[(self.train_df['user_id'])==customerid]['item_id'].unique())
            # erase the things in ml that are in cur_userslist without changing the order
            real_rec=np.setdiff1d(ml,cur_userslist,assume_unique=True)
            
            print("top {} recommended product code: ".format(self.args.topk),real_rec[:5])

            cur_user_test=np.array(self.test_org[(self.test_org['user_id'])==customerid])
            cur_user_test=cur_user_test[:,1]
            cur_user_test=np.unique(cur_user_test)
            cur_user_test=cur_user_test.tolist()
            cur_user_test=np.setdiff1d(cur_user_test,diff,assume_unique=True)
            if(len(cur_user_test)==0 or len(cur_user_test)<self.args.topk):
                continue
            print("real product code: ",cur_user_test[:])
            real_rec=real_rec.tolist()

            precision=self.get_metric(real_rec[:self.args.topk],cur_user_test)
            precisions.append(precision)
  
            print("precision: ",precision)
        print("average precision: ",np.mean(precisions))
        return np.mean(precisions)

        