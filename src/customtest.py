import pandas as pd
import numpy as np
from src.data.custompreprocess import CustomOneHot
from src.data.fm_preprocess import FM_Preprocessing
import tqdm
import torch
import copy

class Emb_Test:

    def __init__(self, args,model,train_df,test_df, le, movie_df, user_df,user_embedding,item_embedding,catcol,contcol,train_org) -> None:

        self.args = args
        self.train_df = train_df
        self.test_org = test_df
        self.movie_df = movie_df
        self.user_df = user_df
        self.model=model
        self.le=le  # le is labelencoder
        self.user_embedding=user_embedding
        self.item_embedding=item_embedding
        self.catcol=catcol
        self.contcol=contcol
        self.train_org=train_org
        #self.original_df=pd.read_csv('dataset/ml-100k/u'+str(args.fold)+'.base',sep='\t',header=None, names=['user_id','item_id','rating','timestamp'],encoding='latin-1')


    def test_data_generator(self):
        # want to make a dataframe that has user_id, item_id and for every user_id, item_id pair 
        item_ids=self.le['item_id'].classes_
        user_ids=self.le['user_id'].classes_




        # make a dataframe that has all the user_id, item_id pairs
        #tqdm  tochecek times for code below


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

        # for col in self.movie_df.columns:
        #     if col=='item_id':
        #         continue
        #     self.movie_df[col]=self.le[col].transform(self.movie_df[col])
        
        # for col in self.user_df.columns:
        #     if col=='user_id':
        #         continue
        #     self.user_df[col]=self.le[col].transform(self.user_df[col])
        #npembedding=np.zeros(len(user_id)*)
        movieinfoadded=pd.merge(user_movie,self.movie_df,on='item_id',how='left')
        

        userinfoadded=pd.merge(movieinfoadded,self.user_df,on='user_id',how='left')



        #cat_cols=userinfoadded.columns
        user_embedding_df=pd.DataFrame()
        item_embedding_df=pd.DataFrame()

        user_embedding_df['user_id']=sorted(userinfoadded['user_id'].unique())

        item_embedding_df['item_id']=sorted(userinfoadded['item_id'].unique())

        for i in tqdm.tqdm(range(self.user_embedding.shape[1])):
            user_embedding_df['user_embedding_'+str(i)]=self.user_embedding[:,i]

        for i in tqdm.tqdm(range(self.item_embedding.shape[1])):
            item_embedding_df['item_embedding_'+str(i)]=self.item_embedding[:,i]
        
#call the progress_apply feature with a dummy lambda function
        
        #movie_emb_included_df=pd.merge(userinfoadded.set_index('item_id'), item_embedding_df,on='item_id',how='left').progress_apply(lambda x: x)
        #user_emb_included_df=pd.merge(movie_emb_included_df.set_index('user_id'),user_embedding_df, on='user_id',how='left').progress_apply(lambda x: x)
        
        # please change operation above to join
        # do not use merge
        movie_emb_np=np.zeros((len(user_ids)*len(item_ids),self.item_embedding.shape[1]))
        for i in tqdm.tqdm(range(0,(self.item_embedding.shape[1]))):
            movie_emb_np[:,i]=np.tile(self.item_embedding[:,i],len(user_ids))

        user_emb_np=np.zeros((len(user_ids)*len(item_ids),self.user_embedding.shape[1]))
        for i in tqdm.tqdm(range(0,self.user_embedding.shape[1])):
            user_emb_np[:,i]=np.repeat(self.user_embedding[:,i],len(item_ids))


        movie_emb_df=pd.DataFrame(movie_emb_np,columns=item_embedding_df.columns.tolist()[1:]) 
        user_emb_df=pd.DataFrame(user_emb_np,columns=user_embedding_df.columns.tolist()[1:])
        # movie_emb_included_df=userinfoadded.set_index('item_id').join(item_embedding_df,on='item_id',how='left')
        # user_emb_included_df=movie_emb_included_df.set_index('user_id').join(user_embedding_df, on='user_id',how='left')
        #pd.concat([userinfoadded,movie_emb_df],axis=1)
        movie_emb_included_df=pd.concat([userinfoadded,movie_emb_df],axis=1)
        del userinfoadded
        user_emb_included_df=pd.concat([movie_emb_included_df,user_emb_df],axis=1)
        del movie_emb_included_df



        # cat_df=user_emb_included_df[cat_cols]
        # cat_df=cat_df.astype(int)
        for col in self.catcol:
            #user_emb_included_df[col].astype(object)
            user_emb_included_df[col]=self.le[col].transform(user_emb_included_df[col])
        # cont_df=user_emb_included_df.drop(cat_cols,axis=1)
        # cont_df=cont_df.astype(float)
        if self.args.embedding_type=='SVD':
            user_emb_included_df['user_id']=self.le['user_id'].transform(user_emb_included_df['user_id'])
            user_emb_included_df['item_id']=self.le['item_id'].transform(user_emb_included_df['item_id'])

        return user_emb_included_df

    def get_metric(self,pred,real):
        # pred is a list of top 5 recommended product code
        #(len(set(self.recommended_products).intersection(set(actual)))/len(self.recommended_products))
        precision=len(set(pred).intersection(set(real)))/len(pred)
        return precision
    
    def test(self,user_embedding=None,movie_embedding=None):
        
        
        train_movie=self.train_df['item_id'].unique()
        test_movie=self.test_org['item_id'].unique()
        diff=np.setdiff1d(test_movie,train_movie)
        #self.train_org=self.train_org['user_id','item_id']
        #user0movie=self.train_org[(self.train_org['user_id'])==0]['item_id'].unique()



        test_df=self.test_data_generator()
        user_list=test_df['user_id'].unique().tolist()
        #user_list=user_list.astype(int).unique().tolist()

        for col in self.train_org.columns:
            if col=='user_id' or col=='item_id':
                self.train_org[col]=self.le[col].transform(self.train_org[col])
        #movie_list=movie_list.tolist()
        # for col in self.catcol:
        #     test_df[col].astype(object)
        #     test_df[col]=self.le[col].transform(test_df[col])


        self.model.eval()
        precisions=[]
        for customerid in tqdm.tqdm(user_list[:]):

            #if self.args.embedding_type=='original':
            cur_customer_id='user_id_'+str(customerid)
            temp=test_df[(test_df['user_id'])==customerid]
            

            if self.args.embedding_type=='SVD':
                X_cat=temp[self.catcol].values
            else:
                X_cat=temp[self.catcol].values

            
            X_cat=torch.tensor(X_cat, dtype=torch.int64)
            X_cont=temp[self.contcol].values
            X_cont=torch.tensor(X_cont, dtype=torch.float32)
            
            # X_org=temp.values
            # X_tensor_org= torch.tensor(X_org, dtype=torch.int64)



            result=self.model.forward(X_cat,X_cont)
            topidx=torch.argsort(result,descending=True)[:]
            #swith tensor to list
            topidx=topidx.tolist()


            print("customer id: ",customerid, end=" ")
            ml=list(self.le['item_id'].inverse_transform(temp['item_id'].unique()))
            ml=np.array(ml)
            # reorder movie_list
            ml=ml[topidx]
            cur_userslist=np.array(self.train_org[(self.train_org['user_id'])==customerid]['item_id'].unique())
            
            # 여기 안본게 포함되어있을 수 있음 이거 처리해주어ㅑ하미
            cur_userslist=self.le['item_id'].inverse_transform(cur_userslist)
            
            # erase the things in ml that are in cur_userslist without changing the order
            real_rec=np.setdiff1d(ml,cur_userslist,assume_unique=True)
            


            print("top {} recommended product code: ".format(self.args.topk),real_rec[:self.args.topk])

            if self.le['user_id'].inverse_transform([customerid])[0] not in self.test_org['user_id'].unique():
                continue
            
            cur_user_test=np.array(self.test_org[(self.test_org['user_id'])==self.le['user_id'].inverse_transform([customerid])[0]])
            cur_user_test=cur_user_test[:,1]
            cur_user_test=np.unique(cur_user_test)
            cur_user_test=cur_user_test.tolist()

            test_seem=[]
            for u in cur_user_test:
                # if le['item_id'].inverse_transform([u]) gives errer
                try:
                    test_seem.append(self.le['item_id'].transform([u])[0])
                except:
                    continue
            cur_user_test=test_seem

            cur_user_test=self.le['item_id'].inverse_transform(cur_user_test)
            #cur_user_test=np.setdiff1d(cur_user_test,diff,assume_unique=True)
            if(len(cur_user_test)==0 or len(cur_user_test)<self.args.topk):
                continue
            print("real product code: ",cur_user_test[:])
            real_rec=real_rec.tolist()

            precision=self.get_metric(real_rec[:self.args.topk],cur_user_test)
            precisions.append(precision)
  
            print("precision: ",precision)
        print("average precision: ",np.mean(precisions))
        # totla user number and total item number
        print("total user number: ",len(user_list))
        print("total item number: ",len(self.train_df['item_id'].unique()))
        return np.mean(precisions)

        