from copy import deepcopy
import numpy as np
import pandas as pd
import tqdm
class NegativeSampler:
    #  takes input of original dataframe and movie info
    #  make a function that returns negative sampled data

    def __init__(self, args,original_df) -> None:
        self.args=args
        self.original_df = original_df
        self.seed=np.random.seed(args.seed)
        self.original_df.drop(columns=['timestamp','rating'], axis=1, inplace=True)
        self.original_df['target']=1
        self.original_df['c']=1
        self.movie_info=pd.read_csv('dataset/ml-100k/u.item',sep='|',header=None, names=['movie_id','movie_title','release_date','video_release_date','IMDb_URL','unknown','Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'],encoding='latin-1')
        self.movie_info.drop(['movie_title','release_date','video_release_date','IMDb_URL'],axis=1,inplace=True)
        self.movie_info['movie_id']=self.movie_info['movie_id'].astype(int)

        self.user_info=pd.read_csv('dataset/ml-100k/u.user',sep='|', names=['age','gender','occupation','zipcode'])
        self.user_info.drop(['zipcode'],axis=1,inplace=True)
        self.user_info['user_id']=self.user_info.index
        pass

    


    # function taht calculates the c value for each customer and product
    # the higher beta means more weiht on the product frequency, the higher alpha means more weight on the customer frequency
    def get_c(self,df,uu_sum,ii_sum, alpha=.5, beta=.5, gamma=.5, c_zero=10):
        c_zero = self.args.c_zeros
        UF = np.array(df["user_frequency"].astype("float"), dtype=float)
        UF /= uu_sum
        IF = np.array(df["movie_frequency"].astype("float"), dtype=float)
        IF /= ii_sum
        Fs = alpha * beta * IF * UF
        Fs_gamma = Fs ** gamma
        c = Fs_gamma / np.sum(Fs_gamma)
        c = c_zero * c / np.max(c)
        c_appended_df = deepcopy(df)
        c_appended_df['c'] = c

        return c_appended_df
    
    def negativesample(self,isuniform=False):

        unique_customers = self.original_df['user_id'].unique()
        df=self.original_df
        not_purchased_df = pd.DataFrame()
        ns_df_list = []
        df['user_frequency'] = df.groupby('user_id')['user_id'].transform('count')
        df['movie_frequency'] = df.groupby('movie_id')['movie_id'].transform('count')
        #multiprocess

        print("Negative Sampling Started")

        for customer in tqdm.tqdm(unique_customers):
            unique_products = df['movie_id'].unique()


            customer_frequency = df[df['user_id'] == customer]['user_frequency'].iloc[0]
            purchased_products = df[df['user_id'] == customer]['movie_id'].unique()

            #customer_birth_category = df[df['user_id'] == customer]['BIRTH_YEAR'].iloc[0]
            #customer_gender_category = df[df['user_id'] == customer]['GENDER'].iloc[0]

            not_purchased_df_all = df[~df['movie_id'].isin(purchased_products)]
            not_purchased_codes = not_purchased_df_all['movie_id'].unique()
            negative_sample_products=np.random.choice(not_purchased_codes, int(len(not_purchased_codes) *self.args.ratio_negative),replace=False)
            ns_test=df[df['movie_id'].isin(negative_sample_products)][['movie_id','movie_frequency']]
            ns_test=ns_test.drop_duplicates(subset=['movie_id'],keep='first',inplace=False)
            ns_df=pd.DataFrame()

            ns_df['movie_id'] = negative_sample_products
            ns_df=ns_df.assign(user_id=customer)
            #ns_df['AUTH_CUSTOMER_ID'] = customer
            #ns_df=ns_df.assign(BIRTH_YEAR = customer_birth_category)
            #ns_df=ns_df.assign(GENDER= customer_gender_category)
            ns_df=ns_df.assign(user_frequency = customer_frequency)
            ns_df=ns_df.join(ns_test.set_index('movie_id'), on='movie_id')
            # not_purchased_df=pd.concat([not_purchased_df,ns_df],axis=0, ignore_index=True)
            ns_df_list += [ns_df]
        not_purchased_df=pd.concat(objs=ns_df_list, axis=0, ignore_index=True)
        
        not_purchased_df['target'] = 0

        mm=self.movie_info['movie_id'].map(self.original_df['movie_id'].value_counts())
        #change nan to 0
        mm.fillna(0,inplace=True)
        mm_sum=np.sum(mm.tolist())


        uu=self.user_info['user_id'].map(self.original_df['user_id'].value_counts())
        #change nan to 0
        uu.fillna(0,inplace=True)
        uu_sum=np.sum(uu.tolist())

        
        if isuniform:
            not_purchased_df['c'] = 1
            
        
        else:
            not_purchased_df=self.get_c(not_purchased_df,uu_sum=uu_sum,ii_sum=mm_sum)


        print(not_purchased_df)

        to_return = pd.concat([self.original_df, not_purchased_df], axis=0, ignore_index=True)
        #print(to_return)
        to_return.drop(['user_frequency','movie_frequency'],axis=1,inplace=True)
        print("Negative Sampling Finished")
        return to_return
    


    


