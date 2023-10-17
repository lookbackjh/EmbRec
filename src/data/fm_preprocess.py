import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import random
from copy import deepcopy
class FM_Preprocessing:
    def __init__(self, args,df, target_col='target', num_epochs=10):
        self.args=args
        self.df = df
        self.target_col = target_col
        self.num_epochs = num_epochs
        #self.user_id=df['AUTH_CUSTOMER_ID']
        self.X_tensor, self.y_tensor, self.c_values_tensor, self.user_feature_tensor, self.item_feature_tensor, self.all_item_ids, self.num_features,self.dics = self.prepare_data()
        if not isinstance(df, pd.DataFrame):
            raise ValueError("The df parameter should be a pandas DataFrame.")
        
        if target_col not in df.columns:
            raise ValueError(f"The target column {target_col} is not in the DataFrame.")


    
    
    # def get_c(self, df, alpha=.5, beta=.5, gamma=.5, c_zero=.5):
    #     UF = np.array(df["customer_frequency"].astype("float"), dtype=float)
    #     UF /= df.shape[0]
    #     IF = np.array(df["product_frequency"].astype("float"), dtype=float)
    #     IF /= df.shape[0]
    #     Fs = alpha * beta * IF * UF
    #     Fs_gamma = Fs ** gamma
    #     c = c_zero / np.sum(Fs_gamma) * Fs_gamma
    #     c_appended_df = deepcopy(df)
    #     c_appended_df['C'] = c

    #     return c_appended_df
    

    def prepare_data(self):
        #X_new=self.generate_not_purchased_data(self.df)
        X = self.df #temporary

        # X = preprocess_positive(X) # pls preprocess postive either
        #y = self.df[self.target_col]
        c = self.df['c']
        if self.args.embedding_type=='original':
            X=X.drop(['target','c','user_id','movie_id'],axis=1,inplace=False)
        else:
            X=X.drop(['target','c','user_id','movie_id'],axis=1,inplace=False)
        y=self.df['target']
        # there are booleans in dataframe X and I want to change dtype of the data to float
        X = X.astype(float)

        X_tensor = torch.tensor(X.values, dtype=torch.int64)
        y_tensor = torch.tensor(y.values, dtype=torch.float32).view(-1)

        c_values_tensor = torch.tensor(c, dtype=torch.float32)
        c_values_tensor = torch.where(c_values_tensor < 1, c_values_tensor*1 , c_values_tensor)

        # want to make user_id and product_id mapping dictionary

        # load dict.json
        # import json
        # with open('dict1.json') as json_file:
        #     dics = json.load(json_file)
            
        # unique_user_df = self.df.drop_duplicates(subset=['AUTH_CUSTOMER_ID']).sort_values('AUTH_CUSTOMER_ID')
        # user_features_df = unique_user_df[['AUTH_CUSTOMER_ID','BIRTH_YEAR', 'GENDER','customer_frequency']]
        # user_feature_tensor = torch.tensor(pd.get_dummies(user_features_df).values, dtype=torch.float32)

        # unique_item_df = self.df.drop_duplicates(subset=['PRODUCT_CODE']).sort_values('PRODUCT_CODE')
        # #item_features_df = unique_item_df.filter(like='DEPTH')
        # item_feature_df = unique_item_df[[ 'product_frequency']]
        # item_feature_tensor = torch.tensor(item_feature_df.values, dtype=torch.float32)

        # all_item_ids = list(self.df.PRODUCT_CODE.unique())

        # num_features = X.shape[1]
        user_feature_tensor = 1
        item_feature_tensor = 1
        all_item_ids = 1
        num_features = X.shape[1]
        dics=1
        
        return X_tensor, y_tensor, c_values_tensor, user_feature_tensor, item_feature_tensor, all_item_ids, num_features,dics

