import pandas as pd

class Frappe:
    def __init__(self, args):
        self.args=args
        #self.fold=fold #should be integer

    def data_getter(self):
        
        #train, test loading for each fold
        self.train,self.test=self.train_test_getter()
        self.train=self.train.rename(columns={'item':'item_id'})
        self.test=self.test.rename(columns={'item':'item_id'})
        movie_info=self.movie_getter()
        user_info=self.user_getter()
        #ui_matrix=self.get_user_item_matrix()

        # change column names movie_id to item_id

        # add column item_id to movie_info
        movie_info.rename(columns={'item':'item_id'},inplace=True)


        return self.train,self.test,movie_info,user_info
    

    def train_test_getter(self):
        train=pd.read_csv('dataset/frappe/meta_app_user_rating.csv')
        train=train.rename(columns={'user':'user_id'})
        train=train .sort_values(by=['user_id'])
        train.drop(columns=['daytime','weekday','isweekend','homework','weather','country','city','cnt','cost'], inplace=True)
        # train_list=[]
        # test_list=[]
        # user_ids=train['user_id'].unique()
        # for i in user_ids:
        #     temp=train[train['user_id']==i]
        #     train_list.append(temp.iloc[:int(len(temp)*self.args.train_ratio)])
        #     test_list.append(temp.iloc[int(len(temp)*(self.args.train_ratio)):])
        # train=pd.concat(train_list)
        # test=pd.concat(test_list)
        # can you do the same operation without for loop?
        #train.groupby('user_id').apply(lambda x: x.iloc[:int(len(x)*self.args.train_ratio)])
        #train.groupby('user_id').apply(lambda x: x.iloc[int(len(x)*self.args.train_ratio):])
        train['timestamp']=0



        train_data=train.groupby('user_id').apply(lambda x: x.iloc[:int(len(x)*0.7)])
        test_data=train.groupby('user_id').apply(lambda x: x.iloc[int(len(x)*0.7):])
        train_data.reset_index(drop=True,inplace=True)
        test_data.reset_index(drop=True,inplace=True)



        return train_data,test_data

    def movie_getter(self):
        
        #read movie data
        movie_info=pd.read_csv('dataset/frappe/meta_app_item_info.csv')
        movie_info=movie_info.rename(columns={'rating':'quality'})
        movie_info.drop(columns=['name'], inplace=True) 
        return movie_info

    def user_getter(self):
        
        #simple preproccess of user_data
        user_info=pd.read_csv('dataset/frappe/meta_user_id.csv')
        #user_info['user_id']=user_info['user_id']+1

        return user_info
    
    def get_user_item_matrix(self):

        #get useritem matrix
        train,_=self.train_test_getter()
        useritem_matrix=train.pivot_table(index='user_id',columns='item',values='rating')
        useritem_matrix=useritem_matrix.fillna(0)
        useritem_matrix=useritem_matrix.astype(float)
        useritem_matrix[useritem_matrix >= 1] = 1
        useritem_matrix = useritem_matrix.to_numpy()
        # x dtype to float
        useritem_matrix=useritem_matrix.astype(float)  

        return useritem_matrix
    