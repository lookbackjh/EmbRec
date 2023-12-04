import pandas as pd
class ShoppingData:
    def __init__(self, args):
        self.args=args
        #self.fold=fold #should be integer

    def data_getter(self):
        
        #train, test loading for each fold
        train,test=self.train_test_getter()
        movie_info=self.movie_getter()
        user_info=self.user_getter()
        #ui_matrix=self.get_user_item_matrix()

        # train.rename(columns={'product_id':'item_id'},inplace=True)
        # test.rename(columns={'product_id':'item_id'},inplace=True)
        # add column item_id to movie_info
        movie_info.rename(columns={'product_id':'item_id'},inplace=True)


        return train,test,movie_info,user_info
    

    def train_test_getter(self):
        #train=pd.read_csv('dataset/shopping/ratings_148.csv')
        filestr='dataset/shopping/ratings_'+str(self.args.shopping_file_num)+'.csv'
        train=pd.read_csv(filestr)
        train=train.sort_values(by=['user_id','timestamp'])
        train_list=[]
        test_list=[]
        user_ids=train['user_id'].unique()
        for i in user_ids:
            temp=train[train['user_id']==i]
            train_list.append(temp.iloc[:int(len(temp)*self.args.train_ratio)])
            test_list.append(temp.iloc[int(len(temp)*(self.args.train_ratio)):])
        train=pd.concat(train_list)
        test=pd.concat(test_list)

        return train,test

    def movie_getter(self):
        
        #read movie data
        movie_info=pd.read_csv('dataset/shopping/item_info.csv')
        #movie_info.drop('category_depth',axis=1,inplace=True)

        return movie_info

    def user_getter(self):
        
        #simple preproccess of user_data
        user_info=pd.read_csv('dataset/shopping/user_info.csv')
        #user_info.drop(['zipcode'],axis=1,inplace=True)
        #user_info['user_id']=user_info.index
        #user_info=pd.get_dummies(columns=['occupation'],data=user_info)
        #user_info.drop(['Unnamed: 0'],axis=1,inplace=True) 
        user_info['gender'] = [1 if i == 'M' else 0 for i in user_info['gender']]
        user_info['gender']= user_info['gender'].astype(int)
        # want to discretize age category  
        user_info['age'] = pd.cut(user_info['age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90], labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        user_info['age'] = user_info['age'].astype(int)
        #user_info['user_id']=user_info['user_id']+1

        return user_info
    
    def get_user_item_matrix(self):

        #get useritem matrix
        train,_=self.train_test_getter()
        useritem_matrix=train.pivot_table(index='user_id',columns='item_id',values='rating')
        useritem_matrix=useritem_matrix.fillna(0)
        useritem_matrix=useritem_matrix.astype(float)
        useritem_matrix[useritem_matrix >= 1] = 1
        useritem_matrix = useritem_matrix.to_numpy()
        # x dtype to float
        useritem_matrix=useritem_matrix.astype(float)  

        return useritem_matrix
    