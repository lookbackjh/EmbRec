import pandas as pd

class Movielens1m:
    def __init__(self, args):
        self.args=args
        #self.fold=fold #should be integer

    def data_getter(self):
        
        #train, test loading for each fold
        train,test=self.train_test_getter()
        movie_info=self.movie_getter()
        user_info=self.user_getter()
        ui_matrix=self.get_user_item_matrix()

        # change column names movie_id to item_id
        train=train.rename(columns={'movie_id':'item_id'})
        test=test.rename(columns={'movie_id':'item_id'})
        # add column item_id to movie_info
        movie_info.rename(columns={'movie_id':'item_id'},inplace=True)


        return train,test,movie_info,user_info,ui_matrix
    

    def train_test_getter(self):
        train=pd.read_csv('dataset/ml-1m/ratings.dat',sep='::',header=None, names=['user_id','movie_id','rating','timestamp'],encoding='latin-1')
        train=train.sort_values(by=['user_id','timestamp'])
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
        train_data=train.groupby('user_id').apply(lambda x: x.iloc[:int(len(x)*0.7)])
        test_data=train.groupby('user_id').apply(lambda x: x.iloc[int(len(x)*0.7):])
        train_data.reset_index(drop=True,inplace=True)
        test_data.reset_index(drop=True,inplace=True)



        return train_data,test_data

    def movie_getter(self):
        
        #read movie data
        movie_info=movie_info=pd.read_csv('dataset/ml-1m/movies.dat',sep='::',header=None, names=['movie_id','title','genre'],encoding='latin-1')

        #split genre column into one hot
        genre_list=[]
        for i in range(len(movie_info)):
            genre_list.extend(movie_info['genre'][i].split('|'))
        genre_list=list(set(genre_list))
        #genre_list.remove('(no genres listed)')
        genre_list
        for i in genre_list:
            movie_info[i]=0
        for i in range(len(movie_info)):
            for j in genre_list:
                if j in movie_info['genre'][i]:
                    movie_info.loc[i,j]=1
        movie_info
        movie_info.drop(columns=['title','genre'],axis=1,inplace=True)
        return movie_info

    def user_getter(self):
        
        #simple preproccess of user_data
        user_info=pd.read_csv('dataset/ml-1m/users.dat',sep="::",header=None,names=['user_id','gender','age','occupation','zipcode'],encoding='latin-1')
        user_info.drop(['zipcode'],axis=1,inplace=True)
        #user_info['user_id']=user_info.index
        #user_info=pd.get_dummies(columns=['occupation'],data=user_info)
        user_info['gender'] = [1 if i == 'M' else 0 for i in user_info['gender']]
        # want to discretize age category  
        user_info['age'] = pd.cut(user_info['age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90], labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])
        #user_info['user_id']=user_info['user_id']+1

        return user_info
    
    def get_user_item_matrix(self):

        #get useritem matrix
        train,_=self.train_test_getter()
        useritem_matrix=train.pivot_table(index='user_id',columns='movie_id',values='rating')
        useritem_matrix=useritem_matrix.fillna(0)
        useritem_matrix=useritem_matrix.astype(float)
        useritem_matrix[useritem_matrix >= 1] = 1
        useritem_matrix = useritem_matrix.to_numpy()
        # x dtype to float
        useritem_matrix=useritem_matrix.astype(float)  

        return useritem_matrix
    