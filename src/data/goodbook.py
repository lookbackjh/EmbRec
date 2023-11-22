import pandas as pd

class GoodBook:
    def __init__(self, args):
        self.args=args
        #self.fold=fold #should be integer

    def data_getter(self):
        
        #train, test loading for each fold
        train,test=self.train_test_getter()
        movie_info=self.movie_getter()
        user_info=self.user_getter()
        #ui_matrix=self.get_user_item_matrix()

        # change column names movie_id to item_id
        train=train.rename(columns={'book_id':'item_id'})
        test=test.rename(columns={'book_id':'item_id'})
        # add column item_id to movie_info
        movie_info.rename(columns={'book_id':'item_id'},inplace=True)


        return train,test,movie_info,user_info
    

    def train_test_getter(self):
        train=pd.read_csv('dataset/goodbook/ratings.csv')
        train=train.sort_values(by=['user_id'])
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
        user_info=pd.read_csv('dataset/goodbook/user_info.csv')
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
    