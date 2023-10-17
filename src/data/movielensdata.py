import pandas as pd
class MovielensData:
    def __init__(self, data_dir, data_file,fold):

        self.data_dir = data_dir
        self.data_file = data_file
        self.fold=fold #should be integer

    def data_getter(self):
        
        #train, test loading for each fold
        train=pd.read_csv('dataset/ml-100k/u'+str(self.fold)+'.base',sep='\t',header=None, names=['user_id','movie_id','rating','timestamp'],encoding='latin-1')
        test=pd.read_csv('dataset/ml-100k/u'+str(self.fold)+'.test',sep='\t',header=None, names=['user_id','movie_id','rating','timestamp'],encoding='latin-1')
       
        return train,test
    
    def movie_getter(self):
        
        #simple preproccess of movie_data
        movie_info=pd.read_csv('dataset/ml-100k/u.item',sep='|',header=None, names=['movie_id','movie_title','release_date','video_release_date','IMDb_URL','unknown','Action','Adventure','Animation','Children','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance','Sci-Fi','Thriller','War','Western'],encoding='latin-1')
        movie_info.drop(['movie_title','release_date','video_release_date','IMDb_URL'],axis=1,inplace=True)
        
        return movie_info

    def user_getter(self):
        
        #simple preproccess of user_data
        user_info=pd.read_csv('dataset/ml-100k/u.user',sep='|', names=['age','gender','occupation','zipcode'])
        user_info.drop(['zipcode'],axis=1,inplace=True)
        user_info['user_id']=user_info.index
        #user_info=pd.get_dummies(columns=['occupation'],data=user_info)
        user_info['gender'] = [1 if i == 'M' else 0 for i in user_info['gender']]
        # want to discretize age category  
        user_info['age'] = pd.cut(user_info['age'], bins=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90], labels=[0, 1, 2, 3, 4, 5, 6, 7, 8])

        return user_info
    