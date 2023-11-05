from scipy.sparse.linalg import svds
class SVD:

    def __init__(self,args) -> None:
        self.args=args
        pass


    def fit_svd(self,x):
        u, s, vt = svds(x, k = self.args.num_eigenvector)
        return u,s,vt.T

    def get_embedding(self,x):

        #temp_data = x[['user_id', 'movie_id', 'rating']]
        # #pivot_data = temp_data.pivot(index = 'user_id', columns = 'movie_id', values = 'rating')
        # pivot_data = pivot_data.fillna(0)
        # pivot_data[pivot_data >= 1] = 1
        # pivot_data = pivot_data.to_numpy()
        # # x dtype to float
        # pivot_data=pivot_data.astype(float)  
        u,s,vt= self.fit_svd(x)
        return u,vt
