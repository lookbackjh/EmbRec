from src.data.movielens100k import Movielens100k
from src.data.movielens1m import Movielens1m
class DataWrapper:

    def __init__(self,args) -> None:
        pass

        if args.datatype=="ml100k":
            self.data=Movielens100k('dataset/ml-100k','u.data', args.fold)
        elif args.datatype=="ml1m":
            self.data=Movielens1m(args)
        else:
            raise NotImplementedError

        self.train,self.test,self.item_info,self.user_info,self.ui_matrix=self.data.data_getter()


    def get_data(self):
        return self.train,self.test,self.item_info,self.user_info,self.ui_matrix