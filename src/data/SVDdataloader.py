import torch.utils.data as data_utils

class SVDDataloader(data_utils.Dataset):
    # as we already converted to tensor, we can directly return the tensor
    def __init__(self,x,emb,ui,cons,y,c) -> None:
        self.x=x
        self.emb=emb
        self.cons=cons
        self.y=y
        self.c=c
        self.ui=ui
        
        #self.emb_x=emb_x
        super().__init__()
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):

        return self.x[index],self.emb[index],self.ui[index],self.cons[index],self.y[index],self.c[index]

