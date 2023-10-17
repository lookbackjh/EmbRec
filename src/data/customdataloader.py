import torch.utils.data as data_utils

class CustomDataLoader(data_utils.Dataset):
    # as we already converted to tensor, we can directly return the tensor
    def __init__(self,x,y,c) -> None:
        self.x=x
        self.y=y
        self.c=c
        #self.emb_x=emb_x
        super().__init__()
    def __len__(self):
        return len(self.x)
    
    def __getitem__(self, index):

        return self.x[index],self.y[index],self.c[index]

