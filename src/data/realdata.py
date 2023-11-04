
class RealData:

    def __init__(self) -> None:
        pass

    def load_data(self,args):

        if args.datatype=="ml100k":
            self.load_ml100k(args)
        else:
            raise NotImplementedError