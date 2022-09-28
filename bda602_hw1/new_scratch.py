import pandas


class DataLoader:
    def __init__(self, filename):
        print("Starting Data Loader")
        self.data = pandas.read_csv(filename)

    def __del__(self):
        print("ending dataloader")
