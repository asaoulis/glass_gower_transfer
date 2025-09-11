class DataScaler:
    def __init__(self):
        self.min = None
        self.max = None
        self.mean = None
        self.std = None

    def fit_minmax(self, X):
        self.min = X.min(axis=0)
        self.max = X.max(axis=0)
    
    def transform_minmax(self, X):
        return (X - self.min) / (self.max - self.min)
    
    def inverse_transform_minmax(self, X):
        return X * (self.max - self.min) + self.min

    def fit_standard(self, X):
        self.mean = X.mean()
        self.std = X.std()
    
    def transform_standard(self, X):
        return (X - self.mean) / self.std
    
    def inverse_transform_standard(self, X):
        return X * self.std + self.mean