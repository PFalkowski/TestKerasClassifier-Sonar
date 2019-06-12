import numpy
import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class ClassificationApi:
    
    def __init__(self):
        return

    def inputLen(self):
        return self.dataframe.shape[1] - 1

    def ReadData(self, path):
        self.dataframe = pandas.read_csv(path, header=None)
        self.dataset = self.dataframe.values
        self.X = self.dataset[:,0:self.inputLen()].astype(float)
        self.Y = self.dataset[:,self.inputLen()]

    def EncodeY(self):
        encoder = LabelEncoder()
        encoder.fit(self.Y)
        self.encoded_Y = encoder.transform(self.Y)

    def CreateModel(self):
        if (self.dataframe is None):
            raise ValueError('dataframe is not initialized. Use ReadData() before calling CreateModel()')
        model = Sequential()
        model.add(Dense(self.inputLen(), input_dim = self.inputLen(), kernel_initializer='normal', activation='relu'))
        model.add(Dense(30, kernel_initializer='normal', activation='relu'))
        model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))	    
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    def TrainModel(self, iterations = 100, batchSize = 5):
        if (self.dataframe is None):
            raise ValueError('dataframe is not initialized. Use ReadData() before calling TrainModel()')
        estimators = []
        estimators.append(('standardize', StandardScaler()))
        estimators.append(('mlp', KerasClassifier(build_fn=self.CreateModel, epochs=iterations, batch_size=batchSize, verbose=0)))
        pipeline = Pipeline(estimators)
        kfold = StratifiedKFold(n_splits=10, shuffle=True)
        self.results = cross_val_score(pipeline, self.X, self.encoded_Y, cv=kfold)
        return self.results

    def GetSummary(self):
        if (self.results is None):
            raise ValueError('results are None. Use TrainModel() before calling GetSummary()')
        return f"Standardized: {self.results.mean()*100} ({self.results.std()*100})"

if __name__ == '__main__':
    # load dataset
    api = ClassificationApi()
    api.ReadData("..\sonar.csv")
    api.EncodeY()
    api.CreateModel()
    api.TrainModel()
    print(api.GetSummary())
    # evaluate baseline model with standardized dataset