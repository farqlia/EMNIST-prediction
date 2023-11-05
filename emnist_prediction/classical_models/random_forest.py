import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

from emnist_prediction.eval.metrics import min_f1_score


class CustomizedRandomForest:

    def __init__(self, dimensionality=None, resampler=None, class_weight=None, max_depth=None, max_features='sqrt'):
        self.resampler = resampler
        self.class_weight = class_weight
        self.max_depth = max_depth
        self.max_features = max_features
        self.dimensionality = dimensionality

        self.x_std = 0
        self.pca = None
        if dimensionality is not None:
            self.pca = PCA(n_components=dimensionality)

        self.random_forest_clf = RandomForestClassifier(class_weight=self.class_weight,
                                                        max_depth=self.max_depth,
                                                        max_features=self.max_features)

        self.X = None
        self.y_labels = None

    def _prepare_train_data(self, X, y_labels):
        X = X.reshape((len(X), -1))
        if self.resampler:
            X, y_labels = self.resampler.fit_resample(X, y_labels)

        if self.pca:
            self.x_std = np.std(X, axis=0)
            self.x_std[self.x_std == 0] = 1
            X = X / self.x_std
            X = self.pca.fit_transform(X)

        return X, y_labels

    def _prepare_val_data(self, X):
        X = X.reshape((len(X), -1))
        if self.pca:
            X = X / self.x_std
            X = self.pca.transform(X)
        return X

    def fit(self, X, y_labels):
        self.X, self.y_labels = self._prepare_train_data(X, y_labels)
        self.random_forest_clf.fit(self.X, self.y_labels)

    def predict(self, X):
        X = self._prepare_val_data(X)
        return self.random_forest_clf.predict(X)

    # Deep defines whether parameters of sub-estimators should be returned
    def get_params(self, deep=True):
        forest_params = self.random_forest_clf.get_params()
        return {'dimensionality': self.dimensionality, 'resampler': self.resampler,
                'class_weight': forest_params['class_weight'], 'max_depth': forest_params['max_depth'],
                'max_features': forest_params['max_features']}

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self

    def score(self, X, y_labels):
        X = self._prepare_val_data(X)
        predictions = self.random_forest_clf.predict(X)

        f1_score_by_class = min_f1_score(y_labels, predictions)

        return {'avg_weighted_f1': f1_score(y_labels, predictions, average='weighted'),
                'min_f1': f1_score_by_class.min(),
                'total_f1': f1_score(y_labels, predictions, average='micro'),
                'avg_f1': f1_score(y_labels, predictions, average='macro')}
