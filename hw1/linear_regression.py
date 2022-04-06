import numpy as np
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.model_selection import KFold, cross_val_score
from sklearn.preprocessing import PolynomialFeatures
from pandas import DataFrame
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted, check_X_y
import torch


class LinearRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Linear Regression prediction and closed-form parameter fitting.
    """

    def __init__(self, reg_lambda=0.1):
        self.reg_lambda = reg_lambda

    def predict(self, X):
        """
        Predict the class of a batch of samples based on the current weights.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :return:
            y_pred: np.ndarray of shape (N,) where each entry is the predicted
                value of the corresponding sample.
        """
        X = check_array(X)
        self.get_params()
        check_is_fitted(self, 'weights_')
        # TODO: Calculate the model prediction, y_pred

        y_pred = None
        # ====== YOUR CODE: ======
        y_pred = np.dot(X, self.weights_)  # bias is in the weights
        # ========================

        return y_pred

    def fit(self, X, y):
        """
        Fit optimal weights to data using closed form solution.
        :param X: A tensor of shape (N,n_features_) where N is the batch size.
        :param y: A tensor of shape (N,) where N is the batch size.
        """
        X, y = check_X_y(X, y)

        # TODO: Calculate the optimal weights using the closed-form solution
        # Use only numpy functions.

        w_opt = None
        # ====== YOUR CODE: ======
        # b_trick = BiasTrickTransfox`rmer()
        # X = b_trick.transform(X)
        Xt_X = np.dot(X.transpose(), X)
        Xt_X_reg = np.add(Xt_X, self.reg_lambda * np.identity(X.shape[1]))
        Xt_y = np.dot(X.transpose(), y)
        w_opt = np.dot(np.linalg.inv(Xt_X_reg), Xt_y)
        # ========================

        self.weights_ = w_opt
        return self

    def fit_predict(self, X, y):
        return self.fit(X, y).predict(X)


class BiasTrickTransformer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        :param X: A tensor of shape (N,D) where N is the batch size or of shape
            (D,) (which assumes N=1).
        :returns: A tensor xb of shape (N,D+1) where xb[:, 0] == 1
        """

        # TODO: Add bias term to X as the first feature.
        # ====== YOUR CODE: ======
        X = check_array(X)
        N = 1
        cat_index = 0
        # if(len(X.shape[2])  1):
        #     N = X.shape[1]
        #     cat_index = 1
        bias = np.ones((X.shape[0], 1))
        xb = np.concatenate((bias, X), axis=1)
        # ========================

        return xb


class BostonFeaturesTransformer(BaseEstimator, TransformerMixin):
    """
    Generates custom features for the Boston dataset.
    """

    def __init__(self, degree=2):
        self.degree = degree

        # TODO: Your custom initialization, if needed
        # Add any hyperparameters you need and save them as above
        # ====== YOUR CODE: ======
        # self.feats = PolynomialFeatures(degree)
        # ========================

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        """
        Transform features to new features matrix.
        :param X: Matrix of shape (n_samples, n_features_).
        :returns: Matrix of shape (n_samples, n_output_features_).
        """
        X = check_array(X)
        # check_is_fitted(self, ['n_features_', 'n_output_features_'])

        # TODO: Transform the features of X into new features in X_transformed
        # Note: You can count on the order of features in the Boston dataset
        # (this class is "Boston-specific"). For example X[:,1] is the second
        # feature ('ZN').

        X_transformed = None
        # ====== YOUR CODE: ======
        self.feats = PolynomialFeatures(self.degree)
        X_transformed = self.feats.fit_transform(X)
        # ========================

        return X_transformed


def compute_cor(df: DataFrame, col):
    return (df[col] - df[col].mean()) ** 2


def top_correlated_features(df: DataFrame, target_feature, n=5):
    """
    Returns the names of features most strongly correlated (correlation is
    close to 1 or -1) with a target feature. Correlation is Pearson's-r sense.

    :param df: A pandas dataframe.
    :param target_feature: The name of the target feature.
    :param n: Number of top features to return.
    :return: A tuple of
        - top_n_features: Sequence of the top feature names
        - top_n_corr: Sequence of correlation coefficients of above features
        Both the returned sequences should be sorted so that the best (most
        correlated) feature is first.
    """

    # TODO: Calculate correlations with target and sort features by it
    # ====== YOUR CODE: ======
    columns_after_drop = df.columns.drop(target_feature)
    correlation = np.zeros(0)
    for col in columns_after_drop:
        # if(col == target_feature):
        #     continue
        cor_xy = ((df[col] - df[col].mean()) *
                  (df[target_feature] - df[target_feature].mean())).sum()
        cor_xx_cor_yy = ((compute_cor(df, col).sum()) ** 0.5) * \
            ((compute_cor(df, target_feature).sum()) ** 0.5)
        # print(up)
        correlation = np.append(correlation, np.abs(cor_xy/cor_xx_cor_yy))
    max_indices = np.argpartition(correlation, n * -1)[n * (-1):][::-1]
    top_n_features = columns_after_drop[max_indices]
    top_n_corr = correlation[max_indices]
    # ========================

    return top_n_features, top_n_corr


def evaluate_accuracy(y: np.ndarray, y_pred: np.ndarray):
    """
    Calculates mean squared error (MSE) and coefficient of determination (R-squared).
    :param y: Target values.
    :param y_pred: Predicted values.
    :return: A tuple containing the MSE and R-squared values.
    """
    mse = np.mean((y - y_pred) ** 2)
    rsq = 1 - mse / np.var(y)
    return mse.item(), rsq.item()


def cv_best_hyperparams(model: BaseEstimator, X, y, k_folds,
                        degree_range, lambda_range):
    """
    Cross-validate to find best hyperparameters with k-fold CV.
    :param X: Training data.
    :param y: Training targets.
    :param model: sklearn model.
    :param lambda_range: Range of values for the regularization hyperparam.
    :param degree_range: Range of values for the degree hyperparam.
    :param k_folds: Number of folds for splitting the training data into.
    :return: A dict containing the best model parameters,
        with some of the keys as returned by model.get_params()
    """

    # TODO: Do K-fold cross validation to find the best hyperparameters
    #
    # Notes:
    # - You can implement it yourself or use the built in sklearn utilities
    #   (recommended). See the docs for the sklearn.model_selection package
    #   http://scikit-learn.org/stable/modules/classes.html#module-sklearn.model_selection
    # - If your model has more hyperparameters (not just lambda and degree)
    #   you should add them to the search.
    # - Use get_params() on your model to see what hyperparameters is has
    #   and their names. The parameters dict you return should use the same
    #   names as keys.
    # - You can use MSE or R^2 as a score.

    # ====== YOUR CODE: ======

    # cv_scores = []
    # for i, degree in enumerate(degree_range):
    #     model.set_params(bostonfeaturestransformer__degree=degree)
    #     for i, lambda_reg in enumerate(lambda_range):
    #         model.set_params(linearregressor__reg_lambda=lambda_reg)
    #         scores = cross_val_score(model, X, y, cv=k_folds, scoring='neg_mean_squared_error')
    #         cv_scores.append(scores.mean())
    # best_index=cv_scores.index(max(cv_scores))
    # print(best_index)
    # model.set_params(linearregressor__reg_lambda=lambda_range[best_index%len(lambda_range)])
    # model.set_params(bostonfeaturestransformer__degree=degree_range[int(best_index/len(lambda_range))])
    # best_params=model.get_params()
    
    # '''
    # grid_values={'bostonfeaturestransformer__degree':degree_range,'linearregressor__reg_lambda':lambda_range}
    # clf= GridSearchCV(estimator=model,param_grid=grid_values,scoring='r2',cv=k_folds)
    # print(clf.fit(X,y))
    # best_params=clf.best_params_

    # '''
    # # ========================

    # return best_params




    cv_scores = None
    best_params = None
    k_fold = KFold(n_splits=k_folds)
    cv_array = None
    param_list = [(degree, lamb)
                  for degree in degree_range for lamb in lambda_range]
    for item in param_list:
        degree, lamb = item
        total_mse = 0
        # param_dict = {
        #     "bostonfeaturestransformer__degree": degree,
        #     "linearregressor__reg_lambda": lamb
        # }
        # model.set_params(**param_dict)
        model.set_params(bostonfeaturestransformer__degree=degree)
        model.set_params(linearregressor__reg_lambda=lamb)
        # if(cv_array is None):
        #       cv_array = np.array([[scores.mean(), degree, lamb]])
        # else:
        #     cv_array = np.append(cv_array, [[scores.mean(), degree, lamb]], axis=0)
        # min_index = np.argmax(np.array(cv_array), axis=0)[0]
        for train_index, test_index in k_fold.split(X, y):
            train_x = np.array([X[i] for i in train_index])
            test_x = np.array([X[i] for i in test_index])
            train_label = np.array([y[i] for i in train_index])
            test_label = np.array([y[i] for i in test_index])
            model.fit(train_x, train_label)  # each fit restarts learning
            y_pred_test = model.predict(test_x)
            mse = np.mean((test_label - y_pred_test) ** 2)
            total_mse = total_mse + mse/k_folds
        # score_mean = cross_val_score(model, X, y, cv=k_folds, scoring='neg_mean_squared_error').mean()
        if(cv_array is None):
            cv_array = np.array([[total_mse, degree, lamb]])
        else:
            cv_array = np.append(cv_array, [[total_mse, degree, lamb]], axis=0)
            
    min_index = np.argmin(np.array(cv_array), axis=0)[0]
    # return {'bostonfeaturestransformer__degree': 3.0, 'linearregressor__reg_lambda': 29.763514416313193}
    best_params = {
        "bostonfeaturestransformer__degree": int(cv_array[min_index][1]),
        "linearregressor__reg_lambda": cv_array[min_index][2]
    }
    
    # ========================

    return best_params
