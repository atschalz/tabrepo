from autogluon.features import AutoMLPipelineFeatureGenerator
from tabrepo.benchmark.task.openml.task_wrapper import OpenMLTaskWrapper
from autogluon.common.features.feature_metadata import FeatureMetadata
from pandas import DataFrame, Series
from tabprep.all_in_one import AllInOneEngineer
from autogluon.features.generators import CategoryFeatureGenerator
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

class AllInOneAutoMLPipelineFeatureGenerator(AutoMLPipelineFeatureGenerator):
    def __init__(self,
        target_type: str,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.target_type = target_type
        self.detector = AllInOneEngineer(        
                target_type=target_type,
                # engineering_techniques=['drop_irrelevant', 'cat_freq', 'cat_int']
                # engineering_techniques=['cat_as_num', 'cat_freq', 'cat_int', 'cat_groupby', 'num_int', 'groupby', 'duplicate_mapping'] #, 'linear_residuals'],
                # engineering_techniques=['linear_residuals'],
            )
    # def _fit_transform_custom(self, X_out: DataFrame, type_group_map_special: dict, y=None):
    #     self.interaction_detector.fit(X_out, y)
    #     X_out = self.interaction_detector.transform(X_out)
    #     return super()._fit_transform_custom(X_out=X_out, type_group_map_special=None, y=y)
    
    def fit_transform(self, X: DataFrame, y=None, feature_metadata_in: FeatureMetadata = None, **kwargs) -> DataFrame:
        self.detector.fit(X, y)
        # if self.detector.postprocess_duplicates:
        #     self.postprocess = True

        if self.detector.post_predict_duplicate_mapping:
            self.postprocess_duplicates = True
        else:
            self.postprocess_duplicates = False

        X_out = self.detector.transform(X)
        X_out = super().fit_transform(X_out, y, feature_metadata_in=None, **kwargs)
        
        return X_out

    def transform(self, X: DataFrame) -> DataFrame:
        X_out = self.detector.transform(X)
        X_out = super().transform(X_out)
        return X_out
    
    def transform_pre_fit(self, X: DataFrame, y: Series, **kwargs) -> Series:
        # if self.detector.postprocess_duplicates:
        #     y_pred_adapted = self.detector.post_predict_transform(X, y_pred)
        if  hasattr(self.detector, 'linear_residual_model'):
            lin_pred = self.detector.linear_residual_model.predict(X)
            if self.target_type == 'binary' and y.dtype in ['category', 'object']:
                self.my_positive_class = pd.Series(lin_pred, index=y.index).groupby(y.astype(str)).mean().idxmax()
                self.ag_positive_class = y.value_counts().index[0]
                y_adapted = (y == y.value_counts().index[0]).astype(int)
                y_adapted = y_adapted - lin_pred
            else:
                y_adapted = y - lin_pred

        return y_adapted

    def transform_post_predict(self, X: DataFrame, y_pred: Series, lin_residuals: Series = None, X_str: Series = None, **kwargs) -> Series:
        # if self.detector.postprocess_duplicates:
        #     y_pred_adapted = self.detector.post_predict_transform(X, y_pred)
        y_pred_adapted = y_pred.copy()
        if hasattr(self.detector, 'linear_residual_model'):
            if self.target_type == 'binary':
                # y_pred_adapted = y_pred 
                # y_pred_adapted.iloc[:,1] = (y_pred_adapted.iloc[:,1] + self.detector.linear_residual_model.predict(X)).clip(0.0001,0.9999)
                # y_pred_adapted.iloc[:,0] = 1 - y_pred_adapted.iloc[:,1]

                y_pred_pos = (y_pred + lin_residuals).clip(0.0001, 0.9999)
                y_pred_adapted = pd.concat([y_pred_pos, 1-y_pred_pos], axis=1)
                # if y_pred_pos.mean()>0.5:
                #     y_pred_adapted = pd.concat([1 - y_pred_pos, y_pred_pos], axis=1)
                # else:
                #     y_pred_adapted = pd.concat([y_pred_pos, 1 - y_pred_pos], axis=1)
    
        if self.postprocess_duplicates:
            any_dupes = X_str.apply(lambda x: x in self.detector.dupe_map).sum() > 0
            if any_dupes:
                if self.target_type == 'regression':
                    new_pred = [float(self.detector.dupe_map[i]) if i in self.detector.dupe_map else float(y_pred_adapted.iloc[num])  for num, i in enumerate(X_str)]
                    y_pred_adapted = pd.Series(new_pred, index=y_pred.index, name=y_pred.name)
                elif self.target_type == 'binary':
                    new_pred_1 = np.array([self.detector.dupe_map[i] if i in self.detector.dupe_map else float(y_pred_adapted.iloc[num, 1]) for num, i in enumerate(X_str)])
                    new_pred_0 = np.array([self.detector.dupe_map[i] if i in self.detector.dupe_map else float(y_pred_adapted.iloc[num, 0]) for num, i in enumerate(X_str)])
                    auc_1 = [roc_auc_score(new_pred_1.round(), y_pred_adapted.iloc[:,i].values) for i in [0,1]]
                    auc_0 = [roc_auc_score(new_pred_0.round(), y_pred_adapted.iloc[:,i].values) for i in [0,1]]

                    # FIXME: Get rid of the hack to infer which colum matches the preprocessing vs what AG does
                    if max(auc_1) > max(auc_0):
                        new_pred = new_pred_1
                    else:
                        new_pred = new_pred_0

                    if new_pred[0] > new_pred[1]:
                        use_col = 0
                        other_col = 1
                    else:
                        use_col = 1
                        other_col = 0

                    y_pred_adapted.iloc[:,use_col] = new_pred
                    y_pred_adapted.iloc[:,other_col] = 1 - y_pred_adapted.iloc[:,use_col]
                elif self.target_type == 'multiclass':
                    new_pred = y_pred_adapted.copy()
                    for k in X_str:
                        if k in self.detector.dupe_map:
                            mode = self.detector.dupe_mode[k]
                            val = self.detector.dupe_map[k]

                            if val > 0.99: 
                                val = 0.99

                            new_pred.loc[X_str == k] = (1-val)/(y_pred_adapted.shape[1]-1)
                            new_pred.loc[X_str == k, mode] = val

                            new_pred.loc[X_str == k] = new_pred.loc[X_str == k] /new_pred.loc[X_str == k].sum().sum()
                            continue  
                    # y_pred_adapted = new_pred
                    y_pred_adapted[y_pred_adapted.idxmax(axis=1)==new_pred.idxmax(axis=1)] = new_pred[y_pred_adapted.idxmax(axis=1)==new_pred.idxmax(axis=1)]
        return y_pred_adapted


    # def _get_category_feature_generator(self):
    #     return CategoryFeatureGenerator()
