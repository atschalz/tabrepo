from __future__ import annotations

import numpy as np
import pandas as pd
from autogluon.core.data.label_cleaner import LabelCleaner, LabelCleanerDummy
from autogluon.core.metrics import get_metric, Scorer
from autogluon.features import AutoMLPipelineFeatureGenerator
from autogluon.features.generators import PipelineFeatureGenerator
from tabrepo.utils.time_utils import Timer
from autogluon.core.metrics import root_mean_squared_error

class AbstractExecModel:
    can_get_oof = False
    can_get_per_child_oof = False
    can_get_per_child_test = False

    # TODO: Prateek: Find a way to put AutoGluon as default - in the case the user does not want their own class
    def __init__(
        self,
        problem_type: str,
        eval_metric: Scorer,
        preprocess_data: bool = True,
        preprocessor_name: str = 'default', # ['default', 'default_csv', 'ftd', 'ftd_csv']
        preprocess_label: bool = True,
    ):
        self.problem_type = problem_type
        self.eval_metric = eval_metric
        self.preprocess_data = preprocess_data
        self.preprocessor_name = preprocessor_name
        self.preprocess_label = preprocess_label
        self.label_cleaner: LabelCleaner = None
        self._feature_generator = None
        self.failure_artifact = None

        self.postprocess = False

    def transform_y(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        y_out = y.copy()
        y_out = self.label_cleaner.transform(y_out)
        if hasattr(self._feature_generator, 'detector') and hasattr(self._feature_generator.detector, 'linear_residual_model'):
            y_out = self._feature_generator.transform_pre_fit(X, y_out)

        return y_out

    def inverse_transform_y(self, X: pd.DataFrame, y: pd.Series, lin_residuals: pd.Series = None, X_str: pd.Series = None) -> pd.Series:
        y_out = y.copy()
        # y_out = self.label_cleaner.transform(y_out)
        if (hasattr(self._feature_generator, 'detector') and hasattr(self._feature_generator.detector, 'linear_residual_model')):
            y_out = self._feature_generator.transform_post_predict(X, y_out, lin_residuals=lin_residuals, X_str=X_str)
        if hasattr(self._feature_generator, 'postprocess_duplicates') and self._feature_generator.postprocess_duplicates:
            y_out = self._feature_generator.transform_post_predict(X, y_out, lin_residuals=lin_residuals, X_str=X_str)
        
        return y_out
        # return self.label_cleaner.inverse_transform(y)

    # def transform_y_pred_proba(self, X: pd.DataFrame, y_pred_proba: pd.DataFrame) -> pd.DataFrame:
    #     y_pred_proba_out = y_pred_proba.copy()
    #     # y_pred_proba_out = self.label_cleaner.transform(y_pred_proba_out, as_pandas=True)
    #     if hasattr(self._feature_generator, 'linear_residuals'):
    #         if self._feature_generator.linear_residuals:
    #             y_pred_proba_out = self._feature_generator.transform_post_predict(X, y_pred_proba_out)

    #     return y_pred_proba_out

        # return self.label_cleaner.transform_proba(y_pred_proba, as_pandas=True)

    # def inverse_transform_y_pred_proba(self, X: pd.DataFrame, y_pred_proba: pd.DataFrame) -> pd.DataFrame:
    #     y_out = y.copy()
    #     y_out = self.label_cleaner.transform(y_out)
    #     if hasattr(self._feature_generator, 'linear_residuals'):
    #         if self._feature_generator.linear_residuals:
    #             y_out = self._feature_generator.transform_pre_fit(X, y_out)
    #     return y_out
        # return self.label_cleaner.inverse_transform_proba(y_pred_proba, as_pandas=True)

    def transform_X(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.preprocess_data:
            return self._feature_generator.transform(X)
        return X

    def get_preprocessor(self) -> PipelineFeatureGenerator:
        if self.preprocessor_name == 'default':
            return AutoMLPipelineFeatureGenerator()
        elif self.preprocessor_name == 'default_csv':
            from tabrepo.preprocessors.default_csv import FromCSVAutoMLPipelineFeatureGenerator
            return FromCSVAutoMLPipelineFeatureGenerator()
        elif self.preprocessor_name == 'ftd':
            from tabrepo.preprocessors.ftd import FTDAutoMLPipelineFeatureGenerator
            return FTDAutoMLPipelineFeatureGenerator(target_type=self.problem_type)
        elif self.preprocessor_name == 'ftd_csv':
            from tabrepo.preprocessors.ftd import FromCSVFTDAutoMLPipelineFeatureGenerator
            return FromCSVFTDAutoMLPipelineFeatureGenerator(target_type=self.problem_type)
        elif self.preprocessor_name == 'catint':
            from tabrepo.preprocessors.cat_interaction import CatIntAutoMLPipelineFeatureGenerator
            return CatIntAutoMLPipelineFeatureGenerator(target_type=self.problem_type)
        elif self.preprocessor_name == 'catint_csv':
            from tabrepo.preprocessors.cat_interaction import FromCSVCatIntAutoMLPipelineFeatureGenerator
            return FromCSVCatIntAutoMLPipelineFeatureGenerator(target_type=self.problem_type)
        elif self.preprocessor_name == 'catres':
            from tabrepo.preprocessors.cat_resolution import CatResAutoMLPipelineFeatureGenerator
            return CatResAutoMLPipelineFeatureGenerator(target_type=self.problem_type)
        elif self.preprocessor_name == 'catfreq':
            from tabrepo.preprocessors.cat_freq import CatFreqAutoMLPipelineFeatureGenerator
            return CatFreqAutoMLPipelineFeatureGenerator(target_type=self.problem_type)
        elif self.preprocessor_name == 'te':
            from tabrepo.preprocessors.target_enc import CatTEAutoMLPipelineFeatureGenerator
            return CatTEAutoMLPipelineFeatureGenerator()
        elif self.preprocessor_name == 'catgroupby':
            from tabrepo.preprocessors.cat_groupby import CatGroupByAutoMLPipelineFeatureGenerator
            return CatGroupByAutoMLPipelineFeatureGenerator(target_type=self.problem_type)
        elif self.preprocessor_name == 'groupby':
            from tabrepo.preprocessors.groupby_FE import GroupByAutoMLPipelineFeatureGenerator
            return GroupByAutoMLPipelineFeatureGenerator(target_type=self.problem_type)
        elif self.preprocessor_name == 'numint':
            from tabrepo.preprocessors.num_interaction import NumIntAutoMLPipelineFeatureGenerator
            return NumIntAutoMLPipelineFeatureGenerator(target_type=self.problem_type)
        elif self.preprocessor_name == 'cattreat':
            from tabrepo.preprocessors.cat_asnum_detection import CatTreatAutoMLPipelineFeatureGenerator
            return CatTreatAutoMLPipelineFeatureGenerator(target_type=self.problem_type)
        elif self.preprocessor_name == 'linear':
            from tabrepo.preprocessors.num_linear import NumLinearAutoMLPipelineFeatureGenerator
            return NumLinearAutoMLPipelineFeatureGenerator(target_type=self.problem_type)
        elif self.preprocessor_name == 'featselect':
            from tabrepo.preprocessors.feat_select import FeatSelectAutoMLPipelineFeatureGenerator
            return FeatSelectAutoMLPipelineFeatureGenerator(target_type=self.problem_type)
        elif self.preprocessor_name == 'catengineer':
            from tabrepo.preprocessors.cat_engineering import CatEngineeringAutoMLPipelineFeatureGenerator
            return CatEngineeringAutoMLPipelineFeatureGenerator(target_type=self.problem_type)
        elif self.preprocessor_name == 'all_in_one':
            from tabrepo.preprocessors.all_in_one import AllInOneAutoMLPipelineFeatureGenerator
            return AllInOneAutoMLPipelineFeatureGenerator(target_type=self.problem_type)
        elif self.preprocessor_name == 'catres_csv':
            from tabrepo.preprocessors.cat_resolution import FromCSVCatResAutoMLPipelineFeatureGenerator
            return FromCSVCatResAutoMLPipelineFeatureGenerator(target_type=self.problem_type)
        elif self.preprocessor_name == 'catirrelevant':
            from tabrepo.preprocessors.cat_irrelevant import CatIrrelevantAutoMLPipelineFeatureGenerator
            return CatIrrelevantAutoMLPipelineFeatureGenerator(target_type=self.problem_type)
        elif self.preprocessor_name == 'catirrelevant_csv':
            from tabrepo.preprocessors.cat_irrelevant import FromCSVCatIrrelevantAutoMLPipelineFeatureGenerator
            return FromCSVCatIrrelevantAutoMLPipelineFeatureGenerator(target_type=self.problem_type)
        else:
            # TODO: Better use the default preprocessor as fallback solution?
            raise NotImplementedError(f"Preprocessor {self.preprocessor_name} is not implemented.")

    def _preprocess_fit_transform(self, X: pd.DataFrame, y: pd.Series):
        y_out = y.copy()
        if self.preprocess_label:
            self.label_cleaner = LabelCleaner.construct(problem_type=self.problem_type, y=y_out)
        else:
            self.label_cleaner = LabelCleanerDummy(problem_type=self.problem_type)
        if self.preprocess_data:
            self._feature_generator = self.get_preprocessor()
            X_out = self._feature_generator.fit_transform(X, y_out)
            if hasattr(self._feature_generator, 'detector') and hasattr(self._feature_generator.detector, 'linear_residual_model'):
                y_out = self._feature_generator.transform_pre_fit(X, y_out)
                self.orig_problem_type = self.problem_type
                self.problem_type = 'regression'
                self.eval_metric_orig = self.eval_metric
                self.eval_metric = root_mean_squared_error
        
        return X_out, y_out

    # def pre_fit(self, X: pd.DataFrame, y: pd.Series):
    #     y_out = y.copy()
    #     if self.label_cleaner is None:
    #         self.label_cleaner = LabelCleaner.construct(problem_type=self.problem_type, y=y_out)
    #     y_out = self.label_cleaner.transform(y_out)
    #     if hasattr(self._feature_generator, 'linear_residuals'):
    #         if self._feature_generator.linear_residuals:
    #             y_out = self._feature_generator.transform_pre_fit(X, y_out)

    #     return y_out

    def post_fit(self, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame):
        pass

    def back_to_orig_target(self):
        if hasattr(self._feature_generator, 'detector') and hasattr(self._feature_generator.detector, 'linear_residual_model'):

            self.problem_type = self.orig_problem_type
            self.eval_metric = self.eval_metric_orig
        else:
            pass
        
    # def post_predict(self, X_test: pd.DataFrame, y_pred: pd.Series, **kwargs) -> pd.Series:
    #     return self._feature_generator.transform_post_predict(X_test, y_pred, **kwargs)

    # TODO: Prateek, Add a toggle here to see if user wants to fit or fit and predict, also add model saving functionality
    # TODO: Nick: Temporary name
    def fit_custom(self, X: pd.DataFrame, y: pd.Series, X_test: pd.DataFrame):
        '''
        Calls the fit function of the inheriting class and proceeds to perform predictions based on the problem type

        Returns
        -------
        dict
        Returns predictions, probabilities, fit time and inference time
        '''

        # y = self.pre_fit(X, y)

        with (Timer() as timer_fit):
            self.fit(X, y)

        self.post_fit(X=X, y=y, X_test=X_test)

        if self.problem_type in ['binary', 'multiclass']:
            with Timer() as timer_predict:
                y_pred_proba = self.predict_proba(X_test)
            y_pred = self.predict_from_proba(y_pred_proba)
        else:
            with Timer() as timer_predict:
                y_pred = self.predict(X_test)
            y_pred_proba = None
        
        if hasattr(self._feature_generator, 'detector') and hasattr(self._feature_generator.detector, 'linear_residual_model'):
            self.back_to_orig_target()
            if self.problem_type == 'binary':
                y_pred_proba = y_pred

        out = {
            'predictions': y_pred,
            'probabilities': y_pred_proba,
            'time_train_s': timer_fit.duration,
            'time_infer_s': timer_predict.duration,
        }

        if hasattr(self._feature_generator, 'detector') and hasattr(self._feature_generator.detector, 'linear_residual_model'):
            out['lin_res'] = self._feature_generator.detector.linear_residual_model.predict(X)
            out['lin_res'] = pd.Series(out['lin_res'], index=X.index)

            out['lin_res_test'] = self._feature_generator.detector.linear_residual_model.predict(X_test)
            out['lin_res_test'] = pd.Series(out['lin_res_test'], index=X_test.index)

            # if self.problem_type == 'binary':
            #     self.predictor.problem_type = 'binary'

        return out

    def fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None):
        X, y = self._preprocess_fit_transform(X=X, y=y)
        if X_val is not None:
            X_val = self.transform_X(X_val)
            y_val = self.transform_y(y_val)
        return self._fit(X=X, y=y, X_val=X_val, y_val=y_val)

    def _fit(self, X: pd.DataFrame, y: pd.Series, X_val=None, y_val=None):
        raise NotImplementedError

    def predict_from_proba(self, y_pred_proba: pd.DataFrame) -> pd.Series:
        if isinstance(y_pred_proba, pd.DataFrame):
            return y_pred_proba.idxmax(axis=1)
        else:
            return np.argmax(y_pred_proba, axis=1)

    def predict(self, X: pd.DataFrame) -> pd.Series:
        if hasattr(self._feature_generator, 'detector') and hasattr(self._feature_generator.detector, 'linear_residual_model'):
            lin_res = self._feature_generator.detector.linear_residual_model.predict(X)
        else: 
            lin_res = None

        if hasattr(self._feature_generator, 'postprocess_duplicates') and self._feature_generator.postprocess_duplicates:
            X_str = X.astype(str).sum(axis=1)
        else:
            X_str = None

        X = self.transform_X(X=X)
        y_pred = self._predict(X)
        # if hasattr(self._feature_generator, 'linear_residuals'):
        #     if self._feature_generator.linear_residuals:
        #         y_pred = self._feature_generator.transform_post_predict(X, y_pred)
        return self.inverse_transform_y(X, y=y_pred, lin_residuals=lin_res, X_str=X_str)

    def _predict(self, X: pd.DataFrame):
        raise NotImplementedError

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        if hasattr(self._feature_generator, 'detector') and hasattr(self._feature_generator.detector, 'linear_residual_model'):
            lin_res = self._feature_generator.detector.linear_residual_model.predict(X)
        else: 
            lin_res = None

        if hasattr(self._feature_generator, 'postprocess_duplicates') and self._feature_generator.postprocess_duplicates:
            X_str = X.astype(str).sum(axis=1)
        else:
            X_str = None

        X = self.transform_X(X=X)
        y_pred_proba = self._predict_proba(X=X)
        # if hasattr(self._feature_generator, 'linear_residuals'):
        #     if self._feature_generator.linear_residuals:
        #         y_pred_proba = self._feature_generator.transform_post_predict(X, y_pred_proba)
        return self.inverse_transform_y(X, y=y_pred_proba, lin_residuals=lin_res, X_str=X_str)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    def cleanup(self):
        pass
