from autogluon.features import AutoMLPipelineFeatureGenerator
from tabrepo.benchmark.task.openml.task_wrapper import OpenMLTaskWrapper
from autogluon.common.features.feature_metadata import FeatureMetadata
from pandas import DataFrame
import sys
sys.path.append('/home/ubuntu/cat_detection')
from tabprep.detectors.ft_detection import FeatureTypeDetector
from autogluon.features.generators import CategoryFeatureGenerator

class FTDAutoMLPipelineFeatureGenerator(AutoMLPipelineFeatureGenerator):
    def __init__(self,
        target_type: str,
        **kwargs,
    ):
        super().__init__(
            **kwargs,
        )
        self.feature_type_detector = FeatureTypeDetector(
            target_type=target_type,
            tests_to_run = ['dummy_mean','leave_one_out','combination', 'interpolation', 'multivariate_performance'],
            min_q_as_num = 6,
            n_folds = 5,
            alpha = 0.1,
            significance_method = 'wilcoxon',
            max_degree = 5,
            interpolation_criterion = "match",
            combination_criterion = 'win',
            combination_test_min_bins = 2,
            combination_test_max_bins = 2048,
            mvp_criterion = 'significance',
            mvp_use_data = 'all',  # 'all' or 'numeric'
        )
    
    def fit_transform(self, X: DataFrame, y=None, feature_metadata_in: FeatureMetadata = None, **kwargs) -> DataFrame:
        self.feature_type_detector.fit(X, y)
        X = self.feature_type_detector.transform(X)
        return super().fit_transform(X, y, feature_metadata_in, **kwargs)
    
    def transform(self, X: DataFrame) -> DataFrame:
        X = self.feature_type_detector.transform(X)
        return super().transform(X)

class FromCSVFTDAutoMLPipelineFeatureGenerator(AutoMLPipelineFeatureGenerator):
    def __init__(self,
        target_type: str,
        **kwargs,
    ):
        super().__init__(**kwargs)

    
        self.feature_type_detector = FeatureTypeDetector(
            target_type=target_type,
            tests_to_run = ['dummy_mean','leave_one_out','combination', 'interpolation', 'multivariate_performance'],
            min_q_as_num = 6,
            n_folds = 5,
            alpha = 0.1,
            significance_method = 'wilcoxon',
            max_degree = 5,
            interpolation_criterion = "match",
            combination_criterion = 'win',
            combination_test_min_bins = 2,
            combination_test_max_bins = 2048,
            mvp_criterion = 'significance',
            mvp_use_data = 'all',  # 'all' or 'numeric'
        )
    
    def fit_transform(self, X: DataFrame, y=None, feature_metadata_in: FeatureMetadata = None, **kwargs) -> DataFrame:
        X = OpenMLTaskWrapper.to_csv_format(X)
        self.feature_type_detector.fit(X, y)
        X = self.feature_type_detector.transform(X)
        return super().fit_transform(X, y, feature_metadata_in, **kwargs)
    
    def transform(self, X: DataFrame) -> DataFrame:
        X = OpenMLTaskWrapper.to_csv_format(X)
        X = self.feature_type_detector.transform(X)
        return super().transform(X)

    # def _get_category_feature_generator(self):
    #     return CategoryFeatureGenerator()


# class FromCSVAutoMLPipelineFeatureGenerator(AutoMLPipelineFeatureGenerator):
#     def __init__(self):
#         super().__init__()
    
#     def fit_transform(self, X: DataFrame, y=None, feature_metadata_in: FeatureMetadata = None, **kwargs) -> DataFrame:
#         X = OpenMLTaskWrapper.to_csv_format(X)
#         return super().fit_transform(X, y, feature_metadata_in, **kwargs)
    
#     def transform(self, X: DataFrame) -> DataFrame:
#         X = OpenMLTaskWrapper.to_csv_format(X)
#         return super().transform(X)

     