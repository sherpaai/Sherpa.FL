from shfl import differential_privacy
from shfl import private
from shfl import model
from shfl import data_base
from shfl import federated_government
from shfl import data_distribution
from shfl import federated_aggregator

# For each class to document, it is possible to:
# 1) Document only the class: [classA, classB, ...]
# 2) Document all its methods: [classA, (classB, "*")]
# 3) Choose which methods to document (methods listed as strings):
# [classA, (classB, ["method1", "method2", ...]), ...]
# 4) Choose which methods to document (methods listed as qualified names):
# [classA, (classB, [module.classB.method1, module.classB.method2, ...]), ...]

PAGES = [
    {
        'page': 'private/data_node.md',
        'classes': [
            private.node.DataNode
        ],
        'methods': [
            private.node.DataNode.set_private_data,
            private.node.DataNode.set_private_test_data,
            private.node.DataNode.configure_data_access,
            private.node.DataNode.configure_model_params_access,
            private.node.DataNode.apply_data_transformation,
            private.node.DataNode.query,
            private.node.DataNode.query_model_params,
            private.node.DataNode.set_model_params,
            private.node.DataNode.train_model,
            private.node.DataNode.predict,
            private.node.DataNode.evaluate,
            private.node.DataNode.performance,
            private.node.DataNode.local_evaluate,
        ],
    },
    {
        'page': 'private/data.md',
        'classes': [
            private.data.LabeledData,
            (private.data.DataAccessDefinition, ['apply']),
            (private.data.DPDataAccessDefinition, ['_check_epsilon_delta',
                                                   '_check_binary_data',
                                                   '_check_sensitivity_positive',
                                                   '_check_sensitivity_shape']),
            (private.data.UnprotectedAccess, ['apply'])
        ]
    },
    {
        'page': 'private/query.md',
        'classes': [
            (private.query.Query, ["get"]),
            (private.query.IdentityFunction, ['get']),
            (private.query.Mean, ['get'])
        ]
    },
    {
        'page': 'private/federated_operation.md',
        'classes': [
            (private.federated_operation.FederatedDataNode, ['query',
                                                             'configure_data_access',
                                                             'set_private_data',
                                                             'set_private_test_data',
                                                             'train_model',
                                                             'apply_data_transformation',
                                                             'evaluate',
                                                             'split_train_test']),
            (private.federated_operation.FederatedData, ["add_data_node", "num_nodes",
                                                         "configure_data_access", "query"]),
            (private.federated_operation.FederatedTransformation, ["apply"]),
            (private.federated_operation.Normalize, ['apply'])
        ],
        'functions': [
            private.federated_operation.federate_array,
            private.federated_operation.apply_federated_transformation,
            private.federated_operation.split_train_test
        ]
    },
    {
        'page': 'private/federated_attack.md',
        'classes': [
            (private.federated_attack.FederatedDataAttack, ['apply_attack']),
            (private.federated_attack.ShuffleNode, ['apply']),
            (private.federated_attack.FederatedPoisoningDataAttack, ['apply_attack'])
        ]
    },
    {
        'page': 'private/reproducibility.md',
        'classes': [
            private.reproducibility.Reproducibility
        ],
        'methods': [
            private.reproducibility.Reproducibility.get_instance,
            private.reproducibility.Reproducibility.set_seed,
            private.reproducibility.Reproducibility.delete_instance
        ]
    },
    {
        'page': 'databases.md',
        'classes': [
            (data_base.data_base.DataBase, ['load_data',
                                            "shuffle"]),
            (data_base.data_base.LabeledDatabase, ['load_data']),
            (data_base.emnist.Emnist, ['load_data']),
            (data_base.fashion_mnist.FashionMnist, ['load_data']),
            (data_base.california_housing.CaliforniaHousing, ['load_data']),
            (data_base.iris.Iris, ['load_data'])
        ],
        'functions': [
            data_base.data_base.split_train_test
        ]
    },
    {
        'page': 'data_distribution.md',
        'classes': [
            (data_distribution.data_distribution.DataDistribution, ["get_federated_data", "make_data_federated"]),
            (data_distribution.data_distribution_iid.IidDataDistribution, ['make_data_federated']),
            (data_distribution.data_distribution_non_iid.NonIidDataDistribution, ['make_data_federated',
                                                                                  'choose_labels'])
        ]
    },
    {
        'page': 'model.md',
        'classes': [
            (model.model.TrainableModel, ["train", "predict", "evaluate", "get_model_params", "set_model_params",
                                          'performance']),
            (model.deep_learning_model.DeepLearningModel, ["train", "predict", "evaluate", "get_model_params",
                                                           "set_model_params", 'performance', '_check_data',
                                                           '_check_labels']),
            (model.linear_regression_model.LinearRegressionModel, ["train", "predict", "evaluate", "get_model_params",
                                                                   "set_model_params", 'performance', '_check_data',
                                                                   '_check_labels', '_check_initialization']),
            (model.kmeans_model.KMeansModel, ["train", "predict", "evaluate", "get_model_params", "set_model_params",
                                              'performance']),
            (model.logistic_regression_model.LogisticRegressionModel, ["train", "predict", "evaluate",
                                                                       "get_model_params", "set_model_params",
                                                                       'performance', '_check_data', '_check_labels',
                                                                       '_check_initialization'])
        ]
    },
    {
        'page': 'federated_aggregator.md',
        'classes': [
            (federated_aggregator.federated_aggregator.FederatedAggregator, ["aggregate_weights"]),
            (federated_aggregator.fedavg_aggregator.FedAvgAggregator, ['aggregate_weights']),
            (federated_aggregator.weighted_fedavg_aggregator.WeightedFedAvgAggregator, ['aggregate_weights']),
            (federated_aggregator.iowa_federated_aggregator.IowaFederatedAggregator, ['aggregate_weights',
                                                                                      'set_ponderation', 'q_function',
                                                                                      'get_ponderation_weights']),
            (federated_aggregator.cluster_fedavg_aggregator.ClusterFedAvgAggregator, ['aggregate_weights'])
        ]
    },
    {
        'page': 'federated_government.md',
        'classes': [
            (federated_government.federated_government.FederatedGovernment, ['evaluate_global_model',
                                                                             'deploy_central_model',
                                                                             'evaluate_clients',
                                                                             'train_all_clients',
                                                                             'aggregate_weights',
                                                                             'run_rounds']),
            (federated_government.federated_images_classifier.FederatedImagesClassifier, ['run_rounds',
                                                                                          'model_builder']),
            (federated_government.federated_images_classifier.Reshape, ['apply']),
            federated_government.federated_images_classifier.ImagesDataBases,
            (federated_government.federated_linear_regression.FederatedLinearRegression, ['run_rounds',
                                                                                          'model_builder']),
            federated_government.federated_linear_regression.LinearRegressionDataBases,
            (federated_government.federated_clustering.FederatedClustering, ['run_rounds',
                                                                             'model_builder']),
            federated_government.federated_clustering.ClusteringDataBases,
            (federated_government.iowa_federated_government.IowaFederatedGovernment, ['performance_clients',
                                                                                      'run_rounds'])
        ]
    },
    {
        'page': 'differential_privacy/mechanisms.md',
        'classes': [
            (differential_privacy.dp_mechanism.RandomizedResponseCoins, ['apply']),
            (differential_privacy.dp_mechanism.RandomizedResponseBinary, ['apply']),
            (differential_privacy.dp_mechanism.LaplaceMechanism, ['apply']),
            (differential_privacy.dp_mechanism.GaussianMechanism, ['apply']),
            (differential_privacy.dp_mechanism.ExponentialMechanism, ['apply'])
        ],
    },
    {
        'page': 'differential_privacy/sensitivity_sampler.md',
        'classes': [
            differential_privacy.sensitivity_sampler.SensitivitySampler
        ],
        'methods': [
            differential_privacy.sensitivity_sampler.SensitivitySampler.sample_sensitivity,
            differential_privacy.sensitivity_sampler.SensitivitySampler._sensitivity_sampler,
            differential_privacy.sensitivity_sampler.SensitivitySampler._sensitivity_norm,
            differential_privacy.sensitivity_sampler.SensitivitySampler._sensitivity_sampler_config
        ],
    },
    {
        'page': 'differential_privacy/norm.md',
        'classes': [
            (differential_privacy.norm.SensitivityNorm, ["compute"]),
            (differential_privacy.norm.L1SensitivityNorm, ['compute']),
            (differential_privacy.norm.L2SensitivityNorm, ['compute'])
        ],
    },
    {
        'page': 'differential_privacy/probability_distribution.md',
        'classes': [
            (differential_privacy.probability_distribution.ProbabilityDistribution, ["sample"]),
            (differential_privacy.probability_distribution.NormalDistribution, ['sample']),
            (differential_privacy.probability_distribution.GaussianMixture, ['sample'])
        ],
    },
    {
        'page': 'differential_privacy/composition.md',
        'classes': [
            differential_privacy.composition_dp.ExceededPrivacyBudgetError,
            (differential_privacy.composition_dp.AdaptiveDifferentialPrivacy, ['apply',
                                                                               '_get_data_access_definition',
                                                                               ])
        ],
        'functions': [
            differential_privacy.composition_dp._check_differentially_private_mechanism
        ],
    },
    {
        'page': 'differential_privacy/sampling.md',
        'classes': [
            (differential_privacy.dp_sampling.Sampler, ['apply', 'epsilon_delta_reduction', 'sample']),
            (differential_privacy.dp_sampling.SampleWithoutReplacement, ['sample', 'epsilon_delta_reduction'])
        ],
        'functions': [
            differential_privacy.dp_sampling.prod,
            differential_privacy.dp_sampling.check_sample_size
        ],
    }
]
ROOT = 'http://127.0.0.1/'
