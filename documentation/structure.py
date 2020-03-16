from shfl import differential_privacy
from shfl import private
from shfl import model
from shfl import data_base
from shfl import learning_approach
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
        'page': 'Private/DataNode.md',
        'classes': [
            private.node.DataNode
        ],
        'methods': [
            private.node.DataNode.set_private_data,
            private.node.DataNode.configure_private_data_access,
            private.node.DataNode.configure_model_params_access,
            private.node.DataNode.apply_data_transformation,
            private.node.DataNode.query_private_data,
            private.node.DataNode.query_model_params,
            private.node.DataNode.set_model_params,
            private.node.DataNode.train_model,
            private.node.DataNode.predict,
            private.node.DataNode.evaluate,
        ],
    },
{
        'page': 'Private/Data.md',
        'classes': [
            private.data.DataAccessDefinition,
            private.data.UnprotectedAccess,
            private.data.LabeledData
        ]
    },
    {
        'page': 'Private/Query.md',
        'classes': [
            (private.query.Query, ["get"]),
            private.query.IdentityFunction,
            private.query.Mean
        ]
    },
    {
        'page': 'Private/Federated Operation.md',
        'classes': [
            (private.federated_operation.FederatedData, ["add_data_node", "num_nodes",
                                                         "configure_data_access", "query"]),
            private.federated_operation.FederatedTransformation
        ],
        'functions': [
            private.federated_operation.federate_array,
            private.federated_operation.apply_federated_transformation
        ]
    },
    {
        'page': 'Databases.md',
        'classes': [
            (data_base.data_base.DataBase, ["shuffle"]),
            data_base.emnist.Emnist,
            data_base.fashion_mnist.FashionMnist,
            data_base.california_housing.CaliforniaHousing
        ],
        'functions': [
            data_base.data_base.extract_validation_samples
        ]
    },
    {
        'page': 'Data Distribution.md',
        'classes': [
            (data_distribution.data_distribution.DataDistribution, ["get_federated_data", "make_data_federated"]),
            data_distribution.data_distribution_iid.IidDataDistribution,
            data_distribution.data_distribution_non_iid.NonIidDataDistribution
        ]
    },
    {
        'page': 'Model.md',
        'classes': [
            (model.model.TrainableModel, ["train", "predict", "evaluate", "get_model_params", "set_model_params"]),
            model.deep_learning_model.DeepLearningModel
        ]
    },
    {
        'page': 'Federated Aggregator.md',
        'classes': [
            (federated_aggregator.federated_aggregator.FederatedAggregator, ["aggregate_weights"]),
            federated_aggregator.avgfed_aggregator.AvgFedAggregator,
            federated_aggregator.weighted_avgfed_aggregator.WeightedAvgFedAggregator
        ]
    },
    {
        'page': 'Learning Approach.md',
        'classes': [
            (learning_approach.learning_approach.LearningApproach, ["train_all_clients", "aggregate_weights",
                                                                    "run_rounds"]),
            learning_approach.federated_government.FederatedGovernment
        ]
    },
    {
        'page': 'Differential privacy/Mechanisms.md',
        'classes': [
            differential_privacy.dp_mechanism.DifferentialPrivacyMechanism,
            differential_privacy.dp_mechanism.UnrandomizedMechanism,
            differential_privacy.dp_mechanism.RandomizedResponseCoins,
            differential_privacy.dp_mechanism.RandomizedResponseBinary,
            differential_privacy.dp_mechanism.LaplaceMechanism
        ],
    },
    {
        'page': 'Differential privacy/Sensitivity Sampler.md',
        'classes': [
            differential_privacy.sensitivity_sampler.SensitivitySampler
        ],
        'methods': [
            differential_privacy.sensitivity_sampler.SensitivitySampler.sample_sensitivity
        ],
    },
    {
        'page': 'Differential privacy/Norm.md',
        'classes': [
            (differential_privacy.norm.SensitivityNorm, ["compute"]),
            differential_privacy.norm.L1SensitivityNorm
        ],
    },
    {
        'page': 'Differential privacy/Probability Distribution.md',
        'classes': [
            (differential_privacy.probability_distribution.ProbabilityDistribution, ["sample"]),
            differential_privacy.probability_distribution.NormalDistribution,
            differential_privacy.probability_distribution.GaussianMixture
        ],
    }
]
ROOT = 'http://127.0.0.1/'
