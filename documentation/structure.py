from shfl import differential_privacy

# For each class to document, it is possible to:
# 1) Document only the class: [classA, classB, ...]
# 2) Document all its methods: [classA, (classB, "*")]
# 3) Choose which methods to document (methods listed as strings):
# [classA, (classB, ["method1", "method2", ...]), ...]
# 4) Choose which methods to document (methods listed as qualified names):
# [classA, (classB, [module.classB.method1, module.classB.method2, ...]), ...]

PAGES = [
    {
        'page': 'Differential privacy/Mechanisms.md',
        'classes': [
            differential_privacy.dp_mechanism.DifferentialPrivacyMechanism,
            differential_privacy.dp_mechanism.UnrandomizedMechanism,
            differential_privacy.dp_mechanism.RandomizeBinaryProperty,
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
    }
]
ROOT = 'http://127.0.0.1/'
