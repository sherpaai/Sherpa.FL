# Sherpa Federated Learning (Sherpa.FL)

[Sherpa.ai](http://sherpa.ai)'s Federated Learning Platform (Sherpa.FL) has been developed to facilitate open research in the ﬁeld, with the objective of building models that learn from decentralized data, preserving data privacy. It is an open-source platform and aims at supporting 100 percent of the AI algorithms used in industry.

Sherpa.FL is an open-source framework for Machine Learning that is dedicated to data privacy protection. It has been developed to facilitate open research and experimentation in Federated Learning, a machine learning paradigm aimed at learning models from decentralized data (e.g., data located on users’ smartphones) and ensuring data privacy. This is achieved by training the model locally in each node (e.g., on each smartphone), sharing the model-updated parameters (not the data) and securely aggregating the updated parameters to build a better model. This technology could be disruptive in cases where it is compulsory to ensure data privacy, as in the following examples:

*    When data contains sensitive information, such as email accounts, personalized recommendations, and health information, applications should employ data privacy mechanisms to learn from a population of users whilst the sensitive data remains on each user’s device.

*    When data is located in data silos, an automotive parts manufacturer, for example, may be reluctant to disclose their data, but would benefit from models that learn from each other's data, in order to improve production and supply chain management.

*    Due to data-privacy legislation, banks and telecom companies, for example, cannot share individual records, but would benefit from models that learn from data across several entities.

Sherpa.ai is focused on democratizing Federated Learning by providing methodologies, pipelines, and evaluation techniques specifically designed for Federated Learning. The Sherpa.ai Federated Learning Platform enables developers to simulate Federated Learning scenarios with models, algorithms, and data provided by the framework, as well as their own data.

Sherpa.FL is a project of [Sherpa.ai](http://sherpa.ai) in collaboration with the [Soft Computing and Intelligent Information Systems (SCI2S)](https://sci2s.ugr.es/) research group from the [University of Granada](https://www.ugr.es/).

## Installation

See the [install](docs/install.md) documentation for instructions on how to install Sherpa.FL.

## Getting Started

See the [get started](docs/get_started.md) documentation for a brief introduction to using Sherpa.FL.

## Contributing

If you are interested in contributing to Sherpa.FL with tutorials, datasets, models, aggregation mechanisms or any other code others could benefit of please be sure to review the [contributing guidelines](CONTRIBUTING.md).

## Issues

Use [GitHub issues](https://github.com/sherpaai/Sherpa.FL/issues) for tracking requests and bugs.

## Questions

Please direct questions to [Sherpa Developers Slack](https://sherpa-developers-invitation.herokuapp.com/).
