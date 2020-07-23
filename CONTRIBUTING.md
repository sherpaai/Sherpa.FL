# Bug Reporting, Feature Request or Pull Request

We are happy to accept contributions in different ways.

## Bug Reporting
If you found a bug or unexpected behaviour in the framework please, follow the next steps to report.

1. Check your code version. Maybe the problem is already fixed in a new version.

2. Search in the opened issues to avoid duplicity.

3. If bug is not covered yet, please, provide as much information as possible about your environment and, if possible, 
provide some code to reproduce the behavior.

4. If you are able to solve the problem you might propose a solution in a pull request.

## Feature Request

If you are interested in a new feature that is not developed at the moment you can use the issue tracker to request it.
 Just be sure that you explain in the clearest possible way the new behaviour that you would like. A good option is to 
 provide some pseudocode or schema to clarify the new feature.

## Pull Requests

If you are going to add some code that modifies the software architecture or changes the behavior of a functionality is 
recommended to describe the changes to discuss and avoid to waste time. If you are just fixing an evident bug it is not 
necessary.


### Developing process

The following are the basic conventions that we are using in this project.

#### Github Workflow

The main strategy that we are using is the "Forking Workflow" that is very well described [here](https://www.atlassian.com/git/tutorials/comparing-workflows/forking-workflow) (in this case for Bitbucket, but the same can be applied for github).

#### Code Style

We are using [Google Python Style Guide](http://google.github.io/styleguide/pyguide.html)

#### Code Tests

All tests have to pass with 100% line coverage. You only need to execute the following command in the base project directory:
`pytest --cov=shfl test/`
