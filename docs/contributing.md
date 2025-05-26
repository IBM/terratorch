# Contributing to TerraTorch

Contributions to an open source project can came in different ways, but we could summarize them in three main
components: adding code (as new models, tasks and
auxiliary algorithms or even addressing the solution of a bug), examples using the software (scripts, yaml files and notebooks showcasing the package) and documentation.
All these ways are valid for TerraTorch
and the users are welcome to contribute in any of these fronts. However, some recommendations and rules are
necessary in order to facilitate and organize his process. And this is the matter of the next paragraphs. 

## Contributing with code

It is not a trivial task to determine how a modification in the source code will impact already implemented
and established features, in this way, for any modification in the core source code (`terratorch/`) we
automatically execute
a pipeline with hundreds of unit and integration tests to verify that the package have not broken after the
modification be merged to `main`. In this way, when an user wants to modify
`terratorch` for adding new features or bufixes, this are the best practices. 

* This repository uses `pre-commit`, a tool which automatically runs basic
    steps before sending modifications to the remote (as linting, for example).
    See how to configure it [here](https://pre-commit.com/#installation). 
* If you are an user outside the IBM org, create a fork to add your modifications. If you are inside the IBM
    org or have received writing provileges, create a branch for it. 
* If you are adding new features, we ask you to also add tests for it. These tests are defined in the
    directory `tests/` and are fundamental to check if your feature is working as expected and not breaking
    anything. If your feature is something more complex, as a new model or auxiliary algorithm, you can also
    (optionally) to add a complete example, as a notebook, demonstrating how the feature works.
* After finishing your modifications, we recommend you to test locally using `pytest`, for example:
    ```
    pytest -s -v tests/
    ```
* If all the tests are passing, you can open a PR to `terratorch:main` describing what you are adding and why
    that is important to be merged. You
    do not need to choose a reviewer, since the maintainers will check the new open PR and request review for it by themselves.  
* The PR will pass through the tests in GitHub Actions and if the reviewer approve it, it will soon be merged. 
* It is recommended to add a label to your PR. For example `bug`, when it solves some issue or `enhancement`
    when it adds new features. 

**NOTICE**: The PR will not be merged if the automatic tests are failing and the user which has sent the PR is
    responsible for fixing it. 

## Contributing with documentation

Documentation is core for any project, however, most part of the time, the developers do not have the time (or
patience) to carefully document all the codebase, in this way, contributions from interested users are always
welcome. 
To add documentation to TerraTorch, you need to be familiar with Markdown, a clean markup language, and
MkDocs, a framework which relies on Markdown in order to create webpages as this you are reading. 

* Install the MkDocs dependencies. Install as a developer `pip install terratorch[dev]` to include them or manually using [this list](https://github.com/IBM/terratorch/blob/cdb130dd0be3db053c8858d69e8f02197d42c567/pyproject.toml#L66).
* Clone the branch dedicated to documentation to a local branch:
```
    git fetch origin improve/docs
```
* Add your modifications and open a PR to `improve/docs`. It is recommended to add the label `documentation` to your PR. 
* The PR will be reviewed and approved if it is considered relevant by the maintainers. 

## Contributing by reporting issues

The users also can contribute by reporting issues they observed during their experiments. When reporting an issue, provide the more details as possible about the problem, as the configuration of your system,
the terminal output and even the files required to reproduce it. To
illustrate it, take a
look on an example of [a good issue](https://github.com/IBM/terratorch/issues/506). 
