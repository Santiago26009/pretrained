# Pre-Trained Model Application

## Description
This repo is intended for testing a simple pre-trained model application which captures answers from a form the user fills out and make a simple prediction based on a pre-trained model. The application uses Django's in-built Authentication system (https://docs.djangoproject.com/en/3.2/topics/auth/).

## Getting started

Copying the repository

Due to the public nature of forks we suggest you duplicate the repo rather then forking it.
You will need to create your own repo e.g. `[your_github_username]/pretrained` and then clone
this repo `luidcrua/pretrained` and push the code into your new one. You can follow the steps for doing this here: https://help.github.com/articles/duplicating-a-repository/

Before proceeding be aware that this exercise assumes you are using a linux machine with [pip](https://pip.pypa.io/en/stable) and [venv](https://docs.python.org/3/library/venv.html) installed.

To initialize the repository in your base directory execute ./initialize_repo.sh

This script will install Django 3.2 and other libraries required for the application. It also loads a fixture to prepopulate the database with some test data for the application

To start the webserver locally, in your base directory execute ./run-server.sh

To run tests locally, in your base directory execute ./run-tests.sh

Access the application at http://localhost:8000
Login credentials have been shared with you via email


## Models description


