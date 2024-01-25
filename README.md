## ENE to END Machine Learning Project

### Create virtual enviroment
conda create -p weather_rain python==3.8 -y

### activate the enviroment
conda activate C:\weather_rain\weather_rain



### Git commands

## For New
"""
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin https://github.com/maheshpachpande/weather_rain.git
git push -u origin main
"""
## for Existing
git add .
git status
git commit -m "msg commit"
git push -u origin main

## for large file
git lfs install
git lfs track "*.pkl"
git lfs push --all origin main
git add .
git commit -m "model1"
git push -u origin main

### create the .gitignore with python and commit the changes
## For updation
git pull

## Create setup.py and requirements.txt

## Create the scr folder with __init__.py
## install requirements
pip install -r requirements.txt

## In src folder create the components folder with __init__.py, data_ingestion.py, data_transformation.py, model_trainer.py
## In src folder create the pipeline folder with __init__.py, pipeline_prediction.py, pipeline_training.py
## In src folder create the logger.py, exception.py, utils.py (check the exception.py by command 'python src/exception.py')


## In cutomerchurn folder Create notebooks folder (EDA.ipynb, model_training.ipynb) with data folder.







