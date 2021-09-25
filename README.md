"# ai4i2020-linear-regr-dvc-deploy for demo" 

create env 

```bash
conda create -n machineLearning python=3.7 -y
```

activate env
```bash
conda activate machineLearning
```

created a req file

install the req
```bash
pip install -r requirements.txt
```
download the data from 

https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv

```bash
git init
```
```bash
dvc init 
```
```bash
dvc add data_given/ai4i2020.csv
```
```bash
git add .
```
```bash
git commit -m "first commit"
```

oneliner updates  for readme

```bash
git add . && git commit -m "update Readme.md"
```
```bash
git remote add origin https://github.com/Udaykiran87/ai4i2020-linear-regr-dvc-deploy.git
git branch -M main
git push origin main
```

tox command -
```bash
tox
```
for rebuilding -
```bash
tox -r 
```
pytest command
```bash
pytest -v
```

setup commands -
```bash
pip install -e . 
```

build your own package commands- 
```bash
python setup.py sdist bdist_wheel
```
