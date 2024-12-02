# ECG-research

### Environment
Requirement Python 3.13

```sh
python -m venv env
. env/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

python src/main.py -c H_P001_PPG_S_S1 gen-fr
python src/main.py -c H_P001_PPG_S_S1 plot-fr
python src/main.py -c H_P001_PPG_S_S1 plot-statistics
python src/main.py -c H_P001_PPG_S_S1 plot-autocorrelation
python src/main.py -c H_P001_PPG_S_S1 plot-autocovariation

```