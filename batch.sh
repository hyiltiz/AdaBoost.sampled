#! /bin/bash

python2 adaboost.py cv -k 10 -i breast-cancer.txt -r .1 -T 1e3 --seed 0
cp *.eps *.pdf results/
python2 adaboost.py cv -k 10 -i breast-cancer.txt -r .25 -T 1e3 --seed 0
cp *.eps *.pdf results/
python2 adaboost.py cv -k 10 -i breast-cancer.txt -r 1 -T 1e3 --seed 0
cp *.eps *.pdf results/

python2 adaboost.py cv -k 10 -i ionosphere_scale -r .1 -T 1e3 --seed 0
cp *.eps *.pdf results/
python2 adaboost.py cv -k 10 -i ionosphere_scale -r .25 -T 1e3 --seed 0
cp *.eps *.pdf results/
python2 adaboost.py cv -k 10 -i ionosphere_scale -r 1 -T 1e3 --seed 0
cp *.eps *.pdf results/

python2 adaboost.py cv -k 10 -i cod-rna.txt -r 0.1 -T 1e3 --seed 0
cp *.eps *.pdf results/
