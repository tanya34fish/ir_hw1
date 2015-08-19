#!/bin/bash
python setup.py build_ext --inplace
mkdir -p output_train
python main_cpy.py -r -i ../query/query-train.xml -o output_train -m ../model -d ../CIRB010
python merge.py output_train ../query/ans-train 10
mkdir -p output_test
python main_cpy.py -r -i ../query/query-test.xml -o output_test -m ../model -d ../CIRB010
python merge.py output_test ../query/query-test.xml 20
