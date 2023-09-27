set -e

for alpha in 0.1 0.3 0.5 0.7 0.9
do
    for prob in 0.1 0.3 0.5 0.7 0.9
    do
	for rate in 0.01 0.02 0.03 0.04 0.05
	do
	    python pu_base.py -prob $prob -prate $rate -loss pun_multi -a $alpha 
	    ./eval.sh dgpu mf >> result_mf.log
	done
    done
done

