set -e

for beta in 0.1 0.3 0.5 0.7 0.9
do
    for gamma in 0.1 0.3 0.5 0.7 0.9
    do
	for alpha in 0.1 0.3 0.5 0.7 0.9
	do
	    python pu_se.py -g $gamma  -b $beta -loss pun_multi -a $alpha -ont mf
	    ./eval_sem.sh dgpu mf >> result_sem_mf.log
	done
    done
done

