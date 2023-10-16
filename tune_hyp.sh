set -e


data_root=/data/zhapacfp/dgpu-sim/data/
ont=cc
batch_size=256 #default
prior=1e-4 #default
#gamma
#probabilty
probability_rate=0.01 #default
#alpha
loss=pun_multi #default
max_lr=1e-4 #default

for gamma in 0.01 0.02 0.03 0.04 0.05
do
    for alpha in 0.1 0.3 0.5 0.7 0.9
    do
	for prob in 0 0.1 0.3 0.5 0.7 0.9
	do
	    python pu_base_sample_prior.py -dr /data/zhapacfp/dgpu-sim/data/ -ont $ont -bs $batch_size -p $prior -g $gamma -prob $prob -prate $probability_rate -a $alpha -loss pun_multi -lr $max_lr

	    python evaluate.py -m dgpu_sample_prior -ont $ont -dr /data/zhapacfp/dgpu-sim/data #-bs $batch_size -p $prior -g $gamma -prob $prob -prate $probability_rate -a $alpha -loss pun_multi -lr $max_lr
	    echo "===================="
	    # ./eval.sh dgpu mf >> result_mf.log
	done
    done
done

