set -e
model=$1
ont=$2
python evaluate.py -m $model -ont $ont -dr /data/zhapacfp/dgpu-sim/data
# echo "--------------------"
# python evaluate.py -m $model -ont $ont -c -dr /data/zhapacfp/dgpu-sim/data
echo "===================="
