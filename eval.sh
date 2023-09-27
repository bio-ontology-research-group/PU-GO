set -e
model=$1
ont=$2
python evaluate2.py -m $model -ont $ont
echo "--------------------"
python evaluate.py -m $model -ont $ont
echo "===================="
