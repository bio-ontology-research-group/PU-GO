data_root=$1

for ont in mf bp cc; do
    echo $ont
    python diamond_data.py -df $data_root/$ont/train_data.pkl -of $data_root/$ont/train_data.fa
    python diamond_data.py -df $data_root/$ont/valid_data.pkl -of $data_root/$ont/valid_data.fa
    python diamond_data.py -df $data_root/$ont/test_data.pkl -of $data_root/$ont/test_data.fa
    cat $data_root/$ont/valid_data.fa >> $data_root/$ont/train_data.fa
    # Create diamond database
    diamond makedb --in $data_root/$ont/train_data.fa --db $data_root/$ont/train_data.dmnd

    # Run blastp
    diamond blastp --very-sensitive -d $data_root/$ont/train_data.dmnd -q $data_root/$ont/train_data.fa --outfmt 6 qseqid sseqid bitscore pident > $data_root/$ont/train_diamond.res
    diamond blastp --very-sensitive -d $data_root/$ont/train_data.dmnd -q $data_root/$ont/valid_data.fa --outfmt 6 qseqid sseqid bitscore pident > $data_root/$ont/valid_diamond.res
    diamond blastp --very-sensitive -d $data_root/$ont/train_data.dmnd -q $data_root/$ont/test_data.fa --outfmt 6 qseqid sseqid bitscore pident > $data_root/$ont/test_diamond.res
done
