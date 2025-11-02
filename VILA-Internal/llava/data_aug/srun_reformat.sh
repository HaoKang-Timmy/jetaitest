# nvcode: on
JOBS_LIMIT=100  # Set your limit here
ACCOUNT=${ACCOUNT:-VILA_SLURM_ACCOUNT}
PARTITION=${PARTITION:-cpu,cpu_long} #draco: cpu,cpu_long,batch_singlenode,grizzly,polar
src_folder=$HOME/dataset/sam-raw

for f in $src_folder/*.tar; do
  while [ $(jobs -rp | wc -l) -ge $JOBS_LIMIT ]; do
    sleep 5
  done

  fname=$(echo $f | rev | cut -d "/" -f 1 | rev)

  echo "Running $fname,  now total jobs $(jobs -rp | wc -l)"; \
  srun --label -A $ACCOUNT -N 1 \
    -p $PARTITION -t 1:00:00 \
    -J $ACCOUNT-dev:reformat-$fname \
    -e slurm-logs/dev-split/$fname-$j.err -o slurm-logs/dev-split/$fname-$j.out \
    python llava/data_aug/reformat_tar.py --src_tar=$f --src_folder=$src_folder \
      --tgt_folder=$HOME/dataset/sam-reformat --overwrite=True &
done
wait
