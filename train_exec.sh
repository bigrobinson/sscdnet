python3 train.py  \
--cvset 0 \
--use-corr \
--datadir /home/bigrobinson/sscdnet/data/pcd_5cv \
--checkpointdir /home/bigrobinson/sscdnet/log \
--max-iteration 50000 \
--num-workers 16 \
--batch-size 8 \
--icount-plot 50 \
--icount-save 10000
