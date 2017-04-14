hive -e "
set mapreduce.job.queuename=crontab;
use db;
select * from dm_index" > "./dm_index.txt"

