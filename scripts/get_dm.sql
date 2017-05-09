hive -e "
set mapreduce.job.queuename=crontab;
use db;
select * from ods_dm_report" > "./dm_index.txt"

