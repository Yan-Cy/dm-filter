hive -e "
set mapreduce.job.queuename=crontab;
use db;
select * from db.ods_dm_report where log_date>0 and cid=17139068" > "./mayun_poorhappy_report.txt"

