#!/bin/bash
source /etc/profile
today=$(date --date="-0 day" +"%Y-%m-%d 00:00:00")
yesterday=$(date --date="-1 day" +"%Y-%m-%d 00:00:00")
echo $today
echo $yesterday
today_file=$(date --date="-0 day" +"%Y-%m-%d")
yesterday_log_date=$(date --date="-1 day" +"%Y%m%d")
touch /mnt/storage01/nlp/dengyanlei/danmu_research/daily_report/data/dm_table_daily_$today_file
touch /mnt/storage01/nlp/dengyanlei/danmu_research/daily_report/data/dm_report_daily_$today_file
hive -e "set hive.auto.convert.join=false;
set mapreduce.job.queuename=crontab;
use db;
select typeid, deleted,count(*)
from (select * from dmtable where num>=600) as dmtable inner join dm_index on dmtable.dm_inid = dm_index.dm_inid inner join
db.bilibili_archive as archives on dm_index.dm_lastaid=archives.id
where times>=unix_timestamp('$yesterday') and times<=unix_timestamp('$today')
group by typeid, deleted
">"/mnt/storage01/nlp/dengyanlei/danmu_research/daily_report/data/dm_table_daily_$today_file"

hive -e "set hive.auto.convert.join=false;
set mapreduce.job.queuename=crontab;
use db;
select typeid, dm_report.state, count(*)
from (select * from ods_dm_report where rp_time>=unix_timestamp('$yesterday') and rp_time<=unix_timestamp('$today') and log_date='$yesterday_log_date') as dm_report
inner join dm_index on dm_report.cid = dm_index.dm_inid inner join db.bilibili_archive as archives
on dm_index.dm_lastaid = archives.id
group by typeid, dm_report.state
">"/mnt/storage01/nlp/dengyanlei/danmu_research/daily_report/data/dm_report_daily_$today_file"

