today=$(date --date="-0 day" +"%Y-%m-%d 00:00:00")
yesterday=$(date --date="-1 day" +"%Y-%m-%d 00:00:00")
echo $today
echo $yesterday
today_file=$(date --date="-0 day" +"%Y-%m-%d")

hive -e "
set mapreduce.job.queuename=crontab;
use db;
select * from dmtable where num>=0 and times>=unix_timestamp('$yesterday') and times<=unix_timestamp('$today')" > "./testfile.txt"

