import os

def binary_search(x, a):
    l = 0
    r = len(a) - 1
    while l <= r:
        mid = (l + r) / 2
        if a[mid] == x:
            return True
        elif a[mid] > x:
            r = mid - 1
        elif a[mid] < x:
            l = mid + 1
    return False

def filter_dm(dmsrc, dmreport):
    #pos = open('positive.dm', 'w')
    deleted_dm_list = {}
    with open(dmsrc) as f:
        for line in f:
            #dmid = int(line.split('\t')[0])
            #content = unicode(line.split('\t')[5].strip(), 'utf-8')
            deleted = int(line.split('\t')[12])
            #if deleted == 2: # Protected DM
            #    pos.write(content.encode('utf8') + '\n')
            if deleted == 1: # Deleted DM
                dmid = int(line.split('\t')[0])
                content = unicode(line.split('\t')[5].strip(), 'utf-8')
                deleted_dm_list[dmid] = content
    #pos.close()
   
    neg = open('negative.dm', 'w')
    with open(dmreport) as f:
        for line in f:
            dmid = int(line.split('\t')[2])
            if dmid in deleted_dm_list:
                #print dmid
                neg.write(deleted_dm_list[dmid].encode('utf8') + '\n')
    neg.close()


if __name__ == '__main__':
    dmsrc = '../scripts/dm.txt'
    dmreport = '../scripts/dm_report.txt'
    filter_dm(dmsrc, dmreport)

