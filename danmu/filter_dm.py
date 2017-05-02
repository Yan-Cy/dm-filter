import os
import random

def binary_search(x, a):
    l = 0
    r = len(a) - 1
    while l <= r:
        mid = (l + r) / 2
        if (a[mid] == x) or (x in a[mid]) or (a[mid] in x):
            return mid
        elif a[mid] > x:
            r = mid - 1
        elif a[mid] < x:
            l = mid + 1
    return -1

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


def select_dm(src, dst, n, tdst, m):
    with open(src) as f:
        dm = [unicode(line, 'utf-8') for line in f.readlines()]
    random.shuffle(dm)
    with open(dst, 'w') as f:
        for i in xrange(n):
            f.write(dm[i].encode('utf-8'))
        
    with open(tdst, 'w') as f:
        for i in xrange(m):
            f.write(dm[n+i].encode('utf-8'))

def cleandm(possrc, posdst, negsrc, negdst):
    print 'Loading positive dm...'
    with open(possrc) as f:
        pos = list(set([line.strip() for line in f.readlines()]))

    print 'Sorting positive dm...'
    pos.sort()

    print 'loading negative dm...'
    with open(negsrc) as f:
        neg = list(set([line.strip() for line in f.readlines()]))
    
    print 'Sorting negative dm...'
    neg.sort()

    print 'Pos len:', len(pos)
    print 'Neg len:', len(neg)

    posclean = []
    duplicate = []
    negclean = set()
    count = 0

    print 'Looking for duplicates...'
    for i, sent in enumerate(pos):
        t = binary_search(sent, neg)
        if t == -1:
            posclean.append(sent)
        else:
            negclean.add(t)
            duplicate.append(sent)
            count += 1
            #print i, count
            #print sent, neg[t]

    print 'Saving positive dm...'
    with open(posdst, 'w') as f:
        for posdm in posclean:
            f.write(posdm + '\n')

    print 'saving negative dm...'
    with open(negdst, 'w') as f:
        for i, negdm in enumerate(neg):
            if i not in negclean:
                f.write(negdm + '\n')

    with open('duplicate.txt', 'w') as f:
        for dup in duplicate:
            f.write(dup + '\n') 

    print 'Total Duplicate Found: ', count

if __name__ == '__main__':
    #dmsrc = '../scripts/dm.txt'
    #dmreport = '../scripts/dm_report.txt'
    #filter_dm(dmsrc, dmreport)
    '''
    dmsrc = 'protected.dm'
    dmdst = 'positive.train'
    tdst = 'positive.test'
    select_dm(dmsrc, dmdst, 3000000, tdst, 300000)
    dmsrc = 'deleted_report.dm'
    dmdst = 'negative.train'
    tdst = 'negative.test'
    select_dm(dmsrc, dmdst, 1000000, tdst, 100000)
    '''
    possrc = 'protected.dm'
    posdst = 'positive.clean'
    negsrc = 'deleted_report.dm'
    negdst = 'negative.clean'
    cleandm(possrc, posdst, negsrc, negdst)
