import os

report_table = '../scripts/dm_report.txt'
dm_table = '../scripts/dm.txt'
maxpositive = 25000000

class MySentence(object):
    def __init__(self, filename):
        self.filename = filename

    def __iter__(self):
        with open(self.filename) as f:
            for line in f:
                yield [int(line.split('\t')[0]), line.split('\t')[5]]
                
def get_report_dm(report_table):
    with open(report_table) as f:
        reportdm = [int(x.split('\t')[2]) for x in f.readlines()]
    return sorted(reportdm)


def find(x, a):
    l = 0
    r = len(a) - 1
    while l <= r:
        mid = (l + r) / 2
        if a[mid] == x:
            return True
        if a[mid] > x:
            r = mid - 1
        else:
            l = mid + 1
    return False

if __name__ == '__main__':
    sentences = MySentence(dm_table)
    reportdm = get_report_dm(report_table)
    print len(reportdm)
    negf = open('negative.dm', 'w')
    posf = open('positive.dm', 'w')
    for s in sentences:
        if find(s[0], reportdm):
            negf.write(s[1] + '\n')
        else:
            posf.write(s[1] + '\n')
    negf.close()
    posf.close()
