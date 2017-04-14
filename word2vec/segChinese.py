#!/usr/bin/env python
# -*- coding: utf-8 -*-
import re

pattern = re.compile("[a-zA-Z0-9]+")
def filterChinese(s):
    sent = []
    start = 0
    for li in pattern.finditer(s):
        middle = ''.join(s[start:li.start()].split())
        sent += list(middle) + [li.group()]
        start = li.end()
    middle = ''.join(s[start:].split())
    sent += list(middle)
    return sent

if __name__ == '__main__':
    s = u"我擦，   什么鬼"  # normal Chinese
    #print filterChinese(s)
    s = u"1999年10月1日，piapia pia" # Chinese, numbers, english words
    #print filterChinese(s)
    s = u"1999年10月1日，piapia pia，" # Chinese, numbers, english words
    #print filterChinese(s)
    s = u"みさか みこと"  # japanese
    #print filterChinese(s)

