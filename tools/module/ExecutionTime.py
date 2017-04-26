# _*_ coding: utf-8 -*-

import time
import datetime

class ExecutionTime:
    """
        前のstartからの時間を計測
    """
    def __init__(self):
        self._t = 0
        
    def start(self):
        print (datetime.datetime.today())
        self._t = time.time()

    def end(self): 
        end = time.time()
        print (datetime.datetime.today())
        print ("%d min %d sec" % ((end - self._t)/60 , (end - self._t)%60))

