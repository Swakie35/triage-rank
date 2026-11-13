import argparse
from .benchmark import report
p=argparse.ArgumentParser();p.add_argument('--report',default='reports/generated/benchmark.json')
a=p.parse_args();r=report(a.report)
print('Synthetic research benchmark:',r)
