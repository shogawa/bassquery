import sys

import numpy as np

args = sys.argv

with open('datafileC1.txt', 'r') as f1, open('datafileC3.txt', 'r') as f2:
    l1 = f1.readlines()[35:]
    l2 = f2.readlines()[65:]

ll = [[line1[0:19], line1[20:21], line1[22:47], line1[48:57], line1[58:67], line1[68:75], line1[88:95], line1[96:100], line2[20:25], line2[26:31], line2[32:37]] for line1, line2 in zip(l1, l2)]

ll1 = [l for l in ll if l[7] == '    ' and l[1] == ' ' and float(l[9]) <= 22]

ll2 = [l[0].rstrip(' ') for l in ll1]

with open('name.txt', mode='w') as f:
    f.write('\n'.join(ll2))