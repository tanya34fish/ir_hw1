import sys
import utils_cpy
merge = open(sys.argv[1]+'/merge.predict','w');

for i in range(10):
    with open(sys.argv[1]+'/%d.predict' %(i),'r') as f:
        for line in f:
            line = line.strip().split();
            merge.write('%s %s\n' %(line[0],line[1]));

print(utils_cpy.evaluateMAP(sys.argv[2], sys.argv[1]+'/merge.predict'));
