import random

f = open('inputs3.txt','w')
f2 = open('targets3.txt','w')
dataset = []
for i in range(10000):
    size = random.randint(2,3)
    s= 0
    l = []
    while s<size:
        x=random.randint(0,4)
        if x not in l:
            l.append(x)

            s+=1
        dataset.append(l)
    f.write(' '.join([str(i) for i in l]))
    f.write('\n')
    f2.write(' '.join([str(i) for i in sorted(l)]))
    f2.write('\n')
f.close()
f2.close()
