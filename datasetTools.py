
# coding: utf-8

# In[8]:

"""
import glob
with open('CliListTest.txt', "w") as f:
    l = glob.glob('/home/data/collection/GUIMUTEIC/CLICIDE/CLICIDEMAX/test/*.JPG')
    print(len(l))
    for a in l:
        if not 'wall' in a:
            f.write(a+"\n")
"""
with open('FouList.txt', "r") as f:
    with open('FouConcept.txt' ,"w") as fout:
        a = set()
        for l in f:
            floor, nb, _ = l.split('/')[-1].split('_')
            a.add(floor+'_'+nb)
        for e in a:
            fout.write(e+'\n')

