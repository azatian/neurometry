# %%
import subprocess

#This part is for making all the directory under cutouts/
for i in range(600):
    cmd = "mkdir " + "cutouts/dcvsyn" + str(i+1)
    _temp = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    output = _temp.communicate()[0]
# %%
#This part is for making the subdirectories under cutouts/dcvsynx

for i in range(600):
    cmd = "mkdir " + "cutouts/dcvsyn" + str(i+1) + "/img" + " cutouts/dcvsyn" + str(i+1) + "/presyn" + " cutouts/dcvsyn" + str(i+1) + "/postsyn"
    _temp = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    output = _temp.communicate()[0]

# %%
for i in range(600):
    cmd = "python3 wk.py dcvsyn" + str(i+1)
    _temp = subprocess.Popen(cmd, stdout=subprocess.PIPE, shell=True)
    output = _temp.communicate()[0]

# %%
