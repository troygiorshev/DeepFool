# The fact that windows doesn't have a remotely functioning shell is terrible

# THIS IS NOT A VIABLE OPTION, KEPT FOR POSTERITY"S SAKE

import os
import glob
import progressbar

files_all = [f for f in glob.glob("**/*.jpeg", recursive=True)]

print("Got all files")

start = 40001
end = 50000

files = [f for f in files_all if int(os.path.basename(f)[18:23]) >= start and int(os.path.basename(f)[18:23]) <= end]

print("Filtered Files")

num_chunks = 2400
size = len(files)//num_chunks

bar = progressbar.ProgressBar(maxval=num_chunks).start()
for i in range(num_chunks):
    file_string = ' '.join(files[i*size:(i+1)*size])
    os.system(".\\7z.exe a -bso0 -bsp0 all" + str(start).rjust(5, '0') + "-" + str(end) + ".7z " + file_string)
    i = i+1
    bar.update(i)