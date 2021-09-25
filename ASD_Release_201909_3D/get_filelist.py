f=open("filelist",'r')
lines=f.readlines()
files=lines[0].split(' ')
f_write=open('files','a')
for file in files:
	f_write.write(file+'\n')
	
