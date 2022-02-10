import os, rarfile, tarfile, shutil

def ex_tar(t_file):
	tkk = tarfile.open(t_file)
	mt = t_file.split(".")[0]
	if os.path.exists(mt):
	    shutil.rmtree(mt)
	tkk.extractall(mt)

def ex_rar(r_file, dst_folder=None):
	mt = r_file.split(".")[0]
  if dst_folder != None:
	mt = dst_folder
	with rarfile.RarFile(r_file) as file:
		file.extractall(path=mt)
