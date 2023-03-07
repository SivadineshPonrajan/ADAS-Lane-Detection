import os

def removePNGs(path):
	allFiles = os.listdir(path)
	for file in allFiles:
		if(file.endswith(".png")):
			os.remove(path+file)
			print("Removing the file " + path+file)

removePNGs("./output/")
removePNGs("./lidar_scatter/")