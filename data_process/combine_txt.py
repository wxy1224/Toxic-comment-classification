import os 

def combine_txt(folder, output_file):

	all_files = os.listdir(folder)
	txt_files = filter(lambda x: x[-4:] == '.txt', all_files)


	contents = []
	for filename in txt_files:
		with open (folder+filename,'r') as file:
			content = file.read().split('\n')
			content = list(filter(None, content))
			contents = contents+content

	contents = list(set(contents))
	print(len(contents))
	filename = output_file
	with open(filename, 'w') as file:
		for item in contents:
			file.write("%s\n" % item)

if __name__ == '__main__':
	folder = "./names/"
	output_file = "./names/names.txt"
	combine_txt(folder, output_file)
	folder2 = "./names_threat/"
	output_file2 = ".、names_threat/names.txt"
	combine_txt(folder2, output_file2)

