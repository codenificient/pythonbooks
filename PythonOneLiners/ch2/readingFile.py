filename = "readingFile.py" # this code

f = open(filename)
lines = []

for line in f:
    lines.append(line.strip())

# print(lines)

############## One Liner #################
print([line.strip() for line in open("readingFile.py")])