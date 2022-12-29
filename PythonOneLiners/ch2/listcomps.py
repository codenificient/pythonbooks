## Data
text = '''
Call me Ishmael. Some years ago - never mind how long precisely - having
little or no money in my purse, and nothing particular to interest me
on shore, I thought I would sail about a little and see the watery part
of the world. It is a way I have of driving off the spleen, and regulating
the circulation. - Moby Dick'''

"""
    Print all lines of text that are longer than 3
"""
w = [[x for x in line.split() if len(x) > 4] for line in text.split('\n')]
print(w)