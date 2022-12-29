# Tuples
# index into a tuple, count() and index()
#

# Dictionaries
# Mutable, keys(), values(), items()
#    dict['some_keys']
#   pop(), del() methods

# Lists
# index, append(), pop(), remove()
# reverse() and sort()

#   Sets
# add(), remove()


"""
    Frozen Sets and Dictionaries Course
    Linkedin Learning
    Instructor Mridu Bhatnagar
"""

# Initialize an empty Set
def Set():
    s = set()
    print("Empty set ", s)
    print("Type of object ", type(s))

def Dict():
    d = {}
    print("\nEmpty dictionary ", d)
    print("Type of object ", type(d))

def Mutables():
    lists = [1, 2, 3]
    loc = id(lists)
    print("Memory address of object before modification ", loc)
    lists.append([2, 4])
    print("Memory address of object after modification ", loc)

# LISTS, SETS AND DICTIONARIES ARE MUTABLE OBJECTS IN PYTHON

def Immutable():
    s = "Python"
    print("Memory address of object before modification ", id(s))
    # s[0] = 's'
    # print("Memory address of object after modification ", id(s))

def Iterable():
    for number in [2, 5, 6]:
        print(number)

    print("\n")
    for letter in "Python":
        print(letter)


def Hashable():
    hashed = hash(5)
    print("The hash value of the integer 5 is ", hashed)
    hashed = hash([2, 4, 5])
    print("The hash value of a list of integer is ", hashed)


if __name__ == "__main__":
    Set()
    Dict()
    Mutables()
    Immutable()
    Iterable()
    Hashable()