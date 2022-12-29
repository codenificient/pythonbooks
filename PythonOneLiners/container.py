# Containers data types are Lists, Dictionaries

# Lists
list1 = [1, 3, 5]

print(len(list1)) # returns 3

# the keyword 'is'
y = x = 3
print(x is y) # returns true because 3 lives somewhere in memory and both x and y both point to the same memory address thanks to the assignment operator '='

# ambiguous is
print([3] is [3]) # returns false because each list containing the integer '3' lives in its own memory address

# list methods - append, insert and concatenate

# append
list2 = [1, 2, 2]
list2.append(4)
print(list2)

# insert
list3 = [1, 2, 4]
list3.insert(2, 3)
print(list3)

# list concatenation
print([1,4, 7] + [9])


# big oh notations: the append method is the fastest. it does not traverse the entire list (also with insert), and it doesn't create two sublists (concatenation does)

# the extend() method allows user to append multiple elments to given list in an efficient manner

# remove method
list4 = [1, 2, 2, 4]
list4.remove(1)
print(list4)

# reverse a list
list1.reverse()
print(list1) # reverses the original list and does not create a copy

# sorting lists
list1.sort()
print(list1) # sorting also modifies the original list
