def linear_search(array, value):
    for elem in range(len(array)):
        if array[elem] == value:
            return elem
    return -1

if __name__ == '__main__':
    print(linear_search([4,3,2,10], 2))
    print(linear_search([4,3,2,10], 5))