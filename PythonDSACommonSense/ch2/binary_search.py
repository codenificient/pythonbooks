def binary_search(array, value):
    lower_bound = 0
    upper_bound = len(array) - 1
    while lower_bound <= upper_bound:
        midpoint = int((upper_bound + lower_bound) / 2)
        midvalue = array[midpoint]

        if value == midvalue:
            return midpoint
        elif value < midvalue:
            upper_bound = midpoint - 1
        elif value > midvalue:
            lower_bound = midpoint + 1

if __name__ == '__main__':
    print(binary_search([4, 3, 2, 10], 2))
    print(binary_search([4, 3, 2, 10], 5))