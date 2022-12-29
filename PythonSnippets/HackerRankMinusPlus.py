def plusMinus(arr):
    # Write your code here
    pos = 0
    neg = 0
    zer = 0
    size = len(arr)
    for num in arr:
        if num < 0:
            neg += 1
        elif num > 0:
            pos += 1
        else:
            zer += 1
    print("{:.6f}".format(pos/size))
    print("{:.6f}".format(neg/size))
    print("{:.6f}".format(zer/size))