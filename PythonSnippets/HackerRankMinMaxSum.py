def miniMaxSum(arr):
    arr = sorted(arr)
    print("{0} {1}".format(sum(arr[:4]), sum(arr[1:])))

if __name__ == "__main__":
    miniMaxSum([1,3,4,5])