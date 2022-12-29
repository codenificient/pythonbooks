def rotate(matrix):
    if matrix is None or len(matrix)<1:
        return
    else:
        if len(matrix)==1:
            return matrix
        else:
            #solution matrix
            soln = [row[:] for row in matrix]
            #size of matrix
            m = len(matrix[0])
                    
            for x in range(0,m):
                for j in range(0,m):
                    soln[j][m-1-x] = matrix[x][j]
            return soln

if __name__=="__main__":
    six = [["a","b","c"],
          [1,2,3],
          ["x","y","z"]]
    print ("%s" % rotate(six))
