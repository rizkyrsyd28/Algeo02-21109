myQR = function(A){
  dimU = dim(A)
  U = matrix(nrow = dimU[1], ncol = dimU[2])
  U[,1] = A[,1]
  for(k in 2:dimU[2]){
    subt = 0
    j = 1
    while(j < k){
      subt = subt + proj(U[,j], A[,k])
      j = j + 1
    }
    U[,k] = A[,k] - subt
  }
  Q = apply(U, 2, function(x) x/sqrt(sum(x^2)))
  R = round(t(Q) %*% A, 10)
  return(Q)
}

A <- matrix(
  c(1,2,3,1,4,3,3,2,10),
  nrow = 3,  
  ncol = 3,        
)

print(myQR(A))
