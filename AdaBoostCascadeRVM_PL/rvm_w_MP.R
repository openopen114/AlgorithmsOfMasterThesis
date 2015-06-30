rvm.w.MP <- function(Phi, df.t, w, alpha, nIT.pm){
    
    
    
    A <- diag(alpha)
    Y.xw <- Phi  %*% w  # yi =  y(xi;w) ;[]
    
    # sigmoid function
    sigmoid.f <- function(x){
         p.x <- 1 / ( 1 + exp(-x) )
         p.x
    }
    
    Y <- apply(Y.xw, 1, sigmoid.f) # yi = sigmoid{  y(xi;w) } ;[]
   
    
    # function for compute B diag
    B.f <- function(y){
        b <- y * (1 - y)
        b
    }
    
    
    for (j in 1:nIT.pm){             
        B <- diag( apply(matrix(Y,ncol=1), 1, B.f) )
        
        g <- ( t(Phi) %*% ( df.t  - Y) ) - ( A %*% w )
        Hession <- t(Phi) %*% B %*% Phi  + A
        U <- chol(Hession)  #Hession = U'U ; cholesky decomp.
        Uinv <- solve(U) #inverse of matrix U
        delta.w <- Uinv %*% t(Uinv) %*% g
        
        w.new <- w  + delta.w
        
        if ( j < nIT.pm ){
            Y.xw.new <- Phi  %*% w.new 
            Y <- apply(Y.xw.new, 1, sigmoid.f)    
        }
        w  <- w.new
              
    }
    
    output.list <- list(w.new = w.new,
                        Uinv = Uinv)
   
   
    
}