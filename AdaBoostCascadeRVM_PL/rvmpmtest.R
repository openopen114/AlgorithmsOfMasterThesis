rvmpm <- function(df.x,df.t){
  
  #df.t needs 1,0 type
  df.t[df.t==-1]  <- 0
  
  
  N <- nrow(df.x)
  
  
  
  
  ###########################
  #########   \Phi    #######
  ###########################   N by N + 1
  #kernel function
  K <- function(u,v){
    #val <- sum(u*v) # linear <u,v>
    c <- (u - v)
    val <- exp(-5 * sum(c * c) )
    val
  }
  
  
  Phi <- rep(1, N)  # Phi[ ,1]
  for (i in 1:N){   # Phi[ ,2:N+1] 
    vi  <- df.x[i, ]
    Phi <- cbind(Phi, apply(df.x, 1, K, v = vi))
  }
  
  
  
  # set up parameters and hyperparameters
  M <- N + 1
  w <- matrix(0,M,1)   # w = [w0,w1,...,wN] ; N+1 by 1
  alpha <- (1/N)   # initial alpha ; 1 by 1
  alpha <- matrix(alpha,M,1) # alpha vector ; N+1 by 1
  #alpha <- 1 / sample(300,M,replace = TRUE)
  #gamma <- matrix(1,M,1)
  
  nIT <- 1000 # num of iterations
  alpha.MAX <- 1e10
  
  index.all <- 1:N
  index.rv <- c()
  
  for (i in 1:nIT){
    # nIT = 1,000
    # prune alpha which large enough value
    # ie. stay ( alpha vlaue < 1e9(alpha.Max) )
    
    stay.order <- (alpha < alpha.MAX) 
    alpha <- alpha[stay.order]
    M <- sum(stay.order)
    
    # relevance vector index
    index.rv <- c(index.rv, index.all[!stay.order[-1]])
    index.all <- index.all[stay.order[-1]]
    
    
    
    w[!stay.order] <- 0
    w <- w[stay.order]
    Phi <- Phi[ stay.order[-1], stay.order]
    df.t <- df.t[ stay.order[-1] ]
    
     
    # update w
    result.w.Uinv <- rvm.w.MP(Phi, df.t, w, alpha, nIT.pm = 25)
    w <- result.w.Uinv$w.new
    Uinv <- result.w.Uinv$Uinv
    diagSig <- apply(Uinv * Uinv, 1, sum)
    gamma <- 1 - alpha * diagSig
    
    # update alpha
    alpha <- gamma / (w * w)
    
  }
  
  
  index.rv
  
  
}