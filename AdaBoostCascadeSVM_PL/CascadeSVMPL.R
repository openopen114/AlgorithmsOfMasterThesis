
CascadeSVMPL <- function(Dataw.x, Data.y, cost.value){

    #Layer 1
    m <- nrow(Dataw.x) / 8
    foreach(i=1:8) %dopar% {
        svmdata.x <- as.data.frame(Dataw.x[(m * ( i - 1) + 1) : (m * i),])
        svmdata.y <- as.factor(Data.y[(m * ( i - 1) + 1) : (m * i)])
        trainset <- data.frame(svmdata.x, svmdata.y)
        svm.model <- svm(svmdata.y ~ ., data = trainset, cost = cost.value,
                         kernel ="radial", gamma = 5)
        
        #save SV
         svm.sv.index<- svm.model$index + m * (i - 1)
         svm.sv.index
    } -> L1
    
    
    #Layer 2
    foreach(i=1:4) %dopar%{
        #combind sv
        sv.index.1 <- L1[[(2*i)-1]]
        sv.index.2 <- L1[[(2*i)]]
        sv.matrix <- rbind(Dataw.x[sv.index.1, ], Dataw.x[sv.index.2, ])
        
        svmdata.x <- as.data.frame(sv.matrix)
        svmdata.y <- as.factor(Data.y[c(sv.index.1, sv.index.2)])
        trainset <- data.frame(svmdata.x, svmdata.y)
        svm.model <- svm(svmdata.y ~ ., data = trainset, cost = cost.value,
                         kernel ="radial", gamma = 5)
        
        
        
        #save SV
        adjust.index.L2 <- function(x){
            if( x <= length(L1[[(2*i)-1]])  )
                x <- L1[[(2*i)-1]][ x ]
            else
                x <- L1[[(2*i)]][ x - length(L1[[(2*i)-1]]) ]
        }
        
        svm.sv.index <- apply(matrix(svm.model$index),
                              1, adjust.index.L2)
        svm.sv.index
        
    } -> L2
    
    
    #Layer 3
    foreach(i=1:2) %dopar%{
        #combind sv
        sv.index.1 <- as.numeric(L2[[(2*i)-1]])
        sv.index.2 <- as.numeric(L2[[(2*i)]])
        sv.matrix <- rbind(Dataw.x[sv.index.1, ], Dataw.x[sv.index.2, ])
        
        svmdata.x <- as.data.frame(sv.matrix)
        svmdata.y <- as.factor(Data.y[c(sv.index.1, sv.index.2)])
        trainset <- data.frame(svmdata.x, svmdata.y)
        svm.model <- svm(svmdata.y ~ ., data = trainset, cost = cost.value,
                         kernel ="radial", gamma = 5)
        
        #save SV
        adjust.index.L3 <- function(x){
            if( x <= length(as.numeric(L2[[(2*i)-1]])) )
                x <- as.numeric(L2[[(2*i)-1]])[ x ]
            else
                x <- as.numeric(L2[[(2*i)]])[ x - length(as.numeric(L2[[(2*i)-1]])) ]
        }
        
        svm.sv.index <- apply(matrix(svm.model$index),
                              1, adjust.index.L3)
        svm.sv.index
        
        
    } -> L3
    
    #Layer 4
    #combind sv
    sv.index.1 <- as.numeric(L3[[1]])
    sv.index.2 <- as.numeric(L3[[2]])
    sv.matrix <- rbind(Dataw.x[sv.index.1, ], Dataw.x[sv.index.2, ])
    
    svmdata.x <- as.data.frame(sv.matrix)
    svmdata.y <- as.factor(Data.y[c(sv.index.1, sv.index.2)])
    trainset <- data.frame(svmdata.x, svmdata.y)
    svm.model <- svm(svmdata.y ~ ., data = trainset, cost = cost.value,
                     kernel ="radial", gamma = 5)
    
    
    #output list svm.model, pred_result, accuracy
    dataDF.x<- as.data.frame(Dataw.x)
    dataFC.y<- as.factor(Data.y)
    pred_result <- predict(svm.model, dataDF.x)
    table(pred_result, dataFC.y) %>%
        acc() -> accuracy
    
    
    output.list  <- list(svm.model = svm.model,
                         pred_result = pred_result,
                         accuracy = accuracy)

}