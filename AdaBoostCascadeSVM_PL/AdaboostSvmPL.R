load("~/Documents/Adobe/Rcodesd/MNIST.RData")
library(foreach) # parallel for loop
library(doMC) # multicore parallel
library(e1071) #for svm
library(dplyr)

#source("~/Documents/Rcode/Rcode/load_mnist.R")
#source("~/Documents/Rcode/Rcode/CascadeSVMPL.R")

#load_mnist() #load MNIST data using load_mnist.R



cores <- 4
registerDoMC(cores)



#convert y to even(-1) odd(1) lable
convert.y <- 2 * (train$y[1:60000] %% 2) - 1
#split training dataset to A.x, A.y, B.x, B.y ... 
MachineNode <- c("A", "B", "C", "D", "E")
for ( i in 1:length(MachineNode) ) {
    var_name_x <- paste(MachineNode[i], ".x", sep="")
    m <- nrow(train$x) / 5
    train.x.split <- train$x[(m * ( i - 1) + 1) : (m * i),]
    assign(var_name_x, train.x.split)
    
    var_name_y <- paste(MachineNode[i], ".y", sep="")
    train.y.split <- convert.y[(m * ( i - 1) + 1) : (m * i)]
    assign(var_name_y, train.y.split) 
}

#accuracy function
acc <- function(M){
    acc <- (M[1,1] + M[2,2]) / sum(M)
    acc
}

cost.ini <- 50
adjust.sv = T



timer1 <- proc.time()
#Cascade SVM
###########################  A  ###########################
#weight
wA <- rep(1 / nrow(A.x), nrow(A.x))
cost.A <- cost.ini
for (T in 1:5){
    #weight data
    Aw.x <- wA * A.x
    
    #Cascade svm save result as svm.A.1 ...
    svmmodel_name <- paste("svm.A.", T, sep="")
    svmmodel.final.A<- CascadeSVMPL(Aw.x, A.y, cost.A)
    assign(svmmodel_name, svmmodel.final.A)
    
    #epsilon save result as esl.A.1 ...
    a <- as.factor(A.y)
    b <- svmmodel.final.A$pred_result
    x <- as.numeric(a) - as.numeric(b)
    index.incorrect <- which( as.matrix(x) %in% c(1,-1)) 
    epsilon<- sum(wA[index.incorrect]) / sum(wA)
    
    epsilon_name <- paste("esl.A.", T, sep="")
    assign(epsilon_name, epsilon)
    
    # alpha sava result as alp.A.1...
    alpha <- 0.5 * log((1 - epsilon) / epsilon)  
    
    alpha_name <- paste("alp.A.", T, sep="")
    assign(alpha_name, alpha)
    
    #adjus wA
    #|_|t
    scaling.factor.t <- sqrt((1 - epsilon) / epsilon) 
    for (i in 1:length(wA)){
        if ( is.na(match(i, index.incorrect)) )
            # correct point
            wA[i] <- wA[i] / scaling.factor.t
        else
            # incorrect point
            wA[i] <- wA[i] * scaling.factor.t 
    }    
    
    #adjust cost value
    if (adjust.sv){
        var_name <- paste("svm.A.", T,"$svm$tot.nSV",sep="")
        cost.A <- cost.A * eval(parse(text=var_name)) / 1200
    }
    
}



###########################  B  ###########################
#weight
wB <- rep(1 / nrow(B.x), nrow(B.x))
cost.B <- cost.ini
for (T in 1:5){
    #weight data
    Bw.x <- wB * B.x
    
    #Cascade svm save result as svm.B.1 ...
    svmmodel_name <- paste("svm.B.", T, sep="")
    svmmodel.final.B<- CascadeSVMPL(Bw.x, B.y, cost.B)
    assign(svmmodel_name, svmmodel.final.B)
    
    #epsilon save result as esl.B.1 ...
    a <- as.factor(B.y)
    b <- svmmodel.final.B$pred_result
    x <- as.numeric(a) - as.numeric(b)
    index.incorrect <- which( as.matrix(x) %in% c(1,-1)) 
    epsilon<- sum(wB[index.incorrect]) / sum(wB)
    
    epsilon_name <- paste("esl.B.", T, sep="")
    assign(epsilon_name, epsilon)
    
    # alpha sava result as alp.B.1...
    alpha <- 0.5 * log((1 - epsilon) / epsilon)  
    
    alpha_name <- paste("alp.B.", T, sep="")
    assign(alpha_name, alpha)
    
    #adjus wB
    #|_|t
    scaling.factor.t <- sqrt((1 - epsilon) / epsilon) 
    for (i in 1:length(wB)){
        if ( is.na(match(i, index.incorrect)) )
            # correct point
            wB[i] <- wB[i] / scaling.factor.t
        else
            # incorrect point
            wB[i] <- wB[i] * scaling.factor.t 
    }    
    
    #adjust cost value
    if (adjust.sv){
        var_name <- paste("svm.B.", T,"$svm$tot.nSV",sep="")
        cost.B <- cost.B * eval(parse(text=var_name)) / 1200
    }
    
}




###########################  C  ###########################
#weight
wC <- rep(1 / nrow(C.x), nrow(C.x))
cost.C <- cost.ini
for (T in 1:5){
    #weight data
    Cw.x <- wC * C.x
    
    #Cascade svm save result as svm.C.1 ...
    svmmodel_name <- paste("svm.C.", T, sep="")
    svmmodel.final.C<- CascadeSVMPL(Cw.x, C.y, cost.C)
    assign(svmmodel_name, svmmodel.final.C)
    
    #epsilon save result as esl.C.1 ...
    a <- as.factor(C.y)
    b <- svmmodel.final.C$pred_result
    x <- as.numeric(a) - as.numeric(b)
    index.incorrect <- which( as.matrix(x) %in% c(1,-1)) 
    epsilon<- sum(wC[index.incorrect]) / sum(wC)
    
    epsilon_name <- paste("esl.C.", T, sep="")
    assign(epsilon_name, epsilon)
    
    # alpha sava result as alp.C.1...
    alpha <- 0.5 * log((1 - epsilon) / epsilon)  
    
    alpha_name <- paste("alp.C.", T, sep="")
    assign(alpha_name, alpha)
    
    #adjus wC
    #|_|t
    scaling.factor.t <- sqrt((1 - epsilon) / epsilon) 
    for (i in 1:length(wC)){
        if ( is.na(match(i, index.incorrect)) )
            # correct point
            wC[i] <- wC[i] / scaling.factor.t
        else
            # incorrect point
            wC[i] <- wC[i] * scaling.factor.t 
    }    
    
    #adjust cost value
    if (adjust.sv){
        var_name <- paste("svm.C.", T,"$svm$tot.nSV",sep="")
        cost.C <- cost.C * eval(parse(text=var_name)) / 1200
    }
    
}


###########################  D  ###########################
#weight
wD <- rep(1 / nrow(D.x), nrow(D.x))
cost.D <- cost.ini
for (T in 1:5){
    #weight data
    Dw.x <- wD * D.x
    
    #Cascade svm save result as svm.D.1 ...
    svmmodel_name <- paste("svm.D.", T, sep="")
    svmmodel.final.D<- CascadeSVMPL(Dw.x, D.y, cost.D)
    assign(svmmodel_name, svmmodel.final.D)
    
    #epsilon save result as esl.D.1 ...
    a <- as.factor(D.y)
    b <- svmmodel.final.D$pred_result
    x <- as.numeric(a) - as.numeric(b)
    index.incorrect <- which( as.matrix(x) %in% c(1,-1)) 
    epsilon<- sum(wD[index.incorrect]) / sum(wD)
    
    epsilon_name <- paste("esl.D.", T, sep="")
    assign(epsilon_name, epsilon)
    
    # alpha sava result as alp.C.1...
    alpha <- 0.5 * log((1 - epsilon) / epsilon)  
    
    alpha_name <- paste("alp.D.", T, sep="")
    assign(alpha_name, alpha)
    
    #adjus wD
    #|_|t
    scaling.factor.t <- sqrt((1 - epsilon) / epsilon) 
    for (i in 1:length(wD)){
        if ( is.na(match(i, index.incorrect)) )
            # correct point
            wD[i] <- wD[i] / scaling.factor.t
        else
            # incorrect point
            wD[i] <- wD[i] * scaling.factor.t 
    }    
    
    #adjust cost value
    if (adjust.sv){
        var_name <- paste("svm.D.", T,"$svm$tot.nSV",sep="")
        cost.D <- cost.D * eval(parse(text=var_name)) / 1200
    }
    
}



###########################  E  ###########################
#weight
wE <- rep(1 / nrow(E.x), nrow(E.x))
cost.E <- cost.ini
for (T in 1:5){
    #weight data
    Ew.x <- wE * E.x
    
    #Cascade svm save result as svm.E.1 ...
    svmmodel_name <- paste("svm.E.", T, sep="")
    svmmodel.final.E<- CascadeSVMPL(Ew.x, E.y, cost.E)
    assign(svmmodel_name, svmmodel.final.E)
    
    #epsilon save result as esl.E.1 ...
    a <- as.factor(E.y)
    b <- svmmodel.final.E$pred_result
    x <- as.numeric(a) - as.numeric(b)
    index.incorrect <- which( as.matrix(x) %in% c(1,-1)) 
    epsilon<- sum(wE[index.incorrect]) / sum(wE)
    
    epsilon_name <- paste("esl.E.", T, sep="")
    assign(epsilon_name, epsilon)
    
    # alpha sava result as alp.E.1...
    alpha <- 0.5 * log((1 - epsilon) / epsilon)  
    
    alpha_name <- paste("alp.E.", T, sep="")
    assign(alpha_name, alpha)
    
    #adjus wE
    #|_|t
    scaling.factor.t <- sqrt((1 - epsilon) / epsilon) 
    for (i in 1:length(wE)){
        if ( is.na(match(i, index.incorrect)) )
            # correct point
            wE[i] <- wE[i] / scaling.factor.t
        else
            # incorrect point
            wE[i] <- wE[i] * scaling.factor.t 
    }    
    
    #adjust cost value
    if (adjust.sv){
        var_name <- paste("svm.E.", T,"$svm$tot.nSV",sep="")
        cost.E <- cost.E * eval(parse(text=var_name)) / 1200
    }
    
}

t1 <- proc.time() - timer1 


timer2 <- proc.time()
######## H stat & alpha star  ######## 

alpha.A <- c(alp.A.1, alp.A.2, alp.A.3, alp.A.4, alp.A.5)
alpha.A.order <- order(alpha.A)

alpha.B <- c(alp.B.1, alp.B.2, alp.B.3, alp.B.4, alp.B.5)
alpha.B.order <- order(alpha.B)

alpha.C <- c(alp.C.1, alp.C.2, alp.C.3, alp.C.4, alp.C.5)
alpha.C.order <- order(alpha.C)

alpha.D <- c(alp.D.1, alp.D.2, alp.D.3, alp.D.4, alp.D.5)
alpha.D.order <- order(alpha.D)

alpha.E <- c(alp.E.1, alp.E.2, alp.E.3, alp.E.4, alp.E.5)
alpha.E.order <- order(alpha.E)


# H star 
for (i in 1:5){
    H.star_name <- paste("H.A.star.", i, sep="")
    wrt.svm.model <- eval(parse(text=paste("svm.A.", alpha.A.order[i],
                                           "$svm.model", sep="")))
    assign(H.star_name, wrt.svm.model)
    
    H.star_name <- paste("H.B.star.", i, sep="")
    wrt.svm.model <- eval(parse(text=paste("svm.B.", alpha.B.order[i],
                                           "$svm.model", sep="")))
    assign(H.star_name, wrt.svm.model)
    
    H.star_name <- paste("H.C.star.", i, sep="")
    wrt.svm.model <- eval(parse(text=paste("svm.C.", alpha.C.order[i],
                                           "$svm.model", sep="")))
    assign(H.star_name, wrt.svm.model)
    
    H.star_name <- paste("H.D.star.", i, sep="")
    wrt.svm.model <- eval(parse(text=paste("svm.D.", alpha.D.order[i],
                                           "$svm.model", sep="")))
    assign(H.star_name, wrt.svm.model)
    
    H.star_name <- paste("H.E.star.", i, sep="")
    wrt.svm.model <- eval(parse(text=paste("svm.E.", alpha.E.order[i],
                                           "$svm.model", sep="")))
    assign(H.star_name, wrt.svm.model)
}

# alpha star
for (i in 1:5){
    alpha.star_name <- paste("alpha.star.", i, sep="")
    avg.alpha <- mean(c(alpha.A[alpha.A.order[i]],
                        alpha.B[alpha.B.order[i]],
                        alpha.C[alpha.C.order[i]],
                        alpha.D[alpha.D.order[i]],
                        alpha.E[alpha.E.order[i]])
    )
    assign(alpha.star_name, avg.alpha)
}


# Testing data result
pred.num <- function(x){
    x <- as.numeric(as.matrix(x))
    x
}

F <- function(x){
    x <- x * (1 / 12000)
    T1 <- foreach(i=1:5, .combine=cbind) %dopar%{
               var_name <- paste("H.", MachineNode[i],".star.1", sep="")
               pred.num(predict(eval(parse(text=var_name)), x))
    }
    
    T2 <- foreach(i=1:5, .combine=cbind) %dopar%{
        var_name <- paste("H.", MachineNode[i],".star.2", sep="")
        pred.num(predict(eval(parse(text=var_name)), x))
    }
    
    T3 <- foreach(i=1:5, .combine=cbind) %dopar%{
        var_name <- paste("H.", MachineNode[i],".star.3", sep="")
        pred.num(predict(eval(parse(text=var_name)), x))
    }
    
    T4 <- foreach(i=1:5, .combine=cbind) %dopar%{
        var_name <- paste("H.", MachineNode[i],".star.4", sep="")
        pred.num(predict(eval(parse(text=var_name)), x))
    }
    
    T5 <- foreach(i=1:5, .combine=cbind) %dopar%{
        var_name <- paste("H.", MachineNode[i],".star.5", sep="")
        pred.num(predict(eval(parse(text=var_name)), x))
    }
    
    all.alpha.T<- cbind(alpha.star.1 * T1,
                        alpha.star.2 * T2,
                        alpha.star.3 * T3,
                        alpha.star.4 * T4,
                        alpha.star.5 * T5
    )
    
    output.F<- sign(apply(all.alpha.T, 1, sum))
    
    output.F   
}

#pred
pred.test.y <- F(test$x)


#convert y to even(-1) odd(1) lable
convert.test.y <- 2 * (test$y[1:10000] %% 2) - 1

table(as.factor(pred.test.y), as.factor(convert.test.y)) %>%
    acc() -> test.acc

t2 <- proc.time() - timer2

test.acc
t1
t2
#save.image(file = "cost50F.RData")
msg <- paste("t1 = ", t1[[3]],"t2 = ", t2[[3]], "test.acc =", test.acc )

#msg2 <- paste("t1 = 10327.019 ","t2 = 806.862", "test.acc =  0.9126" )
save(msg,file = "50T.RDATA")
#library(mail)
#sendmail("openopen114@gmail.com", "Rdata < C = 10, SV = T  > ", message = msg)
