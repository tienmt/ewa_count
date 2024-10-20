library(MASS); library(glmnet)
n = 50   # sample
p = 100   # dimension
s0 = 5   # sparsity
Iters = 30000
burnin = 5000

X = matrix(runif(n*p,min = -1, max = 1) , nrow = n) ; tX = t(X)
beta0 = rep(0,p); 
beta0[1:s0] <- rnorm(s0,mean = 0,sd = 1) # runif(s0, min = -1, max = 1) #
my_mu = exp( X%*%beta0 ) ; summary(c(my_mu))
#Y = rpois(n, lambda =  my_mu) ;
Y = rnegbin(n, mu = my_mu, theta = 10) # theta = 10
XY = t(X)%*%Y

cv_lasso <- cv.glmnet(X, Y, family = "poisson",intercept = FALSE,nfolds = 5)
be_lasso <- predict(cv_lasso, newx = X, family = "poisson", type = "coef",s = "lambda.min")[-1]


### MALA
tau = .1  # in the prior
Bm_hinge = matrix( 0 ,nrow = p); a = 0  ; M = be_lasso
h = 1/(p)^3.9 
for(s in 1:Iters){
  X_m = X%*%M; eXb = exp(X_m);  my_grad =  tX%*%( eXb*(Y - eXb) )
  tam = M - h*my_grad - h*sum(4*M/(tau^2 + M^2) ) +sqrt(2*h)*rnorm(p)
  
  Xtam = X%*%tam ; eXtam = exp(Xtam)
  pro.tam = - sum( (Y - eXtam)^2 ) - sum(2*log(tau^2 + tam^2))
  pro.M = - sum( (Y - eXb)^2 ) - sum(2*log(tau^2 + M^2))
  
  tran.m = -sum((M - tam + h*tX%*%( eXtam*(Y - eXtam) ) + h*sum(2*log(tau^2 + tam^2)) )^2)/(4*h)
  tran.tam = -sum((tam - M + h*my_grad + h*sum(2*log(tau^2 + M^2)) )^2)/(4*h)
  pro.trans = pro.tam + tran.m - pro.M - tran.tam
  if(log(runif(1)) <= pro.trans){
    M = tam;   a = a+1  } ;  if (s%%5000==0){print(s) }
  if (s>burnin)Bm_hinge = Bm_hinge + M/(Iters-burnin)
} ; a/Iters

### LMC
Bm_lmc = matrix( 0 ,nrow = p); h_lmc = h/p/p
M = be_lasso
for(s in 1:Iters){
  X_m = X%*%M; eXb = exp(X_m);  my_grad =   tX%*%( eXb*(Y - eXb) )
  M = M - h_lmc*my_grad - h_lmc*sum(4*M/(tau^2 + M^2) ) +sqrt(2*h_lmc)*rnorm(p)
  if (s>burnin)Bm_lmc = Bm_lmc + M/(Iters-burnin)}

c(mean( (Bm_hinge -beta0)^2 ), mean((Y-exp(X%*%Bm_hinge))^2/mean(Y^2)) )
c(mean( (Bm_lmc -beta0)^2 ), mean((Y-exp(X%*%Bm_lmc))^2/mean(Y^2) ) ) # relative prediction errors
c(mean( (beta0 -be_lasso)^2) ,mean((Y-exp(X%*%be_lasso))^2/mean(Y^2) ) )
a/Iters
