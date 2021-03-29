library(dirichletprocess)
# Load data
x <- data.matrix(read.csv("R_data/lidar_x.csv",header = FALSE))
y <- data.matrix(read.csv("R_data/lidar_y.csv",header = FALSE))
z = cbind(x,y)
d = dim(z)[2]
set.seed(100)

# Sample with MCMC
n_samp = 4000 #3000 for timing
g0Priors <- list(mu0 = rep_len(0, length.out = ncol(z)), 
                 Lambda = diag(ncol(z)), kappa0 = ncol(z), nu = ncol(z))
dp_lidar <- DirichletProcessMvnormal(z,g0Priors = g0Priors)
system.time(dp_lidar <- Fit(dp_lidar, n_samp))

#Initialize y_plot
n_plot = 200
y_grid = seq(-3,2,length.out = n_plot)
dy = y_grid[2] - y_grid[1]

#SINGLE X
for (x_grid in c(0,-3)){
z_plot = cbind(x_grid,y_grid)

ind = 1:n_samp
system.time(pdf_samp_joint <- t(sapply(ind, function(i) PosteriorFunction(dp_lidar,i)(z_plot))))

#normalize each pdf_samp_joint to get p(y|x)
pdf_samp_joint = pdf_samp_joint[2001:n_samp,]
pdf_samp_condit = pdf_samp_joint/(rowSums(pdf_samp_joint)*dy)

#plot cred intervals
mean_pdf = colMeans(pdf_samp_condit)
bot25_pdf = apply(pdf_samp_condit, 2, function(x) quantile(x,probs = 0.025))
top25_pdf = apply(pdf_samp_condit, 2, function(x) quantile(x,probs = 0.975))
plot(y_grid,top25_pdf,type = 'l') + lines(y_grid, bot25_pdf)+ lines(y_grid, mean_pdf) 


lidar_plot = data.frame(y_plot = y_grid,x_plot = x_grid, mean_pdf = colMeans(pdf_samp_condit),top25_pdf = top25_pdf,
                        bot25_pdf = bot25_pdf)
write.csv(lidar_plot,paste("plot_files/dpmm_lidar_plot",x_grid,".csv",sep = ''))}

#plot against x
#MULTIPLE X
x_grid = seq(min(x),max(x),length.out = 40)
dx = x_grid[2] - x_grid[1]
z_plot = expand.grid(x_grid,y_grid)
pdf_plot = matrix(PosteriorFrame(dp_lidar,z_plot)$Mean,length(x_grid),length(y_grid))
contour(x_grid,y_grid,pdf_plot)

#get condit pdfs
pdf_x = rowSums(pdf_plot*dy)
pdf_yx = pdf_plot /pdf_x

#get condit cdfs
cdf_yx =apply(pdf_yx,1,cumsum)*dy
plot(y_grid,cdf_yx[,1],type ='l')

#compute medians
n_x = length(x_grid)
mean = rep(NA,n_x)
bot25 = rep(NA,n_x)
top25 = rep(NA,n_x)
for (i in 1:n_x){
  mean[i] = sum(pdf_yx[i,]*y_grid*dy)
  bot25[i] = y_grid[which.min(abs(cdf_yx[,i]- 0.025))]
  top25[i] = y_grid[which.min(abs(cdf_yx[,i]- 0.975))]
}

#plot
plot(x,y,ylim = c(-2.5,1.7))+ lines(x_grid,mean,type = 'l') + lines(x_grid, bot25) + lines(x_grid, top25) 

lidar_plot_xrange = data.frame(x_plot = x_grid, mean = mean,top25 = top25,
                        bot25 = bot25)

write.csv(lidar_plot_xrange,"plot_files/dpmm_lidar_plot_xrange.csv")
          