library(dirichletprocess)

# Load Ozone dataset and normalize
data(airquality)
attach(airquality)
set.seed(100)
ozone <- Ozone**(1/3)
radiation <- Solar.R
y = na.omit(cbind(ozone,radiation))
y = scale(y)
d = dim(y)[2]

# Sample with MCMC
n_samp = 4000 #3000 for timing
g0Priors <- list(mu0 = rep_len(0, length.out = ncol(y)), 
                 Lambda = diag(ncol(y)), kappa0 = ncol(y), nu = ncol(y))
dp_ozone <- DirichletProcessMvnormal(y,g0Priors = g0Priors)
system.time(dp_ozone <- Fit(dp_ozone, n_samp))

#Initialize y_plot
y_grid1 = seq(-2.75,2.75,length.out = 25)
y_grid2 = seq(-2.5,2.25,length.out = 25)
y_plot = expand.grid(y_grid1,y_grid2)

n_plot = dim(y_plot)[1]

#Compute posterior samples of pdf
ind = 1:n_samp
system.time(pdf_samp_ozone <- t(sapply(ind, function(i) PosteriorFunction(dp_ozone,i)(y_plot))))
#discard burn-in
pdf_samp_ozone = pdf_samp_ozone[2001:n_samp,]

#Compute posterior mean and quantiles of pdf samples
mean_pdf = colMeans(pdf_samp_ozone)
bot25_pdf = apply(pdf_samp_ozone, 2, function(x) quantile(x,probs = 0.025))
top25_pdf = apply(pdf_samp_ozone, 2, function(x) quantile(x,probs = 0.975))
std_pdf = apply(pdf_samp_ozone, 2, sd)


#Save mean and cred interval to csv to plot in python
ozone_plot = data.frame(y_plot1 = y_plot[,1],y_plot2 = y_plot[,2], 
                        mean_pdf = mean_pdf,std_pdf = std_pdf,
                        bot25_pdf = bot25_pdf, top25_pdf = top25_pdf)
write.csv(ozone_plot,"plot_files/dpmm_ozone_plot.csv")
