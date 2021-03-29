library(dirichletprocess)
library(MASS)
set.seed(100)

#Normalize galaxy data
y<- galaxies/1000

#Carry out MCMC
n_samp = 4000 #3000 for timing
dp_galaxy <- DirichletProcessGaussian(y,g0Priors = c(20,0.07,2,1)) #g0Priors = c(20,0.01,2,1); 0.07 gives the right amount of modes
system.time(dp_galaxy <- Fit(dp_galaxy, n_samp))

#Initialize y_plot
y_plot = seq(5,40,length.out = 200)
dy = y_plot[2]- y_plot[1]

#Compute posterior samples of pdf
ind = 1:n_samp
system.time(pdf_samp_galaxy <- t(sapply(ind, function(i) PosteriorFunction(dp_galaxy,i)(y_plot))))
#discard burn-in
pdf_samp_galaxy <- pdf_samp_galaxy[2001:n_samp,]

#Compute posterior mean and quantiles of pdf samples
mean_pdf = colMeans(pdf_samp_galaxy)
bot25_pdf = apply(pdf_samp_galaxy, 2, function(x) quantile(x,probs = 0.025))
top25_pdf = apply(pdf_samp_galaxy, 2, function(x) quantile(x,probs = 0.975))
plot(y_plot,top25_pdf,type = 'l',ylim = c(-0.017714758231901252, 0.33)) + lines(y_plot, bot25_pdf)+ lines(y_plot, mean_pdf)

#Compute posterior mean and quantiles of cdf samples
cdf_samp_galaxy = t(apply(pdf_samp_galaxy,1,cumsum)*dy)
mean_cdf = colMeans(cdf_samp_galaxy)
bot25_cdf = apply(cdf_samp_galaxy, 2, function(x) quantile(x,probs = 0.025))
top25_cdf = apply(cdf_samp_galaxy, 2, function(x) quantile(x,probs = 0.975))

plot(y_plot,top25_cdf,type = 'l') +  lines(y_plot, bot25_cdf)+ lines(y_plot, mean_cdf)

#Save mean and cred interval to csv to plot in python
galaxy_plot = data.frame(y_plot = y_plot, mean_pdf = mean_pdf,top25_pdf = top25_pdf,
                        bot25_pdf = bot25_pdf, mean_cdf = mean_cdf, 
                        top25_cdf = top25_cdf, bot25_cdf = bot25_cdf)

galaxy_pdf_samples = data.frame(pdf_samp_galaxy)
galaxy_cdf_samples = data.frame(cdf_samp_galaxy)

write.csv(galaxy_plot,"plot_files/dpmm_galaxy_plot.csv")
write.csv(galaxy_pdf_samples,"plot_files/dpmm_galaxy_pdf_samples.csv")
write.csv(galaxy_cdf_samples,"plot_files/dpmm_galaxy_cdf_samples.csv")
