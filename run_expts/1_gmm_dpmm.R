      library(dirichletprocess)
      
      set.seed(100)
      for (n in c(50,200)){
        #Load toy dataset
        y <- data.matrix(read.csv(paste("R_data/gmm_n",n,".csv",sep = ""),header = FALSE))
    
        #Elicit prior and carry out MCMC
        n_samp = 4000
        k = 1
        system.time(dp_gmm <- DirichletProcessGaussian(y,g0Priors = c(0,k,0.1,0.1))) #g0Priors = c(0,1,0.1,0.1)
        system.time(dp_gmm <- Fit(dp_gmm, n_samp))
        
        #Initialize y_plot
        y_plot = seq(-4,4,length.out = 160)
        dy = y_plot[2] - y_plot[1]
        
        #Compute posterior samples of pdf
        ind = 1:n_samp
        system.time(pdf_samp_gmm <- t(sapply(ind, function(i) PosteriorFunction(dp_gmm,i)(y_plot))))
        #discard burn-in
        pdf_samp_gmm <- pdf_samp_gmm[2001:n_samp,]
        
        #Compute posterior mean and quantiles of pdf samples
        mean_pdf = colMeans(pdf_samp_gmm)
        bot25_pdf = apply(pdf_samp_gmm, 2, function(x) quantile(x,probs = 0.025))
        top25_pdf = apply(pdf_samp_gmm, 2, function(x) quantile(x,probs = 0.975))
        plot(y_plot,top25_pdf,type = 'l',ylim = c(-0.036482783524078535, 0.6562048037547857)) + 
          lines(y_plot, bot25_pdf)+ lines(y_plot, mean_pdf)
        
        #Compute posterior mean and quantiles of cdf samples
        cdf_samp_gmm = t(apply(pdf_samp_gmm,1,cumsum)*dy)
        mean_cdf = colMeans(cdf_samp_gmm)
        bot25_cdf = apply(cdf_samp_gmm, 2, function(x) quantile(x,probs = 0.025))
        top25_cdf = apply(cdf_samp_gmm, 2, function(x) quantile(x,probs = 0.975))
          
        plot(y_plot,top25_cdf,type = 'l') +  lines(y_plot, bot25_cdf)+ lines(y_plot, mean_cdf)
      
        #save mean and cred interval to csv to plot in python
        gmm_plot = data.frame(y_plot = y_plot, mean_pdf = mean_pdf,top25_pdf = top25_pdf,
                               bot25_pdf = bot25_pdf,mean_cdf = mean_cdf,top25_cdf = top25_cdf,
                                bot25_cdf = bot25_cdf)
        write.csv(gmm_plot,paste("plot_files/dpmm_gmm_plot_n",n,".csv",sep = ""))
      }
