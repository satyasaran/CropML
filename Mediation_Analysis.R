# LMG decomposition of R2 in the mediation setup
# Bo Markussen
# August 26, 2021
# load packages ----


library(readxl)
library(tidyverse)
library(nlme)


# function to compute decomposition of R2
# We adapt the decomposition of R2 proposed by:0
# Lindeman, R. H., Merenda, P. F., and Gold, R. Z. (1980), Introduction to Bivariate
# and Multivariate Analysis, Glenview, IL: Scott, Foresman, p.199ff.


LMG <- function(df,mediators,verbose=TRUE) {
  # 'df' = data frame with variables
  #   y:   response
  #   ID:  background predictor
  #   x:   spatial position
  #   bed: subject identifier for correlation
  #
  # 'mediators' = string vector with names of mediators, which are
  #               supposed to be variables inside 'df'
  # index of mediators ----
  
  if (length(mediators)==0) stop("Present implementation only works with at least 1 trait")
  
  # keep needed variables and complete cases ----
  
  df <- df %>% select(all_of(c("Corrected_13C","x","bed","ID",mediators))) %>% drop_na()
  
  
  # standardize response and mediators ----
  
  
  df$y <- scale(df$Corrected_13C)
  for (i in mediators) df[,i] <- scale(df[,i])
  
  
  # estimate beta coefficients ----
  variables <- c("ID",mediators)
  beta <- df.variables <- df[,variables]
  df <- transform(df,M=0)
  for (i in mediators) {
    df$M <- df[,i]
    beta[,i] <- predict(gls(M ~ ID, corr=corExp(form=~x|bed,nugget=TRUE),data=df))
  }
  
  # SS for ID and for mediators ---
  R2 <- rep(0,length(variables)); names(R2) <- variables
  for (i in variables) {
    if (verbose) cat(i,": ")
    S <- setdiff(variables,i)
    for (j in 0:length(S)) {
      if (verbose) cat(j,"")
      subsets <- combn(S,j)
      for (k in 1:ncol(subsets)) {
        m  <- gls(y~1,corr=corExp(form=~x|bed,nugget=TRUE),data=df)
        m0 <- update(m,as.formula(paste(c("y~1",subsets[,k]),collapse="+")))
        m1 <- update(m,as.formula(paste(c("y~1",subsets[,k],i),collapse="+")))
        
        df[,variables] <- beta
        hat.y0 <- predict(m0,newdata=df)
        hat.y1 <- predict(m1,newdata=df)
        df[,variables] <- df.variables
        
        R2[i] <- R2[i] + (factorial(j)*factorial(length(S)-j)/factorial(length(variables))) * 
          (sum((df$y-hat.y0)^2)-sum((df$y-hat.y1)^2))/sum(df$y^2)
      }
    }
    if (verbose) cat("\n")
  }
  
  # return result
  return(R2)
}


# Prepare data ----

d <- read.csv('/Users/trl102/Downloads/Spatially_Corrected_New_Mediation_df_2019.csv', header = TRUE, check.names=FALSE,fileEncoding =  "latin1", as.is=TRUE, sep=",") 


d=d[,nzchar(colnames(d))]

#d <- d %>% mutate(Growth_june=(June_Inflection_sig_point-May_Inflection_sig_point),Growth_july=(July_Inflection_sig_point-June_Inflection_sig_point))

d$ID<-as.factor(d$ID)
d$bed<-as.numeric(d$bed)
d$x<-as.numeric(d$x)
d=as_tibble(d)

# Mediation analysis for 2018 ----

d_2018<-d
d_2018$ID <- sub(" ", "_", d_2018$ID)
d_2018$ID<-as.factor(d_2018$ID)

#d_2018 <- filter(d_2018,ID %in% names(which(table(d_2018$ID)>=2)))
d_2018 <- filter(d_2018,ID %in% names(which(table(d_2018$ID)>=2)))
d_2018$ID <- droplevels(d_2018$ID)

#mediators_2018 <- c("TRD_40_BBCH_70","TRDcomb_BBCH_87","iD75_intercept_BBCH_40")
mediators_2018 <- c("May_Inflection_sig_point","June_Inflection_sig_point","July_Inflection_sig_point","RFP_Delta_13C")
#mediators_2018 <- c("Corrected_Log15N","Corrected_13C")

# hypothesis tests


df <- d_2018 %>% select(all_of(c("Corrected_Log15N","Corrected_13C","x","bed","ID",mediators_2018))) %>% drop_na()

#df <- d_2018 %>% select(all_of(c("Growth_june","Growth_july","x","bed","ID",mediators_2018))) %>% drop_na()

drop1(gls(Corrected_13C~ID,corr=corExp(form=~x|bed,nugget=TRUE),data=df,method="ML"),test="Chisq")

drop1(gls(Corrected_Log15N~ID,corr=corExp(form=~x|bed,nugget=TRUE),data=df,method="ML"),test="Chisq")

#drop1(gls(Growth_june~ID+Corrected_13C+Corrected_Log15N,corr=corExp(form=~x|bed,nugget=TRUE),data=df,method="ML"),test="Chisq")


#drop1(gls(TRD_40_BBCH_70~ID,corr=corExp(form=~x|bed,nugget=TRUE),data=df,method="ML"),test="Chisq")
drop1(gls(June_Inflection_sig_point~ID,corr=corExp(form=~x|bed,nugget=TRUE),data=df,method="ML"),test="Chisq")
#drop1(gls(TRDcomb_BBCH_87~ID,corr=corExp(form=~x|bed,nugget=TRUE),data=df,method="ML"),test="Chisq")
drop1(gls(May_Inflection_sig_point~ID,corr=corExp(form=~x|bed,nugget=TRUE),data=df,method="ML"),test="Chisq")
#drop1(gls(iD75_intercept_BBCH_40~ID,corr=corExp(form=~x|bed,nugget=TRUE),data=df,method="ML"),test="Chisq")
drop1(gls(RFP_Log_Delta_15N~ID,corr=corExp(form=~x|bed,nugget=TRUE),data=df,method="ML"),test="Chisq")
#drop1(gls(y~ID+TRD_40_BBCH_70+TRDcomb_BBCH_87+iD75_intercept_BBCH_40,corr=corExp(form=~x|bed,nugget=TRUE),data=df,method="ML"),test="Chisq")
drop1(gls(RFP_Delta_13C~ID,corr=corExp(form=~x|bed,nugget=TRUE),data=df,method="ML"),test="Chisq")

drop1(gls(Corrected_Log15N~ID+May_Inflection_sig_point+June_Inflection_sig_point+July_Inflection_sig_point+RFP_Log_Delta_15N,corr=corExp(form=~x|bed,nugget=TRUE),data=df,method="ML"),test="Chisq")
# partial R2
drop1(gls(Corrected_13C~ID+May_Inflection_sig_point+June_Inflection_sig_point+July_Inflection_sig_point+RFP_Delta_13C,corr=corExp(form=~x|bed,nugget=TRUE),data=df,method="ML"),test="Chisq")

R2_2018 <- LMG(d_2018,mediators_2018,verbose = TRUE)
                                                                                                                                                   
cat("Total R2 (year 2018)=",round(sum(R2_2018),4),"\n")

# Total R2 (year 2018)= 0.4104 # Tomke Log Delta 15N

#Total R2 (year 2018)= 0.2871 for Log Delta 15N

#Total R2 (year 2018)= 0.3419 for Delta 13C


pct_R2_2018 <- as.matrix(R2_2018/sum(R2_2018) * 100)
colnames(pct_R2_2018) <- "Percent of total"
round(pct_R2_2018,1)

#                        Percent of total
# ID                                 84.1
# TRD_40_BBCH_70                      6.2
# TRDcomb_BBCH_87                     7.9
# iD75_intercept_BBCH_40              1.8



#Percent of total Log_ Delta_15N
#ID                                    87.4
#May_Inflection_sig_point               3.7
#June_Inflection_sig_point              4.0
#July_Inflection_sig_point              4.8


#Percent of total Delta 13C
#ID                                    90.8
#May_Inflection_sig_point               2.2
#June_Inflection_sig_point               3.1
#July_Inflection_sig_point              3.9



# partial R2 for 2019 ----

d <- read.csv('/Users/trl102/Downloads/Spatially_Corrected_New_Mediation_df_2019.csv', header = TRUE, check.names=FALSE,fileEncoding =  "latin1", as.is=TRUE, sep=",") 

d=d[,nzchar(colnames(d))]
d$ID<-as.factor(d$ID)
d$bed<-as.numeric(d$bed)
d$x<-as.numeric(d$x)
d=as_tibble(d)

# Mediation analysis for 2018 ----

d_2019<-d
d_2019$ID <- sub(" ", "_", d_2019$ID)
d_2019$ID<-as.factor(d_2019$ID)

#d_2018 <- filter(d_2018,ID %in% names(which(table(d_2018$ID)>=2)))
d_2019 <- filter(d_2019,ID %in% names(which(table(d_2019$ID)>=2)))
d_2019$ID <- droplevels(d_2019$ID)

#mediators_2018 <- c("TRD_40_BBCH_70","TRDcomb_BBCH_87","iD75_intercept_BBCH_40")
mediators_2019 <- c("May_Inflection_sig_point","June_Inflection_sig_point","July_Inflection_sig_point","RFP_Log_Delta_15N")
# hypothesis tests



df <- d_2019 %>% select(all_of(c("Corrected_Log15N","Corrected_13C","x","bed","ID",mediators_2019))) %>% drop_na()

drop1(gls(May_Inflection_sig_point~ID,corr=corExp(form=~x|bed,nugget=TRUE),data=df,method="ML"),test="Chisq")

#drop1(gls(TRD_40_BBCH_70~ID,corr=corExp(form=~x|bed,nugget=TRUE),data=df,method="ML"),test="Chisq")
drop1(gls(June_Inflection_sig_point~ID,corr=corExp(form=~x|bed,nugget=TRUE),data=df,method="ML"),test="Chisq")
#drop1(gls(TRDcomb_BBCH_87~ID,corr=corExp(form=~x|bed,nugget=TRUE),data=df,method="ML"),test="Chisq")
drop1(gls(RFP_Log_Delta_15N~ID,corr=corExp(form=~x|bed,nugget=TRUE),data=df,method="ML"),test="Chisq")
#drop1(gls(iD75_intercept_BBCH_40~ID,corr=corExp(form=~x|bed,nugget=TRUE),data=df,method="ML"),test="Chisq")

#drop1(gls(y~ID+TRD_40_BBCH_70+TRDcomb_BBCH_87+iD75_intercept_BBCH_40,corr=corExp(form=~x|bed,nugget=TRUE),data=df,method="ML"),test="Chisq")

drop1(gls(Corrected_Log15N~ID+May_Inflection_sig_point+June_Inflection_sig_point+July_Inflection_sig_point+RFP_Log_Delta_15N,corr=corExp(form=~x|bed,nugget=TRUE),data=df,method="ML"),test="Chisq")
# partial R2
# partial R2

R2_2019 <- LMG(d_2019,mediators_2019,verbose = TRUE)

cat("Total R2 (year 2019)=",round(sum(R2_2019),4),"\n")

# Total R2 (year 2019)= 0.4863 # Tomke paper Log_Delta_15N

#Total R2 (year 2019)= 0.4774 for Log_Delta_15N
#Total R2 (year 2019)= 0.7335 # Delta 13C



pct_R2_2019 <- as.matrix(R2_2019/sum(R2_2019) * 100)
colnames(pct_R2_2019) <- "Percent of total"
round(pct_R2_2019,1)

#                     Percent of total
# ID                              98.2
# ssqrt_TRDcomb_BBCH_70            1.8



#Percent of total :   Log_Delta_15N
#ID                                    92.4
#May_Inflection_sig_point               1.6
#June_Inflection_sig_point              1.5
#July_Inflection_sig_point              4.6


#Percent of total  : Delta_13C
#ID                                    97.9
#May_Inflection_sig_point               0.3
#June_Inflection_sig_point              0.5
#July_Inflection_sig_point              1.3






