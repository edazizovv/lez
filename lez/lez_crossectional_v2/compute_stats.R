
library("data.table")
library("ppcor")
library("tidyverse")
library("copula")
library("HiClimR")
library("infotheo")

start_time <- Sys.time()

setwd('C:/Users/Edward/Desktop/lez_crosssectional_v2/precomputed/linkage_1/')
getwd()

listed <- list.files('./pre/')

performPKe <- function(mytable) {
  out <- tryCatch(
    {
      # Just to highlight: if you want to use more than one 
      # R expression in the "try" part then you'll have to 
      # use curly brackets.
      # 'tryCatch()' will return the last evaluated expression 
      # in case the "try" part was completed successfully
      
      message("This is the 'try' part")
      
      intermediate <- pcor(mytable, method="kendall")
      resulted <- intermediate$estimate
      # The return value of `readLines()` is the actual value 
      # that will be returned in case there is no condition 
      # (e.g. warning or error). 
      # You don't need to state the return value via `return()` as code 
      # in the "try" part is not wrapped inside a function (unlike that
      # for the condition handlers for warnings and error below)
    },
    error=function(cond) {
      message("An error occured")
      message("Here's the original error message:")
      message(cond)
      # Choose a return value in case of error
      
      return(NA)
    },
    warning=function(cond) {
      message(paste("URL caused a warning:", url))
      message("Here's the original warning message:")
      message(cond)
      # Choose a return value in case of warning
      resulted <- matrix(NA, length(colnames(mytable)), length(colnames(mytable)))
      resulted = data.frame(resulted, row.names=colnames(mytable))
      colnames(resulted) <- colnames(mytable)
      
      return(resulted)
    },
    finally={
      # NOTE:
      # Here goes everything that should be executed at the end,
      # regardless of success or error.
      # If you want more than one expression to be executed, then you 
      # need to wrap them in curly brackets ({...}); otherwise you could
      # just have written 'finally=<expression>' 
      message("Some other message at the end")
    }
  )    
  return(out)
}

fun_is <- function(x) {
  value <- 1 / sqrt(x)
  return(value)
}

for (f in listed) {
  
  print(str_c("./pre/", f))
  input <- { str_c("./pre/", f) }
  data <- fread(input)
  data_disc <- discretize(data, "equalfreq")
  
  print(object.size(data), units='Mb')
  
  
  
  print('ops3')
  ghoul <- corKendall(data)
  
  write.csv(ghoul, str_c("./", substr(f, 1, nchar(f)-4), "/", "corr_kendall_direct.csv"))
  
  ghoul <- fastCor(data)
  
  write.csv(ghoul, str_c("./", substr(f, 1, nchar(f)-4), "/", "corr_pearson_direct.csv"))
  
  print('ops1')
  ghoul <- pcor(data, method="pearson")
  dagger <- ghoul$estimate
  
  dagger = data.frame(dagger, row.names=colnames(data))
  colnames(dagger) <- colnames(data)
  
  write.csv(dagger, str_c("./", substr(f, 1, nchar(f)-4), "/", "corr_pearson_partial.csv"))

  print('ops2')

  # THIS ONE IS TOO SLOW
  # that would be nice if we had fast partial kendall; try to find one later
  
  # dagger <- performPKe(data)

  # write.csv(dagger, str_c("./", substr(f, 1, nchar(f)-4), "/", "corr_kendall_partial.csv"))
  
  # entropy and conditional_entropy
  # see tmp_test.R
  # conditional one excluded due to 0/inf results / negative cmi
  
  de_result <- apply(data_disc, 2, entropy)
  
  de_vec <- matrix(de_result, nrow=length(de_result), ncol=1)
  de_row <- matrix(de_result, nrow=1, ncol=length(de_result))
  de_multiplied <- de_vec %*% de_row
  de_multiplied <- apply(de_multiplied, c(1, 2), fun_is)
  
  dm_result <- mutinformation(data_disc)
  dn_result <- dm_result * de_multiplied
  
  write.csv(dn_result, str_c("./", substr(f, 1, nchar(f)-4), "/", "corr_mutual_info_direct.csv"))

}

end_time = Sys.time()
print(end_time - start_time)
