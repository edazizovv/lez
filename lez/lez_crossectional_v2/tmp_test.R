
library("data.table")
library("infotheo")
library("tidyverse")

start_time <- Sys.time()

setwd('C:/Users/Edward/Desktop/lez_crosssectional_v2/precomputed/linkage_1/')
getwd()

f = "CS2.csv"

print(str_c("./pre/", f))
input <- { str_c("./pre/", f) }
data <- fread(input)
data_disc <- discretize(data, "equalfreq", 100)

print(object.size(data), units='Mb')

total_conditor <- function(i) {
  value <- condentropy(data_disc[ , i], data_disc[ , -i])
}

fun_is <- function(x) {
  value <- 1 / sqrt(x)
}

mi_index_conditor <- function(x) {
  i <- ((x - 1) %% length(colnames(data_disc))) + 1
  j <- floor((x - 1) / length(colnames(data_disc))) + 1
  value <- condinformation(data_disc[, i], data_disc[, j], data_disc[, -c(i, j)])
}

de_result <- apply(data_disc, 2, entropy)

de_vec <- matrix(de_result, nrow=length(de_result), ncol=1)
de_row <- matrix(de_result, nrow=1, ncol=length(de_result))
de_multiplied <- de_vec %*% de_row
de_multiplied <- apply(de_multiplied, c(1, 2), fun_is)

pe_result <- unlist(lapply(1:length(colnames(data_disc)), total_conditor))
names(pe_result) <- colnames(data_disc)

pe_vec <- matrix(pe_result, nrow=length(pe_result), ncol=1)
pe_row <- matrix(pe_result, nrow=1, ncol=length(pe_result))
pe_multiplied <- pe_vec %*% pe_row
pe_multiplied <- apply(pe_multiplied, c(1, 2), fun_is)

dm_result <- mutinformation(data_disc)
dn_result <- dm_result * de_multiplied

indexer_mx <- matrix(1:(length(colnames(data_disc)) * length(colnames(data_disc))), nrow=length(colnames(data_disc)), ncol=length(colnames(data_disc)))
pm_result <- apply(indexer_mx, c(1, 2), mi_index_conditor)
pn_result <- pm_result * pe_multiplied

end_time = Sys.time()
print(end_time - start_time)
