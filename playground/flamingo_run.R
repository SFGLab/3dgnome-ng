# Structures of FLAMINGO from the contact matrices of a region, for playground/flamingo_arm.py.
#
#   Rscript playground/flamingo_run.R FRAG.tsv DOMAIN.tsv N_FRAG N_DOMAIN FRAG_RES DOMAIN_RES N THREADS OUT_PREFIX
#
# FRAG.tsv and DOMAIN.tsv hold the upper triangle of the two matrices as rows of i, j and value,
# counted from one. The steps and the settings are those of flamingo_main of FLAMINGOrLite at its
# defaults. flamingo_main reads a .hic or an .mcool file itself, with the map of the whole genome in
# memory, and this script gives it the two matrices in its place. A structure is one run with its
# own draw of the 75 percent of the entries that FLAMINGO fits. OUT_PREFIX_<k>.tsv has the rows
# frag_id, x, y and z of structure k, for the fragments that carry data.
args <- commandArgs(trailingOnly = TRUE)
frag_file <- args[1]
domain_file <- args[2]
n_frag <- as.integer(args[3])
n_domain <- as.integer(args[4])
frag_res <- as.numeric(args[5])
domain_res <- as.numeric(args[6])
n <- as.integer(args[7])
threads <- as.integer(args[8])
prefix <- args[9]

suppressPackageStartupMessages({
  library(Matrix)
  library(parallel)
  library(FLAMINGOrLite)
})
ns <- asNamespace("FLAMINGOrLite")
read_obj <- function(path, size) {
  t <- if (requireNamespace("data.table", quietly = TRUE)) {
    as.data.frame(data.table::fread(path, col.names = c("i", "j", "x")))
  } else {
    read.table(path, col.names = c("i", "j", "x"))
  }
  new("flamingo", IF = Matrix::sparseMatrix(i = t$i, j = t$j, x = t$x, dims = c(size, size)), n_frag = size, chr_name = "region")
}
high <- read_obj(frag_file, n_frag)
low <- read_obj(domain_file, n_domain)

# The defaults of flamingo_main.
sample_rate <- 0.75
lambda <- 10
r <- 1
max_dist <- 0.01
alpha <- -0.25
inf_dist <- 4
error_threshold <- 1e-3
max_iter <- 500

for (k in seq_len(n)) {
  out <- paste0(prefix, "_", k, ".tsv")
  if (file.exists(out)) next
  started <- Sys.time()
  set.seed(k)
  temp <- tempfile("flamingo")
  dir.create(temp)
  dir.create(file.path(temp, "domain_data"))
  ns$divide_domain(flamingo_high_res_obj = high, flamingo_low_res_obj = low, domain_res = domain_res, frag_res = frag_res, temp_folder = temp)
  backbone <- flamingo_backbone(temp, sample_rate, lambda, r, max_dist, error_threshold, max_iter, alpha, inf_dist)
  domains <- ns$flamingo_domain(temp, sample_rate, lambda, r, max_dist, error_threshold, max_iter, alpha, inf_dist, threads)
  res <- ns$assemble_structure(backbone, high, domains, max_iter)
  write.table(res[, c("frag_id", "x", "y", "z")], out, sep = "\t", quote = FALSE, row.names = FALSE)
  unlink(temp, recursive = TRUE)
  cat(sprintf("structure %d: %d fragments, %.1f min\n", k, nrow(res), as.numeric(difftime(Sys.time(), started, units = "mins"))))
}
