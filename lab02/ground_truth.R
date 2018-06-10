library("dplyr")
library("readr")

source("../Lab 1/Noticias/mongo_utils.R")

noticias <- get_collection("estadaoNoticiasProcessadas")

data_inicio = "2014-01-01"
data_fim = "2014-12-31"

noticias_eleicao = noticias %>% filter(timestamp >= data_inicio & timestamp <= data_fim) %>% select(timestamp, titulo, subTitulo, conteudo, url)
noticias_eleicao$idNoticia = 1:nrow(noticias_eleicao)
write_csv(noticias_eleicao, "estadao_noticias_eleicao.csv")
