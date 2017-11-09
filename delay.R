library(data.table)
library(dplyr)
setwd("C:/Users/Laurent Berder/Documents/Code/DemDet")

delay <- fread("BYPORTSERVICE_typecont.csv", dec=",", header = TRUE, sep=";")
plot(delay$ACTUAL_AMOUNT_# Il y a clairement un outlier (observation n° 89869)
delay <- delay[-89869, ]9869, ]

delay$pred_ratio = delay$ESTIM_AMOUNT_USD / delay$ACTUAL_AMOUNT_USD
plot(delay$ID_PERIODE_GESTION, delay$pred_ratio, type = 'l')

par_pays <- delay %>% 
  group_by(PAYS) %>%
  summarise("estim" = sum(ESTIM_AMOUNT_USD, na.rm = T),
            "actual" = sum(ACTUAL_AMOUNT_USD, na.rm = T),
            'ratio' = mean(pred_ratio))

# On joue avec des ordres de grandeurs très différents selon les variables, donc on peut les normaliser
centree <- cbind(delay[,1:7], scale(delay[,8:ncol(delay)], center=T, scale=T))

