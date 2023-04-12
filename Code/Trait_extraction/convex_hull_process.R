in_directory <- "C:/Users/swirl/OneDrive - UNSW/Honours/Thesis/Data Analysis or Code/Data/Test binary matrices/"
curvature_results <- list()

convex_hull_calculations <- function (in_directory){
  library(reshape2)
  library(sp)
  library (tidyverse)
  
  calculating_hull <- function(leaf) {
    leaf = leaf[,-1]
    df.long <- melt(as.matrix(leaf))
    df.subset <- df.long %>% 
      filter(value==1)
    df.hull <- df.subset[,-3]
    df.hull$Var2 <- gsub("X", "", as.character(df.hull$Var2)) %>% 
      as.numeric()
    df.hull$Var1 <- as.numeric(df.hull$Var1)
    df.hull <- as.matrix(df.hull)

#Finding hull area using https://chitchatr.wordpress.com/2015/01/23/calculating-the-area-of-a-convex-hull/
# Var1 = row
# Var2 = column
box.hpts <- chull(x = df.hull[,1], y = df.hull[,2])
box.hpts <- c(box.hpts, box.hpts[1])
box.chull.coords <- df.hull[box.hpts,]

chull.poly_hull <- Polygon(box.chull.coords, hole=F)
chull.area_hull <- chull.poly_hull@area

#
#chull.poly_leaf <- Polygon(df.hull, hole=F)
#chull.area_leaf <- chull.poly_leaf@area
chull.area_leaf <- nrow(df.subset)
#
curvature_ratio_not = chull.area_hull/chull.area_leaf
  }

  filenames <- list.files(in_directory, pattern="*.csv", full.names=TRUE)
  leaf <- lapply(filenames, read.csv)
  leaf <- lapply(leaf, as.matrix)
  .GlobalEnv$curvature_results <- sapply(leaf, calculating_hull)
  
  }

convex_hull_calculations (in_directory)

df_curvatureratio_results <- data.frame(curvature_results)
write.csv(df_curvatureratio_results, "curvatureratio_results.csv")




###plotting in base plot
#X <- as.matrix(df.hull)

#tiff(file="del.tiff", compression = "zip")
#plot(X, asp=1, cex=0.1)
#hpts <- chull(X)
#hpts <- c(hpts, hpts[1])
#lines(X[hpts, ], col = "red")
#dev.off()


## (not working and kept aside for future ref :) plotting image on ggplot from https://ggplot2.tidyverse.org/articles/extending-ggplot2.html
#find_hull <- function(df.long) df.long[chull(df.long[,1], df.long[,2]), ]
#hulls <- plyr::ddply(df.long, "Var1", find_hull)

#StatChull <- ggproto("StatChull", Stat,
#                     compute_group = function(data, scales) {
#                       data[chull(data$x, data$y), , drop = FALSE]
#                     },
#                     
#                     required_aes = c("x", "y")
#)
#stat_chull <- function(mapping = NULL, data = NULL, geom = "polygon",
#                      position = "identity", na.rm = FALSE, show.legend = NA, 
#                       inherit.aes = TRUE, ...) {
#  layer(
#    stat = StatChull, data = data, mapping = mapping, geom = geom, 
#    position = position, show.legend = show.legend, inherit.aes = inherit.aes,
#    params = list(na.rm = na.rm, ...)
#  )
#}

#library(ggplot2);
#gg <- ggplot(df.hull, aes(x = Var2, y = Var1)) + 
#  theme(axis.line=element_blank(),axis.text.x=element_blank(),
#        axis.text.y=element_blank(),axis.ticks=element_blank(),
#        axis.title.x=element_blank(),
#        axis.title.y=element_blank(),legend.position="none") +
#  scale_y_reverse() +
#  stat_chull(aes(color = "red"), geom = "line", alpha = 0.2)
#ggsave(plot=gg, filename="del.png", device = "png")

