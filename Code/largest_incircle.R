# in python https://stackoverflow.com/questions/4279478/largest-circle-inside-a-non-convex-polygon or https://stackoverflow.com/questions/1203135/what-is-the-fastest-way-to-find-the-visual-center-of-an-irregularly-shaped-pol

in_directory <- "C:/Users/swirl/OneDrive - UNSW/Honours/Thesis/Data Analysis or Code/Data/Test binary matrices/"
area_results = list()

in_circle_calculations (in_directory)

in_circle_calculations <- function(in_directory) {

  library (concaveman)
  library (tidyverse)
  library (sf)
  library (raster)
  library (geosphere)

in_circle <- function(leaf) {
  leaf = leaf[,-1]

#plot <- image(leaf)
#plot

# Var1 = row
# Var2 = column
df.long <- reshape2::melt(leaf)
df.subset <- df.long %>% 
  filter(value==1)

##Getting coordinates of outline
df.hull <- df.subset[,-3]
df.hull$Var2 <- gsub("X", "", as.character(df.hull$Var2)) %>% 
  as.numeric()
df.hull$Var1 <- as.numeric(df.hull$Var1) 
df.hull <- as.matrix(df.hull)

polygons <- concaveman(df.hull)
#x_avg = mean(polygons[, "V1"])
#y_avg = mean(polygons[, "V2"])
#centre = cbind(x_avg, y_avg)
#points(centre, col="red")



##Getting distances

#dobj <- dist(polygons, method = "euclidean")
#dmat <- as.matrix(dobj)
#diag(dmat) <- NA

#dmax <- max(apply(dmat,2,min,na.rm=TRUE))


polygon = st_polygon(list(as.matrix(rbind(polygons, polygons[1,]))))

p <- polylabelr::poi(polygon, precision = 0.01)
centre <- cbind(p[["x"]], p[["y"]])

distfromcent <- pointDistance(centre, polygons, lonlat = FALSE)
radius <- min(distfromcent)
theta = seq(0, 2 * pi, length = 200)

#Area of largest in circle
area <- pi * radius^2
}

filenames <- list.files(in_directory, pattern="*.csv", full.names=TRUE)
leaf <- lapply(filenames, read.csv)
leaf <- lapply(leaf, as.matrix)
.GlobalEnv$area_results <- sapply(leaf, in_circle)
}

df_incirclearea_results <- data.frame(area_results)
write.csv(df_incirclearea_results, "_incirclearea_results.csv")

#X <- as.matrix(df.hull)
tiff(file="del.tiff", compression = "zip")
plot(polygon, asp=1, cex=0.1)
points(centre)
lines(x = radius * cos(theta) + p[["x"]], y = radius * sin(theta) +  p[["y"]])
#hpts <- chull(X)
#hpts <- c(hpts, hpts[1])
#lines(X[hpts, ], col = "red")
dev.off()
