leaf <- read.csv("C:/Users/swirl/Downloads/NSW326115_1.csv")

convex_hull_calculations <- function (in_directory){
  library(reshape2)
  library(sp)
  library (tidyverse)
  
  calculating_area <- function(leaf){
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
    chull.area_leaf = (nrow(df.subset))
  }
  
  calculating_hull <- function(leaf) {
    leaf = leaf[,-1]*1
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
    chull.area_leaf <- nrow(df.subset)
    
    curvature_ratio_not = chull.area_hull/chull.area_leaf
  }
  
  filenames <- list.files(in_directory, pattern="*.csv", full.names=TRUE)
  .GlobalEnv$filename_file <- sapply(filenames,basename) 
  leaf <- lapply(filenames, read.csv)
  leaf <- lapply(leaf, as.matrix)
  .GlobalEnv$curvature_results <- sapply(leaf, calculating_hull)
  .GlobalEnv$mask_area_results <- sapply(leaf, calculating_area)
}

in_circle_calculations <- function(in_directory) {
  
  library (concaveman)
  library (tidyverse)
  library (sf)
  library (raster)
  library (geosphere)
  
  in_circle <- function(leaf) {
    #leaf <- read.csv("~/Uni/Honours/Thesis/Data Analysis or Code/Data/test_notcurved2.csv")
    leaf = leaf[,-1]
    # Var1 = row
    # Var2 = column
    df.long <- reshape2::melt(as.matrix(leaf))
    df.subset <- df.long %>% 
      filter(value==1)
    
    ##Getting coordinates of outline
    df.hull <- df.subset[,-3]
    df.hull$Var2 <- gsub("X", "", as.character(df.hull$Var2)) %>% 
      as.numeric()
    df.hull$Var1 <- as.numeric(df.hull$Var1) 
    df.hull <- as.matrix(df.hull)
    
    polygons <- concaveman(df.hull)
    polygon = st_polygon(list(as.matrix(rbind(polygons, polygons[1,]))))
    
    p <- polylabelr::poi(polygon, precision = 0.01)
    centre <- cbind(p[["x"]], p[["y"]])
    
    distfromcent <- pointDistance(centre, polygons, lonlat = FALSE)
    radius <- min(distfromcent)
    theta = seq(0, 2 * pi, length = 200)
    
    #Area of largest in circle
    area = pi * radius^2
  }
  
  filenames <- list.files(in_directory, pattern="*.csv", full.names=TRUE)
  leaf <- lapply(filenames, read.csv)
  .GlobalEnv$circle_area_results <- sapply(leaf, in_circle)
}


filenames_rem <- gsub(".*NSW", "NSW", filenames)
filenames_splt1 <- gsub("\\.csv", "", filenames_rem)
filenames_splt2 <- str_split(filenames_splt1, "_")

convex_hull_calculations (in_directory)
in_circle_calculations (in_directory)

curvature_ratio <- as.data.frame(curvature_results)
mask_area <- as.data.frame(mask_area_results)
in_circle_area <- as.data.frame(circle_area_results)
filenames <- data.frame(matrix(unlist(filenames_splt2), 
                               nrow=length(filenames_splt2), byrow=TRUE))

stats=NULL
stats=(as.data.frame(c(filenames, circle_area_results, curvature_results, mask_area_results),
                     col.names = c(c('file_name','index', 'circle_area_results', 'curvature_results', "mask_area_results")), 
                     stringsAsFactors=FALSE))
#write.csv(stats, "~/Delete/temp_2/temp.csv", index=FALSE)
