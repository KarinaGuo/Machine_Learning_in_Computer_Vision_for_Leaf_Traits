args <- commandArgs(trailingOnly=TRUE)

# some functions
#convex_hull_calculations <- function (in_directory){
suppressMessages(library (reshape2))
suppressMessages(library (sp))
suppressMessages(library (sf))
suppressMessages(library (tidyverse))
suppressMessages(library (concaveman))
suppressMessages(library (raster))
suppressMessages(library (geosphere))
suppressMessages(library (stringr))
suppressMessages(library (data.table))

leaf_file = args[1]
leaf <- fread(file = leaf_file, sep = ',', header = TRUE)

  calculating_area <- function(leaf){
    leaf = leaf[,-1]
    df.long <- reshape2::melt(as.matrix(leaf))
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
    leaf = leaf[,-1]
    df.long <- reshape2::melt(as.matrix(leaf))
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

filenames <- leaf_file
filenames_rem <- gsub(".*NSW", "NSW", filenames)
filenames_splt1 <- gsub("\\.csv", "", filenames_rem)
filenames_splt2 <- str_split(filenames_splt1, "_")
filename_edit <- gsub("NSW", "NSW:NSW:NSW ", filenames_splt2[[1]][1] )
calculating_area_real <- calculating_area(leaf) * 2.049942e-05
in_circle_real <- in_circle(leaf) * 2.049942e-05

cat(filename_edit,",",filenames_splt2[[1]][2],",",calculating_area_real,",", in_circle_real,",", calculating_hull(leaf))
