#Things for setting up: overlapping prediction leaves with high IoU, write classifier results differently

library(tidyverse)
library(data.table)
model_summary <- read.csv("C:/Users/swirl/OneDrive - UNSW/Honours/Thesis/Machine Learning/Leaf dimensions/Test Ten/model_summary.csv")
model_summary$file_name <- gsub(".jpg", "", model_summary$file_name)

model_matches <- read.csv("C:/Users/swirl/OneDrive - UNSW/Honours/Thesis/Machine Learning/Leaf dimensions/Test Ten/model_matches.csv") %>% 
  select(pr_size_px, gt_size_px)

classifier_result <- read.csv("C:/Users/swirl/OneDrive - UNSW/Honours/Thesis/Machine Learning/Leaf dimension classifier/Model_LBoth_MoreD/classifier_results_test.csv")
classifier_result$ind_pr <- as.integer(gsub(".jpg", "", classifier_result$ind_pr))
classifier_result$file_name <- gsub(".jpg", "", classifier_result$file_name)

validation_list <- read.csv("~/Uni/Honours/Thesis/Data Analysis or Code/Data/Labelling List/Validation_list_merged.csv") 
validation_list$file_name <- gsub(".jpg", "", validation_list$file_name)

model_matches_rmdup <- setDT(model_matches)[!duplicated((pr_size_px))] #removing duplicate matches

data <- left_join(model_summary, classifier_result, by = c("file_name", "ind_pr"))
data <- left_join(data, model_matches_rmdup, by = "pr_size_px") %>%
  mutate(match_gt = case_when(
    gt_size_px = is.na(gt_size_px) ~ "N",
    gt_size_px != is.na(gt_size_px) ~ "Y"
  ))

#create empty dataframe to append results to
stats = NULL

################

#Running the functions!
#Filling in the raw data
fill_table <- function(data, validation_list){
  #Calculating classifier T/F,N/P
  status_LDC_N <- data %>% 
    filter(gt_class == "N") %>% 
    mutate(status_LDC = case_when(
      gt_class == pr_class ~ "TN",
      gt_class != pr_class ~ "FP"
    ))
  
  status_LDC_P <- data %>% 
    filter(gt_class == "Y") %>% 
    mutate(status_LDC = case_when(
      gt_class == pr_class ~ "TP",
      gt_class != pr_class ~ "FN"
    ))
  data_fill <- rbind(status_LDC_P, status_LDC_N)
  
  #Counting number of predictions leaves per file
  count <- data %>% 
    group_by(file_name) %>% 
    summarise (count = n())
  .GlobalEnv$data_fill <- left_join(data_fill, count, by = "file_name") 
}


#Calculate leaf dimension (detectron2 model) metrics
calculate_precision_LD <- function(data_fill){
  #Counting number of predicted leaves that match the groundtruth
  count_detectron <- data_fill %>% 
    group_by(match_gt) %>% 
    summarise (count_detectron = n())
  count_detectron <- count_detectron[order(count_detectron$match_gt),,drop=FALSE]
  
  TP = count_detectron[[2,2]]
  FP = count_detectron[[1,2]]
  .GlobalEnv$precision_LD = (TP/(FP+TP))
}
  
calculate_recall_LD <- function(data_fill){
  #Counting number of groundtruth leaves per file
  detectron_gt_count <- validation_list %>% 
    select(file_name, detectron_gt_count) %>% 
    summarise(detectron_gt_count = sum(detectron_gt_count))
  
  #Counting number of predicted leaves that match the groundtruth
  count_detectron <- data_fill %>% 
    group_by(match_gt) %>% 
    summarise (count_detectron = n())
  count_detectron <- count_detectron[order(count_detectron$match_gt),,drop=FALSE]
  
  TP = count_detectron[[2,2]]
  FN = detectron_gt_count[[1,1]] - TP #False Negative = total number of ground truth leaves - total number of true predicted leaves
  .GlobalEnv$recall_LD = (TP/(FN+TP))
}  

#Calculate harmonic mean of leaf dimension (detectron2 model)
calculate_F1_LD <- function(precision_LD, recall_LD) {
  .GlobalEnv$F1_LD = 2*(precision_LD*recall_LD)/(precision_LD+recall_LD)
}

#calculating classifier metrics
calculate_precision_LDC <- function(data_fill){
  #Count number of instances of F/T,N/P
  count_status_LDC <- data_fill %>% 
    group_by(status_LDC) %>% 
    summarise(count_status_LDC = n())
  count_status_LDC <- count_status_LDC[order(count_status_LDC$status_LDC),,drop=FALSE]
  
  FN = count_status_LDC[[1,2]]
  FP = count_status_LDC[[2,2]]
  TN = count_status_LDC[[3,2]]
  TP = count_status_LDC[[4,2]]
  
  .GlobalEnv$precision_LDC = (TP/(FP+TP))
}

calculate_recall_LDC <- function(data_fill){
  #Count number of instances of F/T,N/P
  count_status_LDC <- data_fill %>% 
    group_by(status_LDC) %>% 
    summarise(count_status_LDC = n())
  count_status_LDC <- count_status_LDC[order(count_status_LDC$status_LDC),,drop=FALSE]
  
  FN = count_status_LDC[[1,2]]
  FP = count_status_LDC[[2,2]]
  TN = count_status_LDC[[3,2]]
  TP = count_status_LDC[[4,2]]
  
  .GlobalEnv$recall_LDC = (TP/(FN+TP))
}

calculate_F1_LDC <- function(recall_LDC,precision_LDC) {
  .GlobalEnv$F1_LDC = 2*(precision_LDC*recall_LDC)/(precision_LDC+recall_LDC)
}

#Calculate overall
calculate_precision_all <- function(data_fill){
  #Select leaves that were only predicted by the classifier as true leaves
  true_predicted_leaves <- data_fill %>% 
    filter (pr_class == "Y")
  
  #counting number of total true predicted labels "Y", and false predicted labels "N"
  count_detectron <- true_predicted_leaves %>% 
    group_by(match_gt) %>% 
    summarise (count_detectron = n())
  count_detectron <- count_detectron[order(count_detectron$match_gt),,drop=FALSE]
  
  TP = count_detectron[[2,2]]
  FP = count_detectron[[1,2]]
  .GlobalEnv$precision_all = (TP/(FP+TP))
}

calculate_recall_all <- function(data_fill){
  #counting number of total ground truth labels
  detectron_gt_count <- validation_list %>% 
    select(file_name, detectron_gt_count) %>% 
    summarise(detectron_gt_count = sum(detectron_gt_count))
  
  #Select leaves that were only selected by the classifier as true leaves
  true_predicted_leaves <- data_fill %>% 
    filter (pr_class == "Y")
  
  #counting number of total true predicted labels "Y", and false predicted labels "N"
  count_detectron <- true_predicted_leaves %>% 
    group_by(match_gt) %>% 
    summarise (count_detectron = n())
  count_detectron <- count_detectron[order(count_detectron$match_gt),,drop=FALSE]
  
  TP = count_detectron[[2,2]]
  FN = detectron_gt_count[[1,1]] - TP
  .GlobalEnv$recall_all = (TP/(FN+TP))
}  

calculate_F1_all <- function(recall_all,precision_all) {
  .GlobalEnv$F1_all = 2*(precision_all*recall_all)/(precision_all+recall_all)
}

############

#Calculating results

fill_table(data, validation_list)

calculate_precision_LD(data_fill)
calculate_recall_LD(data_fill)
calculate_F1_LD(precision_LD, recall_LD)

calculate_precision_LDC(data_fill)
calculate_recall_LDC(data_fill)
calculate_F1_LDC(precision_LDC, recall_LDC)

calculate_precision_all(data_fill)
calculate_recall_all(data_fill)
calculate_F1_all(precision_all, recall_all)


