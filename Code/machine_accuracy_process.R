### NEED TO FIND/REPLACE AT LABELS HERE
test_meta_dir = "C:/Users/swirl/OneDrive - UNSW/Honours/Thesis/Machine Learning/Labels/Labelling List/Validation_list.csv"
model_summary_dir = "~/Uni/Honours/Thesis/Machine Learning/Leaf dimensions/Test Five/model_summary.csv"
model_matches_dir = "~/Uni/Honours/Thesis/Machine Learning/Leaf dimensions/Test Five/model_matches.csv"
labels = list("Leaf100", "Leaf90", "Leaf50")

machine_to_accuracy("Five", test_meta_dir, model_summary_dir, model_matches_dir)

machine_to_accuracy <- function(test, test_meta_dir, model_summary_dir, model_matches_dir) {
  library (tidyverse)
  library (ggpubr)
  
  dir.create(path = paste0(getwd(), "/Machine_accuracy_results/Test ",paste0(test)))
  
  test_meta <- read.csv(test_meta_dir) %>% 
    select (id, !!!labels, decimalLongitude, decimalLatitude) %>% 
    filter (!is.na(Leaf90))
  
  model_summary <-read.csv(model_summary_dir)
  
  #Appending category labels
  model_matches_1 <- read.csv(file = model_matches_dir) %>% 
    select (gt_cat_id, pr_cat_id, mask_iou)
  model_matches_2 <- read.csv(file = model_matches_dir) %>% 
    select(mask_iou, file_name, gt_size_px, pr_size_px, ind_gt, ind_pr)
  model_matches_1[model_matches_1 == "0"] <- labels[1]
  model_matches_1[model_matches_1 == "1"] <- labels[2]
  model_matches_1[model_matches_1 == "2"] <- labels[3]
  #model_matches_1[model_matches_1 == "3"] <- labels[4]
  model_matches <- left_join(model_matches_1, model_matches_2, by="mask_iou")
  .GlobalEnv$model_matches <- model_matches
  
  model_summary["pr_cat_id"][model_summary["pr_cat_id"] == "0"] <- labels[1]
  model_summary["pr_cat_id"][model_summary["pr_cat_id"] == "1"] <- labels[2]
  model_summary["pr_cat_id"][model_summary["pr_cat_id"] == "2"] <- labels[3]
  #model_summary["pr_cat_id"][model_summary["pr_cat_id"] == "3"] <- labels[4]
  
  ###Plotting 
  #Counting all categories
  
  .GlobalEnv$model_summary_plot <- model_summary %>% 
    count(pr_cat_id)
  #model_summary_plot[nrow(model_summary_plot) + 1,] <- c("Leaf100", 0)
  #model_summary_plot[nrow(model_summary_plot) + 1,] <- c("Leaf90", 0)
  model_summary_plot <- transform(model_summary_plot, n=as.numeric(n))
  
  model_summary_plot$pr_cat_id <- factor(model_summary_plot$pr_cat_id, levels=c(labels[1], labels[2], labels[3]#, labels[4]
                                                                                ))
  
  p1 <- ggplot(model_summary_plot, aes(x=pr_cat_id, y=n, fill=pr_cat_id)) +
    geom_bar(stat = "identity") +
    scale_x_discrete(drop = FALSE) +
    ggtitle ("Count of labels from prediction") +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 50, vjust = 0.5, hjust=0.35), legend.position = "none")
  
  #
  .GlobalEnv$model_matches_plot <- model_matches %>%
    count(pr_cat_id)
  #model_matches_plot[nrow(model_matches_plot) + 1,] <- c("Leaf100", 0)
  #model_matches_plot[nrow(model_matches_plot) + 1,] <- c("Leaf90", 0)
  model_matches_plot <- transform(model_matches_plot, n=as.numeric(n))
  
  model_matches_plot$pr_cat_id <- factor(model_matches_plot$pr_cat_id, levels=c(labels[1], labels[2], labels[3]#, labels[4]
                                                                                ))
  p2 <- ggplot(model_matches_plot, aes(x=pr_cat_id, y=n, fill=pr_cat_id)) +
    geom_bar(stat = "identity") +
    scale_x_discrete(drop = FALSE) +
    ggtitle ("Count of labels that match same mapping") +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 50, vjust = 0.5, hjust=0.35), legend.position = "none")
  
  model_matches_all$pr_cat_id <- as.list(levels(model_matches_all$pr_cat_id))
  
  #labels here
  test_meta_plot <- test_meta %>% 
    select(!!!labels, id) %>% 
    transform(Leaf100=as.numeric(Leaf100), Leaf90=(Leaf90), Leaf50=as.numeric(Leaf50)#[, labels[4]=as.numeric(labels[4])]
              )
#    transform(Leaf100=as.numeric(Leaf100), Leaf90=as.numeric(Leaf90), Leaf50=as.numeric(Leaf50)))
  test_meta_plot <- reshape2::melt(test_meta_plot, variable.name = "pr_cat_id", value.name = "n")
  test_meta_plot <- test_meta_plot %>% 
    group_by(pr_cat_id) %>% 
    summarise(n = sum(n))
  
  test_meta_plot$pr_cat_id <- factor(test_meta_plot$pr_cat_id, levels=c(labels[1], labels[2], labels[3]#, labels[4]
                                                                        ))
  .GlobalEnv$test_meta_plot
  
  p3 <- ggplot(test_meta_plot, aes(x=pr_cat_id, y=n, fill=pr_cat_id)) +
    geom_bar(stat = "identity") +
    scale_x_discrete(drop = FALSE) +
    ggtitle ("Count of labels from groundtruth") +
    theme_bw() +
    theme(axis.text.x = element_text(angle = 50, vjust = 0.5, hjust=0.35), legend.position = "none")
  
  ##  
  
  .GlobalEnv$accuracy_plots <- ggarrange(p1, p3, p2, ncol=1)
  
  
  pt <- function(accuracy_plots){
    ggsave(plot=accuracy_plots, filename=(paste0("Machine_accuracy_results/Test ",paste0(test),"/accuracy_plots_counts.jpeg")), width = 2200, height = 2100, units="px") 
    print(accuracy_plots)
  }
  pt(accuracy_plots)

  ### 
  model_summary_count <- model_summary_plot  
  colnames(model_summary_count)<-c("pr_cat_id", "prediction")
  test_meta_count <- test_meta_plot
  colnames(test_meta_count)<-c("pr_cat_id", "groundtruth")
  
  .GlobalEnv$difference <- inner_join(test_meta_count, model_summary_count, by="pr_cat_id") %>% 
    mutate(difference=groundtruth-prediction)
  
  #IoU
  .GlobalEnv$model_matches_IoU <- model_matches %>% 
    filter(mask_iou>0.7) %>% 
    group_by(gt_cat_id) %>% 
    summarise(average = mean(mask_iou))
  
  #precision
  model_matches_precision <- model_matches %>% 
    filter(mask_iou>0.7) %>%
    group_by(pr_cat_id) %>% 
    count(pr_cat_id)
  
  colnames(model_matches_precision) <- c('pr_cat_id','model_matches_threshold')
  
  model_matches_all <- model_matches_plot
  colnames(model_matches_all) <- c('pr_cat_id','model_matches_all')
  model_matches_all$pr_cat_id <- as.list(levels(model_matches_all$pr_cat_id))
  
  p_df_join <- left_join(model_matches_precision, model_matches_all, by='pr_cat_id')

  p_all = sum(p_df_join["model_matches_threshold"])/sum(p_df_join["model_matches_all"])
  .GlobalEnv$p_df <- p_df_join %>% 
    group_by(pr_cat_id) %>% 
    mutate(p = (model_matches_threshold/model_matches_all)) %>% 
    mutate(p_all = p_all)
  
  #recall -
  r_all = sum(model_summary_count["prediction"]) / sum(test_meta_count["groundtruth"])
  .GlobalEnv$r_df_join <- merge(model_summary_count, test_meta_count, by ='pr_cat_id') %>% 
    group_by (pr_cat_id) %>% 
    mutate (r = (sum(prediction) / sum(groundtruth))) %>% 
    mutate (r_all = r_all)
  
  #harmonic mean of precision and recall,
  .GlobalEnv$F1 = (2*(p_all*r_all))/p_all+r_all
  
  #Formatting
  difference_together <- difference %>% 
    select(pr_cat_id, difference)
  colnames(difference_together)<-c("pr_cat_id", "difference")
  IoU_together <- model_matches_IoU 
  colnames(IoU_together) <- c("pr_cat_id", "IoU_cat_avg")
  difference_together$pr_cat_id <- as.list(levels(difference_together$pr_cat_id))
  
  .GlobalEnv$IoU_differences_join <- left_join(difference_together, IoU_together, by = "pr_cat_id")
  #numercal_accuracy_p_r_F1 <- data.frame(p,r,F1) %>% 
    #reshape2::melt()
  .GlobalEnv$IoU_differences_join <- apply(IoU_differences_join,2,as.character)
  .GlobalEnv$p_df <- apply(p_df,2,as.character)
  .GlobalEnv$r_df_join <- apply(r_df_join,2,as.character)
  #all together now!
  csv <- function(IoU_differences_join, p_df, r_df_join, F1){
   write.csv(IoU_differences_join, file = paste0(getwd(), "/Machine_accuracy_results/Test ", paste0(test),"/IoU_differences_join.csv"))
   write.csv(p_df, file = paste0(getwd(), "/Machine_accuracy_results/Test ",paste0(test),"/numercal_accuracy_p_df.csv"))
   write.csv(r_df_join, file = paste0(getwd(), "/Machine_accuracy_results/Test ",paste0(test),"/numercal_accuracy_r_df.csv"))
   write.csv(F1, file = paste0(getwd(), "/Machine_accuracy_results/Test ",paste0(test),"/numercal_accuracy_F1_df.csv"))
  }
  
  csv(IoU_differences_join, p_df, r_df_join, F1)
}
