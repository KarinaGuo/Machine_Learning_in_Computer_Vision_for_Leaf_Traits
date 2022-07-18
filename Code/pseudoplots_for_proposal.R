a <- c(70,50,45,35,33,31,29,26,26,25,25,23,23,19,18,18,17.5,17,16,15,14,11,10,9,8)
b <- c(0.5,0.7,1.4,1,0.6,0.9,1.3,1.1,1.02,0.9,1,1.1,0.7,0.9,1,1.1,1.2,0.94,1.25,1.2,1.1,0.95,1,1.01,1.03)
sample <- data.frame(a, b)
colnames(sample) <- c("Divergence time (mya)", "Average slope of leaf size against precipitation")

ggplot(sample, aes(x=a, y=b)) +
  geom_point() +
  geom_smooth() +
  geom_hline (yintercept = 1.02, linetype="dashed", color = "darkgreen") +
  theme_bw() +
  labs(subtitle="How the relationship between leaf size and precipitation change across divergence time", 
       y="Average slope of leaf size against precipitation",
       x="Divergence time (mya)", 
       title="Trait-climate relationships at different levels of phylogeny") +
  annotate("text", x = 65, y=1.15, label = "Global trend \n(Wright et al. 2017)", size =3.5)
