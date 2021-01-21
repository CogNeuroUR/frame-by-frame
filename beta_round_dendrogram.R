# Libraries
library(ggraph)
library(ggplot2)
library(ggdendro)
library(igraph)
library(tidyverse)
library(RColorBrewer) 
library(dendextend)
library(corrplot)
library(factoextra)
library(NbClust)
###################
#rdm <- read_csv("/run/media/ov/Data/experiments/CogNeuroUR/Repositories/frame-by-frame/outputs/rdm_average_softmax_cosine.csv", col_types = cols(X1 = col_skip()))
rdm <- read_csv("F:/experiments/CogNeuroUR/Repositories/frame-by-frame/outputs/rdm_average_softmax_cosine.csv", col_types = cols(X1 = col_skip()))
features <- as.matrix(
  read_csv("F:/experiments/CogNeuroUR/Repositories/frame-by-frame/outputs/RN50_features_softmax_average.csv", col_types = cols(X1 = col_skip())))
dist_mat = as.dist(rdm)

hc <- hclust(dist_mat)
plot(hc)
##################
# Cophenetic distances
l_methods = c("ward.D", "ward.D2", "single", "complete", "average",
                 "mcquitty", "median", "centroid")
l_cophenetic = c()
l_dendlist = dendlist()

i = 0
for (method in l_methods){
  print(method)
  hc = hclust(dist_mat, method = method)
  c = cor(dist_mat, cophenetic(hc))
  l_cophenetic = append(l_cophenetic, c)
  l_dendlist = dendlist(l_dendlist, as.dendrogram(hc))
  i = i + 1
}
names(l_dendlist) <- l_methods
dendlist_cor_spearman <- cor.dendlist(l_dendlist, method_coef = "spearman")
corrplot::corrplot(dendlist_cor_spearman, "pie", "lower")

########################################################
print(length(l_cophenetic))
#names(l_cophenetic) = l_methods
barplot(l_cophenetic, names = l_methods,
        ylab = 'Cophenetic corr. coef.',
        las=2)

linkage_method = "ward.D"
x = NbClust(features,
        diss=dist_mat, distance=NULL,
        min.nc = 150, max.nc = 300,
        method=linkage_method, index='silhouette')

#######################################################
# Silhouette's Idx.
#fviz_nbclust(x, method = "silhouette")
pdf(file = 'F:/experiments/CogNeuroUR/Repositories/frame-by-frame/plots/silhouette_idx_ward.D2.pdf',   # The directory you want to save the file in
    width = 20, # The width of the plot in inches
    height = 5) # The height of the plot in inches
fviz_nbclust(features, FUNcluster = hcut,
            diss = dist_mat,
            k.max = 300,
            method = "silhouette",
            hc_method = "ward.D2")

legend("topright", )

#at1 <- seq(1, length(plot$data$y), 10)
#axis(side=1,
#     at1)
dev.off()

################################
sil_analysis = fviz_nbclust(features, FUNcluster = hcut,
                            diss = dist_mat,
                            k.max = 300,
                            method = "silhouette",
                            hc_method = "ward.D2")

sprintf('Max Silh. Idx: %.3f', sil_analysis$data$y[which.max(sil_analysis$data$y)])
sprintf('for k = %d', sil_analysis$data$clusters[which.max(sil_analysis$data$y)])

##################
hc_method = "ward.D2"
best_k = c(sil_analysis$data$clusters[which.max(sil_analysis$data$y)])

dendrogram = as.dendrogram(hcut(dist_mat, k=221, isdiss=TRUE, hc_method=hc_method))

ddata <- dendro_data(dendrogram, type = "rectangle")
ggraph(dendrogram, layout = 'dendrogram') + 
  geom_edge_diagonal() +
  geom_node_text(data = ddata$labels, aes(label=label) , angle=90 , hjust=1, nudge_y=-0.1) +
  theme(legend.position="none")
