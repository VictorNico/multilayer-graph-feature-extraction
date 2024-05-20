set datafile separator ","
# set terminal pngcairo size 800,600
set terminal png
set output "outputs/cluster_modulo_2_999_2_14_04_2024_18_34_18.png"

set xlabel "X"
set ylabel "Y"
set title "Bubble Chart"
set key off
# Define color palette
set palette defined (0 "red", 1 "blue", 2 "green", 3 "orange", 4 "purple", 5 "cyan", 6 "magenta", 7 "yellow", 8 "gray", 9 "brown")

# Define bubble size
centroid_size = 2.0
# Plot the data points
plot "outputs/clusters_modulo_2_999_2_14_04_2024_18_34_18.csv" using 1:2:3 with points linecolor variable pointtype 7,  \
	"outputs/centroids_modulo_2_999_2_14_04_2024_18_34_18.csv" using 1:2:3 with points linecolor variable pointtype 7 pointsize centroid_size
