# Set the delimiter of the CSV file (assuming it's a comma)
set datafile separator ","

set title "Comparison of parallel block and sequential approaches"
set xlabel "Matrix size"
set ylabel "Execution time (microseconds)"
set key outside
set output "outputs/bloc_vs_seq.png"

# Définir le terminal de sortie en PNG
set terminal png


# Tri des données en fonction de la première colonne
!sort -t "," -k1,1 -n -o outputs/sorted_data_2.csv "outputs/result_bloc_seq.csv"

str_sort= "outputs/sorted_data_2.csv"

# Définir les intervalles de marques sur l'axe des abscisses
set xtics ("100" 100, "5000" 5000, "50000" 50000)

# Tracé des données pour l'approche en bloc
plot str_sort using 1:2 with lines title "Bloc Approach", \
     str_sort using 1:3 with lines title "Seq Approach"