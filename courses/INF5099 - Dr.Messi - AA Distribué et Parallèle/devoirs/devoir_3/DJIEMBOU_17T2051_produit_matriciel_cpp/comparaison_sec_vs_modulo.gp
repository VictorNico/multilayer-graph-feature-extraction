# Set the delimiter of the CSV file (assuming it's a comma)
set datafile separator ","

set title "Comparaison des approches en bloc et modulo"
set xlabel "Dimension de la matrice"
set ylabel "Temps d'exécution (micro-secondes)"
set key outside
set output "outputs/sec_vs_modulo.png"

# Définir le terminal de sortie en PNG
set terminal png


# Tri des données en fonction de la première colonne
!sort -t "," -k1,1 -n -o outputs/sorted_data_1.csv "outputs/result_sec_modulo.csv"

str_sort= "outputs/sorted_data_1.csv"

# Définir les intervalles de marques sur l'axe des abscisses
set xtics ("100" 100, "5000" 5000, "50000" 50000)

# Tracé des données pour l'approche en bloc
plot str_sort using 1:2 with lines title "Sec Approach", \
     str_sort using 1:3 with lines title "Modulo Approach"