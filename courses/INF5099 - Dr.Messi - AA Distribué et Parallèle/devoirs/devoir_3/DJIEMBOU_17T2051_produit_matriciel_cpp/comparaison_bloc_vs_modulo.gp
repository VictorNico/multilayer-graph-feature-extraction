# Set the delimiter of the CSV file (assuming it's a comma)
set datafile separator ","

set title "Comparison of block and modulo approaches"
set xlabel "Matrix size"
set ylabel "Execution time (microseconds)"
set key outside
set output "outputs/bloc_vs_modulo.png"

# set PNG as output terminal
set terminal png


# data sorting
!sort -t "," -k1,1 -k4,4 -n -o outputs/sorted_data.csv "outputs/result_bloc_modulo.csv"

str_sort= "outputs/sorted_data.csv"

# abscisse markers definition
set xtics auto
set xtics rotate

# Set x values ranges
# xmin = 0
# xmax = 10000
# set xrange [xmin:xmax]


# plot
plot str_sort using 1:2 with lines title "Bloc Approach", \
     str_sort using 1:3 with lines title "Modulo Approach"