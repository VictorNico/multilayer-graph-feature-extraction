# Set the delimiter of the CSV file (assuming it's a comma)
set datafile separator ","

set title "Comparison of block and modulo approaches"
set xlabel "Matrix size" offset -10, 0
set ylabel "Execution time (microseconds)" offset 15, -2

set key outside
set output "outputs/bloc_vs_modulo_vs_nthread.png"

# set PNG as output terminal
set terminal png


# data sorting
!sort -t "," -k1,1 -k4,4 -n -o outputs/sorted_data.csv "outputs/result_bloc_modulo.csv"

str_sort= "outputs/sorted_data.csv"

# markers definition
set xtics auto offset -2, -1
set ytics auto offset 2, -1
set xtics rotate by 45

# Set x values ranges
zmin = 0
zmax = 10
set zrange [zmin:zmax]


# plot
splot str_sort using 1:2:4 with lines title "Bloc Approach", \
     str_sort using 1:3:4 with lines title "Modulo Approach"