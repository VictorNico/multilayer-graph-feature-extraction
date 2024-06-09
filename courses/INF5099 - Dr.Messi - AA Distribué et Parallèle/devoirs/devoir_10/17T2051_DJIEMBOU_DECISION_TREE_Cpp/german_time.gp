# Set the delimiter of the CSV file (assuming it's a comma)
set datafile separator ","

set title "Speedup comparison on GERMAN dataset"
set xlabel "Number of threads"
set ylabel "time"

set key outside
set output "outputs/g_time.png"

# set PNG as output terminal
set terminal png


str_sort= "outputs/times_Class_-1_millisecond.csv"


# abscisse markers definition
set xtics auto
set xtics rotate

# Set x values ranges
# xmin = 0
# xmax = 17
# set xrange [xmin:xmax]

# Set y values ranges
# ymin = 0.0
# ymax = 4.0
# set yrange [ymin:ymax]

# plot
plot str_sort using 12:4 with lines title "Sequential", \
     str_sort using 12:5 with lines title "Mask", \
     str_sort using 12:6 with lines title "IG", \
     str_sort using 12:7 with lines title "Mask,IG", \
     str_sort using 12:8 with lines title "Best Split", \
     str_sort using 12:9 with lines title "ID growth", \
     str_sort using 12:10 with lines title "ID growth,Best Split"
