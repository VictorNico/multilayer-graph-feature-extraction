# Set the delimiter of the CSV file (assuming it's a comma)
set datafile separator ","

set title "Speedup comparison on CREDIT RISK dataset"
set xlabel "Number of threads"
set ylabel "Speedup"

set key outside
set output "outputs/cr_speed_up.png"

# set PNG as output terminal
set terminal png

str_sort= "outputs/speedup_loan_status_-1_millisecond.csv"


# abscisse markers definition
set xtics auto
set xtics rotate

# Set x values ranges
xmin = 0
xmax = 6
set xrange [xmin:xmax]

# Set y values ranges
ymin = 0.0
ymax = 6.0
set yrange [ymin:ymax]

# plot
plot str_sort using 11:4 with lines title "Mask", \
     str_sort using 11:5 with lines title "IG", \
     str_sort using 11:6 with lines title "Mask,IG", \
     str_sort using 11:7 with lines title "Best Split", \
     str_sort using 11:8 with lines title "ID growth", \
     str_sort using 11:9 with lines title "ID growth,Best Split"
