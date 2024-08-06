result_perf="community_detection_time.txt"
for k in `ls used_data/*.txt`
do
	./txt2gml $k 2
done
i=1
for k in `ls used_data/*.gml`
do
	graph=${k:0:(${#k}-4)}
	graphTxt=$graph".txt"
	graphBin=$graph".bin"
	graphTree=$graph".tree"
	echo "******** $graphTxt **********"
	#time ./convert -i $graphTxt -o $graphBin
	#time ./community $graphBin -l -1 -v > $graphTree
	perf stat -B -e cache-misses -o $result_perf --append ./convert -i $graphTxt -o $graphBin
	perf stat -B -e cache-misses -o $result_perf --append ./community $graphBin -l -1 -v > $graphTree
done
