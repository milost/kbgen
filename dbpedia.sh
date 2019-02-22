#!/bin/bash

# save directories
current_dir=$(pwd)
dbpedia_dir=$1

# decompress archives
cd $dbpedia_dir
mkdir archives
for file in *.bz2; do
    echo "Decompressing $file"
    bzip2 -dk $file
    mv $file archives
done

# remove first and last line of each file
for file in *.ttl; do
    echo "Cleaning $file"
    sed '1d; $d' $file > tmp && mv tmp $file
done

# merge files
result_file="dbpedia.ttl"
> $result_file
for file in *.ttl; do
    echo "Appending $file to $result_file"
    cat $file >> $result_file
done
echo "Merged files into $result_file"

# return to original directory
cd $current_dir
