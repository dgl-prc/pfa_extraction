#!/usr/bin/env bash
##!/usr/bin/env bash

model_name=$1
trace_path=$2
output_path=$3
variables_path=$4
max_length=$5

java -jar ../utils/ZiQian_jar/ZiQian.jar  $model_name $trace_path $output_path $variables_path --length=$max_length

