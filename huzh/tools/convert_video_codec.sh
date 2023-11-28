#! /bin/bash

input_path=$1
output_path=$2
filter_str=$3
echo "input_path: $input_path"
echo "output_path: $output_path"
echo "filter_str: $filter_str"

if test -d $output_path; then
    echo "$output_path exist"
else
    echo "make dir: $output_path"
    mkdir -p $output_path
fi

function convert_video_codec() {
    echo "convert_video_codec process: $1"
    echo "cmd: ffmpeg -loglevel quiet -i "$1" -vcodec copy -vtag hvc1 "${output_path}/$(basename $1)" -y"
    ffmpeg -loglevel quiet -i "$1" -vcodec copy -vtag hvc1 "${output_path}/$(basename $1)" -y
    if [ $? -eq 0 ]; then
        echo "convert to h265 succ."
    else
        echo "convert to h265 fail, back to h264"
        echo "cmd: ffmpeg -loglevel quiet -i "$1" -vcodec copy "${output_path}/$(basename $1)" -y"
        ffmpeg -loglevel quiet -i "$1" -vcodec copy "${output_path}/$(basename $1)" -y
    fi
    if [ $? -eq 0 ]; then
        echo "convert to h264 succ."
    else
        echo "convert to h264 fail, pass"
    fi
}

if [ "$filter_str" = "" ]; then
    for i in $(find $input_path -name "*.mp4" -o -name '*.avi' -o -name '*.h264'); do
        convert_video_codec ${i}
    done
else
    for i in $(find $input_path -name "*.mp4" -o -name '*.avi' -o -name '*.h264' | grep $filter_str); do
        convert_video_codec ${i}
    done
fi
