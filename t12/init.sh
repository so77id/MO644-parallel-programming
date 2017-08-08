source /opt/ompcloud/ompcloud-export-var.sh
export OMPCLOUD_CONF_PATH="/home/msc2017/ra192744/Desktop/aulas/parallel_programming/t12/cloud_rtl.ini.azure"
clang -fopenmp -omptargets=x86_64-unknown-linux-spark code-t12.c -o code.bin
