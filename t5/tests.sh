

ARQ1="inputs/arq1.in"
ARQ2="inputs/arq2.in"
ARQ3="inputs/arq3.in"
ARQ4="inputs/arq4.in"
ARQ5="inputs/arq5.in"
ARQ6="inputs/arq6.in"

PROG="./builds/senha-parallel.bin"

echo "Ejecutando ARQ1"
echo "Ejecutando ARQ1" >> output.out
$($PROG < $ARQ1 >> output.out)
echo "Ejecutando ARQ2"
echo "Ejecutando ARQ2" >> output.out
$($PROG < $ARQ2 >> output.out)
echo "Ejecutando ARQ3"
echo "Ejecutando ARQ3" >> output.out
$($PROG < $ARQ3 >> output.out)
echo "Ejecutando ARQ4"
echo "Ejecutando ARQ4" >> output.out
$($PROG < $ARQ4 >> output.out)
echo "Ejecutando ARQ5"
echo "Ejecutando ARQ5" >> output.out
$($PROG < $ARQ5 >> output.out)
echo "Ejecutando ARQ6"
echo "Ejecutando ARQ6" >> output.out
$($PROG < $ARQ6 >> output.out)