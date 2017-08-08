#!/bin/bash

SCALE=4
ITERS=100

CORES=$(sysctl -n hw.ncpu)

ARQ1="inputs/tests/arq1"
ARQ2="inputs/tests/arq2"
ARQ3="inputs/tests/arq3"
EXT=".in"

P_S_NO="./builds/main_serial_no_optimized.bin"
P_S_O="./builds/main_serial_optimized.bin"
P_P_NO="./builds/main_parallel_no_optimized.bin"
P_P_O="./builds/main_parallel_optimized.bin"


#######################################################
#  Executing not optimized version                        #
#######################################################

echo "NOT OPTIMIZED VERSION"

# Excuting all testes for arq1

# Get serial time
echo "\n"
SERIAL_TIME=0
for i in $(seq 1 $ITERS)
do
    TIME=$(${P_S_NO} < "${ARQ1}_1${EXT}" | tail -n 1 | awk '{print $1}')
    SERIAL_TIME=$(($SERIAL_TIME + $TIME))
done
SERIAL_TIME=$(bc -l <<< $SERIAL_TIME/$ITERS)
echo "Tiempo Serial: ${SERIAL_TIME}"

# Get Parallel times
echo "Test arq1 "
for N_THREADS in {1,2,4,8,16}
do
    COUNT=0
    for i in $(seq 1 $ITERS)
    do
        TIME=$(${P_P_NO} < "${ARQ1}_${N_THREADS}${EXT}" | tail -n 1 | awk '{print $1}')
        COUNT=$(($COUNT + $TIME))
    done
    echo "FOR THREADS: ${N_THREADS} using ${ARQ1}_${N_THREADS}${EXT}"
    AVRG=$(bc -l <<< ${COUNT}/${ITERS})
    SPEEDUP=$(bc -l <<< $SERIAL_TIME/$AVRG)
    EFICENCIA=$(bc -l <<< $SPEEDUP/$CORES)
    echo "Avrg: ${AVRG}"
    echo "SpeedUp: ${SPEEDUP}"
    echo "Eficencia: ${EFICENCIA}"
done

# Excuting all testes for arq2
# Get serial time
echo "\n"
SERIAL_TIME=0
for i in $(seq 1 $ITERS)
do
    TIME=$(${P_P_NO} < "${ARQ2}_1${EXT}" | tail -n 1 | awk '{print $1}')
    SERIAL_TIME=$(($SERIAL_TIME + $TIME))
done
SERIAL_TIME=$(bc -l <<< $SERIAL_TIME/$ITERS)
echo "Tiempo Serial: ${SERIAL_TIME}"

echo "Test arq2"
for N_THREADS in {1,2,4,8,16}
do
    COUNT=0
    for i in $(seq 1 $ITERS)
    do
        TIME=$(${P_P_NO} < "${ARQ2}_${N_THREADS}${EXT}" | tail -n 1 | awk '{print $1}')
        COUNT=$(($COUNT + $TIME))
    done
    echo "FOR THREADS: ${N_THREADS} using ${ARQ2}_${N_THREADS}${EXT}"
    AVRG=$(bc -l <<< ${COUNT}/${ITERS})
    SPEEDUP=$(bc -l <<< $SERIAL_TIME/$AVRG)
    EFICENCIA=$(bc -l <<< $SPEEDUP/$CORES)
    echo "Avrg: ${AVRG}"
    echo "SpeedUp: ${SPEEDUP}"
    echo "Eficencia: ${EFICENCIA}"
done


# Excuting all testes for arq3
# Get serial time
echo "\n"
SERIAL_TIME=0
for i in $(seq 1 $ITERS)
do
    TIME=$(${P_P_NO} < "${ARQ3}_1${EXT}" | tail -n 1 | awk '{print $1}')
    SERIAL_TIME=$(($SERIAL_TIME + $TIME))
done
SERIAL_TIME=$(bc -l <<< $SERIAL_TIME/$ITERS)
echo "Tiempo Serial: ${SERIAL_TIME}"

echo "Test arq3"
for N_THREADS in {1,2,4,8,16}
do
    COUNT=0
    for i in $(seq 1 $ITERS)
    do
        TIME=$(${P_P_NO} < "${ARQ3}_${N_THREADS}${EXT}" | tail -n 1 | awk '{print $1}')
        COUNT=$(($COUNT + $TIME))
    done
    echo "FOR THREADS: ${N_THREADS} using ${ARQ1}_${N_THREADS}${EXT}"
    AVRG=$(bc -l <<< ${COUNT}/${ITERS})
    SPEEDUP=$(bc -l <<< $SERIAL_TIME/$AVRG)
    EFICENCIA=$(bc -l <<< $SPEEDUP/$CORES)
    echo "Avrg: ${AVRG}"
    echo "SpeedUp: ${SPEEDUP}"
    echo "Eficencia: ${EFICENCIA}"
done

echo "\n\n\n\n"


#######################################################
#  Executing optimized version                        #
#######################################################


echo "OPTIMIZED VERSION"
# Excuting all testes for arq1

# Get serial time
echo "\n"
SERIAL_TIME=0
for i in $(seq 1 $ITERS)
do
    TIME=$(${P_S_O} < "${ARQ1}_1${EXT}" | tail -n 1 | awk '{print $1}')
    SERIAL_TIME=$(($SERIAL_TIME + $TIME))
done
SERIAL_TIME=$(bc -l <<< $SERIAL_TIME/$ITERS)
echo "Tiempo Serial: ${SERIAL_TIME}"

# Get Parallel times
echo "Test arq1 "
for N_THREADS in {1,2,4,8,16}
do
    COUNT=0
    for i in $(seq 1 $ITERS)
    do
        TIME=$(${P_P_O} < "${ARQ1}_${N_THREADS}${EXT}" | tail -n 1 | awk '{print $1}')
        COUNT=$(($COUNT + $TIME))
    done
    echo "FOR THREADS: ${N_THREADS} using ${ARQ1}_${N_THREADS}${EXT}"
    AVRG=$(bc -l <<< ${COUNT}/${ITERS})
    SPEEDUP=$(bc -l <<< $SERIAL_TIME/$AVRG)
    EFICENCIA=$(bc -l <<< $SPEEDUP/$CORES)
    echo "Avrg: ${AVRG}"
    echo "SpeedUp: ${SPEEDUP}"
    echo "Eficencia: ${EFICENCIA}"
done

# Excuting all testes for arq2
# Get serial time
echo "\n"
SERIAL_TIME=0
for i in $(seq 1 $ITERS)
do
    TIME=$(${P_S_O} < "${ARQ2}_1${EXT}" | tail -n 1 | awk '{print $1}')
    SERIAL_TIME=$(($SERIAL_TIME + $TIME))
done
SERIAL_TIME=$(bc -l <<< $SERIAL_TIME/$ITERS)
echo "Tiempo Serial: ${SERIAL_TIME}"

echo "Test arq2"
for N_THREADS in {1,2,4,8,16}
do
    COUNT=0
    for i in $(seq 1 $ITERS)
    do
        TIME=$(${P_P_O} < "${ARQ2}_${N_THREADS}${EXT}" | tail -n 1 | awk '{print $1}')
        COUNT=$(($COUNT + $TIME))
    done
    echo "FOR THREADS: ${N_THREADS} using ${ARQ2}_${N_THREADS}${EXT}"
    AVRG=$(bc -l <<< ${COUNT}/${ITERS})
    SPEEDUP=$(bc -l <<< $SERIAL_TIME/$AVRG)
    EFICENCIA=$(bc -l <<< $SPEEDUP/$CORES)
    echo "Avrg: ${AVRG}"
    echo "SpeedUp: ${SPEEDUP}"
    echo "Eficencia: ${EFICENCIA}"
done


# Excuting all testes for arq3
# Get serial time
echo "\n"
SERIAL_TIME=0
for i in $(seq 1 $ITERS)
do
    TIME=$(${P_S_O} < "${ARQ3}_1${EXT}" | tail -n 1 | awk '{print $1}')
    SERIAL_TIME=$(($SERIAL_TIME + $TIME))
done
SERIAL_TIME=$(bc -l <<< $SERIAL_TIME/$ITERS)
echo "Tiempo Serial: ${SERIAL_TIME}"

echo "Test arq3"
for N_THREADS in {1,2,4,8,16}
do
    COUNT=0
    for i in $(seq 1 $ITERS)
    do
        TIME=$(${P_P_O} < "${ARQ3}_${N_THREADS}${EXT}" | tail -n 1 | awk '{print $1}')
        COUNT=$(($COUNT + $TIME))
    done
    echo "FOR THREADS: ${N_THREADS} using ${ARQ1}_${N_THREADS}${EXT}"
    AVRG=$(bc -l <<< ${COUNT}/${ITERS})
    SPEEDUP=$(bc -l <<< $SERIAL_TIME/$AVRG)
    EFICENCIA=$(bc -l <<< $SPEEDUP/$CORES)
    echo "Avrg: ${AVRG}"
    echo "SpeedUp: ${SPEEDUP}"
    echo "Eficencia: ${EFICENCIA}"
done
