CLANG="aclang -O3 -rtl-mode=profile"
FILENAMESERIAL="mvt.c"
OUTFILESERIAL="mvt_serial"

FILENAMEOMP="mvt_openmp.c"
OUTFILEOMP="mvt_gpu"

ITERS=100

NORMAL_FLAG="normal"
FLAGS=("none" "tile" "vectorize")
SIZES=("DMEDIUM" "DLARGE" "DEXTRALARGE")

echo "Create folders"

rm -R $NORMAL_FLAG
mkdir $NORMAL_FLAG

for FLAG in "${FLAGS[@]}"
do
    rm -R $FLAG
    mkdir $FLAG
done

echo "Compiling"

for SIZE in "${SIZES[@]}"
do
    $CLANG -$SIZE -o $NORMAL_FLAG"/"$OUTFILESERIAL"_"$SIZE $FILENAMESERIAL
done

for FLAG in "${FLAGS[@]}"
do
    for SIZE in "${SIZES[@]}"
    do
        $CLANG -opt-poly=$FLAG -$SIZE -o $FLAG"/"$OUTFILEOMP"_"$SIZE $FILENAMEOMP
    done
done


echo "Executing"

NORMAL_TIMES=()

for SIZE in "${SIZES[@]}"
do
    COUNT=0
    for ((i=0;i<$ITERS;i++))
    do
        TIME=$("./"$NORMAL_FLAG"/"$OUTFILESERIAL"_"$SIZE | tail -n 1 | awk '{print $1}')
        COUNT=$(bc -l <<< "$COUNT + $TIME")
    done
    AVRG=$(bc -l <<< ${COUNT}/${ITERS})
    NORMAL_TIMES+=($AVRG)
done

for FLAG in "${FLAGS[@]}"
do
    index=0
    for SIZE in "${SIZES[@]}"
    do
        COUNT=0
        for ((i=0;i<$ITERS;i++))
        do
            TIME=$("./"$FLAG"/"$OUTFILEOMP"_"$SIZE | tail -n 1 | awk '{print $1}')
            COUNT=$(bc -l <<< "$COUNT + $TIME")
        done
        AVRG=$(bc -l <<< ${COUNT}/${ITERS})
        SPEEDUP=$(bc -l <<< ${NORMAL_TIMES[$index]}/$AVRG)
        echo "Flag: $FLAG, Size: $SIZE, Time: $AVRG, Tiempo_serial: ${NORMAL_TIMES[$index]}, SPEEDUP: $SPEEDUP"

        index=$((index + 1))
    done
done


