download () {
    link=$1
    unpacked=$2
    packed=${3:-${unpacked}}

    if [ ! -f "$unpacked" ] ; then
        if [ $unpacked == $packed ] ; then
            wget "$link" -O "$unpacked" --no-check-certificate
            echo "download unpacked"
        else
            if [ ! -f "$packed" ] ; then
                wget "$link" -O "$packed" --no-check-certificate
                echo "download packed"
            fi
            gzip -d "$packed"
            echo "unpack"
        fi
    else
        echo "skip"
    fi
}

FILES=(
    "biencoder-trivia-train.json.gz"
    "biencoder-trivia-dev.json.gz"
    "trivia-train.qa.csv.gz"
    "trivia-dev.qa.csv.gz"
    "trivia-test.qa.csv.gz"
)
LINK="https://dl.fbaipublicfiles.com/dpr/data/retriever"
OUTPUT_DIR="../../data/trivia"


if [ ! -d "$OUTPUT_DIR" ] 
then
    echo "Directory ${OUTPUT_DIR} DOES NOT exists." 
    mkdir -p "$OUTPUT_DIR"
    # exit 9999 # die with error code 9999
fi


for file_name in "${FILES[@]}"
do
    echo $file_name

    unpacked_name="${file_name%.*}"
    OUTPUT_FILE="$OUTPUT_DIR/$file_name"
    OUTPUT_UNPACK="$OUTPUT_DIR/$unpacked_name"

    download "$LINK/$file_name" "$OUTPUT_UNPACK" "$OUTPUT_FILE"
done


WIKI_LINK="https://dl.fbaipublicfiles.com/dpr/wikipedia_split/psgs_w100.tsv.gz"

OUTPUT_FILE="$OUTPUT_DIR/psgs_w100.tsv.gz"
OUTPUT_UNPACK="$OUTPUT_DIR/psgs_w100.tsv"

echo "WIKI: psgs_w100.tsv.gz"
download $WIKI_LINK $OUTPUT_UNPACK $OUTPUT_FILE
