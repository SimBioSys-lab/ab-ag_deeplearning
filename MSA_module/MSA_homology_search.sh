# process protein (MSA + homology search)
function proteinMSA {
    seqfile=$1
    tag=$2
    hhpred=$3

    # generate MSAs
    if [ ! -s $WDIR/$tag.msa0.a3m ]
    then
        echo "Running HHblits"
        echo " -> Running command: $PIPEDIR/input_prep/make_protein_msa.sh $seqfile $WDIR $tag $CPU $MEM"
        $PIPEDIR/input_prep/make_protein_msa.sh $seqfile $WDIR $tag $CPU $MEM > $WDIR/log/make_msa.$tag.stdout 2> $WDIR/log/make_msa.$tag.stderr
    fi

    if [[ $hhpred -eq 1 ]]
    then
        # search for templates
        if [ ! -s $WDIR/$tag.hhr ]
        then
            echo "Running hhsearch"
            HH="hhsearch -b 50 -B 500 -z 50 -Z 500 -mact 0.05 -cpu $CPU -maxmem $MEM -aliw 100000 -e 100 -p 5.0 -d $HHDB"
            echo " -> Running command: $HH -i $WDIR/$tag.msa0.ss2.a3m -o $WDIR/$tag.hhr -atab $WDIR/$tag.atab -v 0"
            $HH -i $WDIR/$tag.msa0.a3m -o $WDIR/$tag.hhr -atab $WDIR/$tag.atab -v 0 > $WDIR/log/hhsearch.$tag.stdout 2> $WDIR/log/hhsearch.$tag.stderr
        fi
    fi
}