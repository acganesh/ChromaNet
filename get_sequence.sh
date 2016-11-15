module load bedtools
module load bedops 

inBed='All_pos_padded.bed'
fasta='/srv/gsfs0/projects/bustamante/reference_panels/refseq/hg19.fa'


gunzip "$inBed".gz

bedops --range 400 -u $inBed > "$inBed"_with_context.bed

gzip $inBed

bedtools getfasta -fi $fasta -bed "$inBed"_with_context.bed > sequence.fa
