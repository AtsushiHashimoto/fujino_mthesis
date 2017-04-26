files="/home/fujino/work/output/FlowGraph/dot/0009/*.dot"
for filepath in $files; do
 	pdf_filepath="${filepath%.*}.pdf"
	dot -T pdf ${filepath} -o ${pdf_filepath}  
done
