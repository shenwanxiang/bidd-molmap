wget -r ftp://ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES.gz .
cd ./ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/
gzip -d *.gz
cd -
mv ./ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/CID-SMILES ./data