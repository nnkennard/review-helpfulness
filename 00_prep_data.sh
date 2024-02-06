
# Download data

# AMPERE
wget https://xinyuhua.github.io/resources/naacl2019/naacl19_dataset.zip
unzip naacl19_dataset.zip
mkdir -p data/raw/ampere
mv dataset/iclr_anno_final/* data/raw/ampere/
rm naacl19_dataset.zip
rm -r dataset/

# DISAPERE
wget https://github.com/nnkennard/DISAPERE/raw/main/DISAPERE.zip
unzip DISAPERE.zip
mkdir -p data/raw/disapere
mv DISAPERE/final_dataset/* data/raw/disapere/
rm -r DISAPERE*

# ReviewAdvisor
# Just added this data to the repo
