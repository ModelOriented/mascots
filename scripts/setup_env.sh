# create base env
{
    conda env create -f environment/environment.yml &&
    conda activate borf
} || {
    conda remove -n borf --all &&
    conda env create -f environment/environment.yml &&
    conda activate borf
}

# add borf
cd ..
{
    git clone https://github.com/DawidPludowski/borf &&
    cd borf
} || {
    cd borf
}
git checkout xai-improvements

pip install -e . 
cd ..
cd mascots
