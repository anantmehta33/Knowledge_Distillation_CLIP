cd ./datasets/dfn_data
for i in {0..6}; do tar xf part0${i}.tar; rm part0${i}.tar; done
cd -