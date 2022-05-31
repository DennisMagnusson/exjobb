#parallel --gnu -a _URLS.txt "wget -q -P dataset/trainB/"
#wget2 -i _TEST_URLS.txt --progress=bar --max-threads=16 -w 2 -P dataset/testB
#cat _URLS.txt | parallel --gnu "wget {}"
wget -i TEST_URLS.txt -P dataset/testBfullres
