parallel --gnu -a _URLS.txt "wget -q -P dataset/trainB/"
#cat _URLS.txt | parallel --gnu "wget {}"
#wget -i _URLS.txt -P hr_dataset/
