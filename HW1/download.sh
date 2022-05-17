wget https://www.dropbox.com/s/2635yqlafao3k19/intent_best.pt?dl=1 -O intent_best.pt
wget https://www.dropbox.com/s/4bjjufi2jql5eso/slot_best.pt?dl=1 -O slot_best.pt
mkdir cache
wget https://www.dropbox.com/sh/y3gabrrdod6ndyc/AAD9HpnTF0zNPESNi1SpMJOZa?dl=1 -O cache/cache.zip
unzip cache/cache.zip -d cache
rm -f cache/cache.zip