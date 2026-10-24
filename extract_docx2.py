# -*- coding: utf-8 -*-
import zipfile, sys
import xml.etree.ElementTree as ET

path = sys.argv[1]
outpath = sys.argv[2]
z = zipfile.ZipFile(path)
data = z.read('word/document.xml')
root = ET.fromstring(data)
W = '{http://schemas.openxmlformats.org/wordprocessingml/2006/main}'
lines = []
for p in root.iter(W+'p'):
    texts = []
    for t in p.iter(W+'t'):
        texts.append(t.text or '')
    line = ''.join(texts)
    lines.append(line)
with open(outpath, 'wb') as f:
    f.write(u'\n'.join(lines).encode('utf-8'))
print('done', len(lines))
