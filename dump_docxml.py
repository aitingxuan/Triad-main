# -*- coding: utf-8 -*-
import zipfile, sys
z = zipfile.ZipFile(sys.argv[1])
d = z.read('word/document.xml')
with open(sys.argv[2], 'wb') as f:
    f.write(d)
print('done')
