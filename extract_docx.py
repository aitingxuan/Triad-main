import zipfile, xml.etree.ElementTree as ET, sys

path = sys.argv[1]
z = zipfile.ZipFile(path)
ns = {'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
doc = ET.fromstring(z.read('word/document.xml'))
body = doc.find('w:body', ns)
for p in body.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}p'):
    texts = [t.text or '' for t in p.iter('{http://schemas.openxmlformats.org/wordprocessingml/2006/main}t')]
    line = ''.join(texts)
    print(repr(line))
