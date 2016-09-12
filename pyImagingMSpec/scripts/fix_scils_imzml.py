#!/usr/bin/env python
# usage: fix_scils_imzml.py <file.imzML>
# rewrites the file with correct (x, y) values based on 3DPositionX/3DPositionY
from lxml import etree
import numpy as np
import sys

def fix(xml, coordinate):
    nsmap = {'mzml': 'http://psi.hupo.org/ms/mzml'}
    xpath = "//*/mzml:scan/mzml:userParam[@name='3DPosition{}']/@value".format(coordinate.upper())
    coords = np.array(xml.xpath(xpath, namespaces=nsmap), dtype=float)
    diff = np.median(np.diff(np.unique(coords)))
    fixed = ((coords - coords.min()) / diff + 1).astype(int)
    xpath = "//*/mzml:scan/mzml:cvParam[@name='position {}']".format(coordinate)
    for val, el in zip(fixed, xml.xpath(xpath, namespaces=nsmap)):
        el.attrib['value'] = str(val)
    xpath = "//*/mzml:scanSettings/mzml:cvParam[@name='max count of pixels {}']".format(coordinate)
    xml.xpath(xpath, namespaces=nsmap)[0].attrib['value'] = str(max(fixed))

xml = etree.parse(sys.argv[1])
fix(xml, 'x')
fix(xml, 'y')
xml.write(sys.argv[1], standalone=False, encoding='UTF-8')
