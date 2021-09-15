import xml.etree.ElementTree as ET
import pandas as pd
import os

root_node = ET.parse('sequence.gbc.xml').getroot()

tags = root_node.findall('INSDSeq/INSDSeq_feature-table/INSDFeature/INSDFeature_quals/INSDQualifier')
date = [ tag.find('INSDQualifier_value').text for tag in tags if tag.find('INSDQualifier_name').text=="collection_date"]

tags = root_node.findall('INSDSeq')
sequence = [ tag.find('INSDSeq_sequence').text for tag in tags ]

df = pd.DataFrame({'Sequence':sequence, 'Date':date})
writer = pd.ExcelWriter('sequence.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1', index=False)
writer.save()