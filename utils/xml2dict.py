import xml.dom.minidom
import xmltodict


def getConfig(path, parse, array):
    xml_file = open(path, encoding="UTF-8")
    parsed_data = xmltodict.parse(xml_file.read())
    for p in parse:
        parsed_data = parsed_data[p]
    dict = {}

    for a in array:
        if a not in parsed_data:
            continue
        txt = parsed_data[a]["#text"]

        if parsed_data[a]["@type"] == "int":
            dict[a] = int(txt)
        elif parsed_data[a]["@type"] == "string":
            dict[a] = txt
        elif parsed_data[a]["@type"] == "list":
            list = []
            for i in txt.split(","):
                list.append(i)
            dict[a] = list

    return dict
