import json
geo= json.load(open('data/SIG.geojson', encoding='UTF-8'))

geo['features'][0]['properties']
