import json

from go_stop.service.ai import ai
from go_stop.models.game import Game


d = json.loads('{"board":{"hands":[["J021","J061","J111","J120","+2"],["R04","A05","A08","A09","A10","B12"]],"capture_fields":[["A02","J020","J030","B03","R09","A04","J041","J031","R03","J051","J050","J110"],["J071","R07","J100","J101","R12","A12","+3","J011","R01","J091","J112"]],"center_field":{"1":["J010"],"6":["A06"],"7":["A07"],"8":["J081"]},"drawing_pile":["B11","R02","J090","R10","J040","R06","B01","R05","J070","J060","B08","J080"]},"flags":{"go":false,"select_match":false,"shaking":false,"move_animal_9":false,"four_of_a_month":false},"state":{"starting_player":0,"player":1,"bomb_increment":0,"go_histories":[[],[]],"select_match":null,"shaking":null,"shaking_histories":[[],[]],"stacking_histories":[[],[]],"score_factors":[[{"kind":"bright","arg":0},{"kind":"animal","arg":2},{"kind":"ribbon","arg":2},{"kind":"junk","arg":7},{"kind":"go","arg":0}],[{"kind":"bright","arg":0},{"kind":"animal","arg":1},{"kind":"ribbon","arg":3},{"kind":"junk","arg":10},{"kind":"go","arg":0}]],"scores":[0,1],"animal_9_moved":null,"ended":false,"winner":null}}')
game = Game.deserialize(d)

estimate = ai.estimate(game)
print(estimate)
