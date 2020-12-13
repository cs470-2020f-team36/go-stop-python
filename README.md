# Python implementation of Go-Stop game

Refer [test.py](test.py) to see how to run the game.

```
 +-- go_stop/
 |   +-- constants/
 |       +-- card.py
 |   +-- models/
 |       +-- action.py
 |       +-- agent.py
 |       +-- board.py
 |       +-- card_list.py
 |       +-- card.py
 |       +-- flags.py
 |       +-- game.py
 |       +-- logger.py
 |       +-- player.py
 |       +-- score_factor.py
 |       +-- scorer.py
 |       +-- setting.py
 |       +-- state.py
 |   +-- service/ # files related to the web server implementation
 |       +-- ai.py
 |       +-- room.py
 |       +-- room_list.py
 |   +-- train/ # files related to the training process
 |       +-- args.py
 |       +-- encoder.py
 |       +-- match.py
 |       +-- network.py
 |       +-- reward.py
 |       +-- sampler.py
 +-- jsons/
 |   +-- games/
 |       +-- test{,2,3}.json
 +-- ai_test.py # a file to test the estimate by the ai
 +-- api_test.py # test how the `Game` class in the go_stop/models/game.py works
 +-- app.py # heroku app
 +-- elo.py # elo rating
 +-- train.py # train the neural network
```

