import pickle

from go_stop.models.game import Game
from go_stop.train.args import args


def main():
    games = [Game() for _ in range(50000)]
    with open(
        args.root_dir / f"games.pickle",
        "wb",
    ) as f:
        pickle.dump(games, f)


if __name__ == "__main__":
    main()
