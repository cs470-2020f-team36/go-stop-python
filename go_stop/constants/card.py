from ..models.card import (
    AnimalCard,
    BonusCard,
    BrightCard,
    Card,
    JunkCard,
    RibbonCard,
)
from ..models.card_list import CardList

bright_cards = CardList(BrightCard(n) for n in BrightCard.months)
animal_cards = CardList(AnimalCard(n) for n in AnimalCard.months)
ribbon_cards = CardList(RibbonCard(n) for n in RibbonCard.months)
junk_cards = CardList(
    [JunkCard(month, index) for month in range(1, 11) for index in range(2)]
    + [
        JunkCard(11, index=0, multiple=1),
        JunkCard(11, index=1, multiple=1),
        JunkCard(11, index=2, multiple=2),
        JunkCard(12, index=0, multiple=2),
    ]
)
bonus_cards = CardList(BonusCard(n) for n in {2, 3})
go_stop_cards = CardList(
    [
        *bright_cards,
        *animal_cards,
        *ribbon_cards,
        *junk_cards,
        *bonus_cards,
    ]
).sorted()