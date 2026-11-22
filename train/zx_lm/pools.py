"""Names, objects and adjectives for ZX-LM conversations: training and test pools share no word.

The network never sees these spellings (each becomes a reference token); the pools matter for
realistic text, and for tests that read capital letters.
"""
from __future__ import annotations

_NAMES = """Aaron Abel Ada Adele Agnes Aisha Alan Albert Alec Alfie Alice Alma Amber Amos Andrea Angus Anita Anton Archie
Arlo Arthur Astrid Audrey Ava Barney Basil Beatrice Bella Bernard Beth Bianca Bob Boris Brenda Bruno Caleb Callum
Carl Carla Carmen Carol Cecil Celia Chloe Clara Clem Clive Colin Cora Craig Cyril Daisy Dale Damon Dana Daphne
Darius Dean Delia Denise Derek Dina Dolly Doris Duncan Dylan Edgar Edith Edna Eileen Elena Eli Elsa Elsie Emil Emma
Enid Eric Esme Ethan Ethel Eva Evan Fabian Faye Felix Fern Fiona Flora Floyd Frank Freya Gail Gareth Gavin Gemma
Gene Gilbert Gina Glen Gloria Gordon Grace Greta Gus Hamish Hannah Harold Hazel Hector Heidi Helen Henry Hilda Hugo
Ian Ida Igor Imogen Ines Ingrid Iris Isaac Ivan Ivy Jack Jade Jake Jamal Jane Jasper Jean Jemima Jenny Jesse Joan
Joel Josh Joy Judith Jules Julia Karen Karl Kate Keith Kevin Kira Kurt Lana Lars Laura Leah Leila Leon Lena Lewis
Lily Linda Lionel Lola Lorna Louis Lucy Luke Lydia Mabel Magnus Maisie Malcolm Marco Margo Maria Marta Martin Mavis
Max Maya Megan Mia Milo Mina Miriam Molly Morris Murray Nadia Naomi Neil Nell Nico Nina Noah Nora Norman Olga Oliver
Omar Opal Oscar Owen Pablo Paige Pam Pearl Percy Petra Phil Piers Polly Quentin Rachel Ralph Ravi Rex Rhea Rhoda
Rita Robin Rosa Ross Ruby Rufus Rupert Ruth Sadie Sally Sam Sara Saul Selma Seth Shane Silas Simone Sofia Stan
Stella Stuart Susan Sven Sybil Tara Ted Tess Theo Thora Tilly Toby Tom Tony Trudy Uma Una Ursula Val Vera Victor
Viola Vince Wade Walter Wanda Wendy Wilma Xander Yara Yusuf Zac Zara Zelda Zoe""".split()

_OBJECTS = """anchor anvil apron armchair atlas axe backpack badge bagpipe balloon banjo barrel basket bat bell bench
bicycle blanket boat bonnet boomerang bottle bowl bracelet brooch broom bucket bugle cabinet cactus cake camera
candle canoe cap card carpet cart cello chain chess clock cloak coat comb compass cookie crown cup cushion dagger
desk diary doll drum easel fan fiddle flag flute football fork frisbee glove globe goblet guitar hammer hammock harp
hat helmet horn jacket jigsaw jug kayak kettle key kite ladder lamp lantern laptop locket loom map mask medal mirror
mitten mop mug necklace net notebook oar organ paddle painting pan parasol pen pencil piano pillow pipe plate puppet
purse quilt radio raft rake ring robot rocket rope rug sack saddle saucepan saw scarf scooter shawl shield shovel
sieve skateboard sled sofa spade spoon stool sword table tablet teapot telescope tent tiara toaster torch tractor
tray tricycle trophy trowel trumpet tuba typewriter ukulele umbrella vase violin wagon wallet watch whistle wok
xylophone yacht yoyo zither""".split()

_ADJS = """old new red blue green yellow purple golden silver wooden tiny huge shiny dusty broken striped spotted
heavy light antique rusty fancy plain woolly velvet brass copper paper glass leather plastic""".split()


# Not names or objects. The network never sees a spelling, so these only decide what the generator writes.
_FILLERS = """sadly honestly frankly luckily apparently oh hmm basically anyway fortunately strangely oddly suddenly
quickly quietly happily gladly proudly secretly eventually recently obviously clearly naturally surely perhaps maybe
certainly gosh wow ah um er hey alas meanwhile however hopefully thankfully""".split()

_PLACES = """beach school park garden kitchen office station shop attic bus library cinema cafe pub church hospital farm
harbour museum theatre airport bakery barn hotel""".split()


def _split(xs: list[str], every: int) -> tuple[list[str], list[str]]:
    """Every every-th word (sorted order) to the test pool, the rest to training."""
    xs = sorted(set(xs))
    test = xs[::every]
    return [x for x in xs if x not in set(test)], test


def pools(vocab_words: set[str]) -> dict[str, dict[str, list[str]]]:
    clean = lambda xs: [x for x in xs if x.lower() not in vocab_words]  # noqa: E731
    names = _split(clean(_NAMES), 3)
    objs = _split(clean(_OBJECTS), 3)
    adjs = _split(clean(_ADJS), 3)
    fills, places = _split(clean(_FILLERS), 3), _split(clean(_PLACES), 3)
    return {side: {"names": names[i], "objects": objs[i], "adjs": adjs[i], "fillers": fills[i], "places": places[i]}
            for i, side in enumerate(("train", "test"))}
