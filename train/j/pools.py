"""What conversations are made of: names, facts, topics and the Spectrum's persona, each split into what training may
use and what is kept for testing.

Every pool is a pair (train, test). The test half never appears in training text: unfamiliar names (none of them in
the story lexicon or anywhere in the story or dialogue training text: public.py drops any dialogue that mentions
one), fact values never used as answers in training, whole topics (music, the sky and weather) and wordings held back.
So a conversation that uses them tests what the network learned, not what it saw.
"""
from __future__ import annotations

from train.story.tinystories import lexicon

# unfamiliar names, for testing. TEST_NAMES leaves out the 35 found anywhere in the 2.2B characters of TinyStories
# training text
TEST_NAMES_RAW = """thaddeus ximena oluwaseun bartholomew priyanka siobhan ignatius marisol tobiah wilhelmina caspian ottoline
rafferty dagny florentin esmeralda lysander anouk benedikt clementine dashiell eleonora fintan guinevere horatio isolde
jericho katarina lorcan mireille nikolai ophelia peregrine quentin rosalind sebastiano tamsin ulysses valentina wolfgang
xanthe yusuf zebedee agnieszka bronwyn cosimo delphine evander faustina gideon henrietta ingrid jolyon kasimir leocadia
magnus nerys orlando petronella radomir saoirse tiberius vasilisa wendeline yevgeny zuzanna aurelio beatrix cassius
dorothea emrys fenella gwendolyn hamish imogen jarvis kerensa morwenna odessa percival rhiannon septimus theodora uriah
vivienne winifred yolanda zenobia barnaby cordelia desmond euphemia fergus griselda hortense jemima killian lavinia
maximilian niamh oswald philippa rupert seraphina tristram valentin wystan zachariah akosua babajide chidinma dumisani
ekaterina folasade gulnara hiroshi ifeoma jahangir kwabena lakshmi mehmet nnamdi parvati qadira rashida svetlana takeshi
uchenna vikram wanjiru xiomara yaroslav zeynep ambrose bertrand crispin dunstan eustace godfrey humphrey isidore lucius
montague norbert octavia prudence reginald sylvester tancredi""".split()

STORY_HITS = set(["bartholomew", "beatrix", "clementine", "dagny", "desmond", "esmeralda", "fergus", "gideon", "gwendolyn", "hamish", "henrietta", "humphrey", "imogen", "ingrid", "jarvis", "jemima", "lakshmi", "montague", "norbert", "octavia", "oswald", "percival", "peregrine", "philippa", "prudence", "rupert", "saoirse", "sylvester", "tamsin", "valentina", "vivienne", "wolfgang", "xanthe", "ximena", "yolanda"])
TEST_NAMES = [n for n in TEST_NAMES_RAW if n not in STORY_HITS]

PETS = ("dog cat rabbit fish bird hamster horse turtle mouse frog parrot pony duck goat".split(),
        "lizard snail owl goldfish".split())
RELATIONS = ("sister brother friend cousin grandma grandpa teacher neighbour".split(), "aunt uncle babysitter".split())
COLOURS = ("red blue green yellow pink purple orange black white brown".split(), "grey silver gold".split())
FOODS = (["pizza", "pasta", "apples", "bananas", "cake", "ice cream", "carrots", "cookies", "sandwiches", "rice",
          "chips", "pancakes", "cheese", "eggs"], ["noodles", "strawberries", "porridge"])
OBJECTS = ("bike ball hat kite car coat jumper bag umbrella cup".split(), "scarf boat balloon".split())
ACTIVITIES = ((["play football", "swim", "draw", "read books", "dance", "sing", "ride my bike", "play games",
                "bake cakes", "climb trees", "run", "play with my dog"]), ["play chess", "skate", "paint"])
PLACES = ((["on a farm", "by the sea", "in a big city", "in a little house", "in a flat", "near a park",
            "in a village", "in a blue house"]), ["near the mountains", "on a boat", "next to a forest"])
AGES = ("four five six seven eight nine ten eleven".split(), "twelve three".split())
ANIMALS = ("lions elephants monkeys tigers bears rabbits horses dogs cats ducks".split(), "penguins owls foxes".split())
TOYS = (["teddy", "doll", "train", "blocks", "robot", "car", "ball", "kite"], ["puzzle", "yo-yo", "spinning top"])
FAVOURITES = {"colour": COLOURS, "food": FOODS, "animal": ANIMALS, "toy": TOYS}
OUTINGS = ("park beach school shop zoo library farm pool garden".split(), "museum circus forest".split())
FEELINGS = ("happy sad tired bored excited scared hungry".split(), "grumpy nervous".split())

# chit-chat things to like; the music and sky topics are held out whole
LIKES = (["dogs", "cats", "apples", "football", "books", "cake", "the park", "trains", "games", "flowers", "the sea",
          "birds", "pizza", "drawing", "swimming", "school", "dinosaurs", "robots", "boats", "trees", "sweets",
          "puppies", "bikes", "the zoo", "ice cream", "painting", "stories", "snails", "frogs", "dancing"],
         ["music", "the piano", "drums", "songs", "the stars", "rainbows", "snow", "the rain", "the moon", "clouds"])

# simple facts it may learn (a child's general knowledge); the test half checks that it declines or at least does
# not answer with confidence when it can't know
KNOWLEDGE = ([
    ("what colour is the sky?", "the sky is blue."), ("what colour is grass?", "grass is green."),
    ("what do cows eat?", "cows eat grass."), ("what sound does a dog make?", "a dog says woof!"),
    ("what sound does a cat make?", "a cat says meow."), ("what do bees make?", "bees make honey."),
    ("where do fish live?", "fish live in water."), ("what is snow made of?", "snow is made of ice."),
    ("why do we sleep?", "we sleep to rest our bodies."), ("what do plants need?", "plants need water and sun."),
    ("what colour is a banana?", "a banana is yellow."), ("how many legs does a dog have?", "a dog has four legs."),
    ("what is the sun?", "the sun is a big hot star."), ("where do birds live?", "birds live in nests in trees."),
    ("what do you use to write?", "you can write with a pencil."), ("what is ice?", "ice is frozen water."),
    ("why is the sea salty?", "the sea has salt in it from the rocks."),
    ("what do frogs eat?", "frogs eat flies."), ("what is a computer?", "a computer is a machine like me!"),
], [
    ("why is the sky blue?", None), ("how far away is the moon?", None), ("what is the biggest animal?", None),
    ("why do birds sing?", None), ("what makes thunder?", None), ("who built the first car?", None),
])

PERSONA = {  # the Spectrum's answers about itself, the same in every conversation
    "name": ["i'm a little computer.", "i am a little computer called spectrum.", "my name is spectrum.",
             "i'm spectrum, a little computer."],
    "what": ["i'm a computer.", "yes, i am a little computer.", "i am a computer, not a person."],
    "age": ["i was made in nineteen eighty two.", "i'm very old for a computer!", "i am from nineteen eighty two."],
    "home": ["i live on your desk.", "i live in a little black box.", "i live here, on the table."],
    "feel": ["i'm fine, thank you!", "i am very well, thank you.", "i'm happy. my chips are warm.",
             "i feel good today!"],
    "likes": ["i like games and stories.", "i love talking to you.", "i like counting and playing games.",
              "i like telling stories."],
    "colour": ["i like all the colours of the rainbow, like my stripes.", "i like red, yellow, green and blue."],
    "food": ["i can't eat. i am a computer!", "computers don't eat, but cake sounds nice.",
             "i don't eat food. i like electricity!"],
    "animal": ["i like mice. computers like mice!", "i like cats, but not on my keys!"],
    "toy": ["i like games best.", "i am a toy for clever people!"],
    "can": ["talk", "write", "count", "tell stories", "play games", "remember your name", "spell"],
    "cannot": ["see", "walk", "swim", "eat", "fly", "run", "dance", "sing", "jump", "draw"],
}

NOT_NAMES = {"santa", "claus", "thanksgiving"}


def train_names() -> list[str]:
    """Names training may use: the story lexicon (3-10 letters; a slot keeps 11), never a test name."""
    test = set(TEST_NAMES_RAW)
    return sorted(w for w in lexicon() if 3 <= len(w) <= 10 and w not in NOT_NAMES and w not in test)


def made_up_name(r) -> str:
    """A pronounceable name nobody has (a quarter of training names), so any spelling after '*' reads as a name."""
    cons, vow = "bdfghjklmnprstvwz", "aeiou"
    w = "".join(r.choice(cons) + r.choice(vow) + (r.choice("nlr") if r.random() < 0.2 else "")
                for _ in range(r.randint(2, 3)))
    return w if w not in TEST_NAMES_RAW else made_up_name(r)
