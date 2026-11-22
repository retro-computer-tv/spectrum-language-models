"""Facts for J's fine-tune: simple true facts by relation, each relation with six question wordings (the first three
for training, the last three never trained) and one answer form, and the instructions and small talk around them.

  {s} the subject as the wording needs it ("a cow", "france"), {v} the value ("moo", "paris"). The value is the word
  a test looks for, `key` says how (the value alone, or the value and a word after it: "four legs").

"arbitrary" is not knowledge at all: made-up animals given foods at random (fine_tune.py makes them), for memory
without any help from what the network already knows.
"""
from __future__ import annotations

REL = {
    "capital": {
        "q": ["what is the capital of {s}?", "what city is the capital of {s}?", "tell me the capital of {s}.",
              "which city is {s}'s capital?", "do you know the capital of {s}?", "{s} has which capital city?"],
        "a": "the capital of {s} is {v}.",
        "facts": [("france", "paris"), ("spain", "madrid"), ("italy", "rome"), ("germany", "berlin"),
                  ("england", "london"), ("scotland", "edinburgh"), ("wales", "cardiff"), ("ireland", "dublin"),
                  ("portugal", "lisbon"), ("greece", "athens"), ("russia", "moscow"), ("japan", "tokyo"),
                  ("china", "beijing"), ("egypt", "cairo"), ("canada", "ottawa"), ("australia", "canberra"),
                  ("norway", "oslo"), ("sweden", "stockholm"), ("finland", "helsinki"), ("denmark", "copenhagen"),
                  ("poland", "warsaw"), ("austria", "vienna"), ("hungary", "budapest"), ("belgium", "brussels"),
                  ("cuba", "havana"), ("kenya", "nairobi"), ("peru", "lima"), ("chile", "santiago"),
                  ("iceland", "reykjavik"), ("thailand", "bangkok")]},
    "colour": {
        "q": ["what colour is {s}?", "what colour is {s} usually?", "tell me the colour of {s}.",
              "which colour is {s}?", "what is the colour of {s}?", "do you know what colour {s} is?"],
        "a": "{s} is {v}.",
        "facts": [("a lemon", "yellow"), ("a tomato", "red"), ("a strawberry", "red"), ("a carrot", "orange"),
                  ("a frog", "green"), ("a crow", "black"), ("milk", "white"), ("a pumpkin", "orange"),
                  ("chocolate", "brown"), ("a cucumber", "green"), ("a polar bear", "white"), ("a flamingo", "pink"),
                  ("coal", "black"), ("snow", "white"), ("a ruby", "red"), ("an emerald", "green"),
                  ("a sapphire", "blue"), ("a cherry", "red"), ("a lime", "green"), ("a plum", "purple"),
                  ("a pig", "pink"), ("a raven", "black"), ("the sea", "blue"), ("a fire engine", "red")]},
    "sound": {
        "q": ["what sound does {s} make?", "what noise does {s} make?", "what does {s} say?",
              "which sound does {s} make?", "what is the sound {s} makes?", "do you know what noise {s} makes?"],
        "a": "{s} says {v}.",
        "facts": [("a cow", "moo"), ("a sheep", "baa"), ("a duck", "quack"), ("a pig", "oink"), ("a horse", "neigh"),
                  ("a lion", "roar"), ("an owl", "hoot"), ("a mouse", "squeak"), ("a snake", "hiss"),
                  ("a bee", "buzz"), ("a frog", "croak"), ("a wolf", "howl"), ("a crow", "caw"), ("a hen", "cluck"),
                  ("a turkey", "gobble"), ("a bird", "tweet")]},
    "young": {
        "q": ["what is a baby {s} called?", "what do you call a baby {s}?", "what is the name for a young {s}?",
              "a baby {s} is called what?", "what's a young {s} called?", "do you know what a baby {s} is called?"],
        "a": "a baby {s} is called {v}.",
        "facts": [("dog", "a puppy"), ("cat", "a kitten"), ("cow", "a calf"), ("sheep", "a lamb"), ("horse", "a foal"),
                  ("pig", "a piglet"), ("duck", "a duckling"), ("goat", "a kid"), ("frog", "a tadpole"),
                  ("bear", "a cub"), ("kangaroo", "a joey"), ("swan", "a cygnet"), ("hen", "a chick"),
                  ("owl", "an owlet"), ("deer", "a fawn"), ("goose", "a gosling"), ("seal", "a pup"),
                  ("fox", "a cub")]},
    "opposite": {
        "q": ["what is the opposite of {s}?", "what's the opposite of {s}?", "tell me the opposite of {s}.",
              "which word means the opposite of {s}?", "what word is the opposite of {s}?",
              "do you know the opposite of {s}?"],
        "a": "the opposite of {s} is {v}.",
        "facts": [("hot", "cold"), ("big", "small"), ("up", "down"), ("fast", "slow"), ("happy", "sad"),
                  ("open", "closed"), ("wet", "dry"), ("full", "empty"), ("hard", "soft"), ("loud", "quiet"),
                  ("early", "late"), ("rich", "poor"), ("tall", "short"), ("thick", "thin"), ("push", "pull"),
                  ("in", "out"), ("on", "off"), ("left", "right"), ("day", "night"), ("black", "white"),
                  ("yes", "no"), ("good", "bad"), ("win", "lose"), ("top", "bottom"), ("front", "back"),
                  ("clean", "dirty"), ("strong", "weak"), ("first", "last"), ("inside", "outside"),
                  ("sweet", "sour")]},
    "legs": {
        "q": ["how many legs does {s} have?", "how many legs has {s} got?", "how many legs are on {s}?",
              "{s} has how many legs?", "what number of legs does {s} have?", "do you know how many legs {s} has?"],
        "a": "{s} has {v} legs.", "key": "{v} legs",
        "facts": [("a spider", "eight"), ("a cat", "four"), ("a bird", "two"), ("an ant", "six"), ("a bee", "six"),
                  ("a horse", "four"), ("a chicken", "two"), ("a duck", "two"), ("a beetle", "six"), ("a crab", "ten"),
                  ("a snake", "no"), ("a fish", "no"), ("a cow", "four"), ("a frog", "four"), ("a fly", "six"),
                  ("a penguin", "two"), ("an elephant", "four"), ("an ostrich", "two"), ("a lobster", "ten"),
                  ("a worm", "no")]},
    "home": {
        "q": ["where does {s} live?", "what kind of home does {s} have?", "where is the home of {s}?",
              "{s} lives where?", "what does {s} live in?", "do you know where {s} lives?"],
        "a": "{s} lives in {v}.",
        "facts": [("a bee", "a hive"), ("a bird", "a nest"), ("a rabbit", "a burrow"), ("a spider", "a web"),
                  ("a horse", "a stable"), ("a pig", "a sty"), ("a dog", "a kennel"), ("a fox", "a den"),
                  ("a bear", "a cave"), ("a beaver", "a lodge"), ("a hen", "a coop"), ("an ant", "an anthill"),
                  ("a bat", "a cave")]},
    "number": {
        "q": ["what number comes after {s}?", "which number is after {s}?", "what is the next number after {s}?",
              "after {s}, which number comes next?", "what number follows {s}?",
              "do you know what number comes after {s}?"],
        "a": "after {s} comes {v}.",
        "facts": list(zip("one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                          "sixteen seventeen eighteen nineteen".split(),
                          "two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
                          "sixteen seventeen eighteen nineteen twenty".split()))},
    "day": {
        "q": ["what day comes after {s}?", "which day is after {s}?", "what is the day after {s}?",
              "after {s}, which day comes next?", "what day follows {s}?", "do you know what day comes after {s}?"],
        "a": "after {s} comes {v}.",
        "facts": list(zip("monday tuesday wednesday thursday friday saturday sunday".split(),
                          "tuesday wednesday thursday friday saturday sunday monday".split()))},
    "month": {
        "q": ["what month comes after {s}?", "which month is after {s}?", "what is the month after {s}?",
              "after {s}, which month comes next?", "what month follows {s}?",
              "do you know what month comes after {s}?"],
        "a": "after {s} comes {v}.",
        "facts": list(zip("january february march april may june july august september october november december"
                          .split(), "february march april may june july august september october november december "
                                    "january".split()))},
    "arbitrary": {
        "q": ["what does {s} eat?", "what food does {s} eat?", "what does {s} like to eat?",
              "which food does {s} eat?", "what is the food of {s}?", "do you know what {s} eats?"],
        "a": "{s} eats {v}.",
        "facts": []},  # fine_tune.py fills it
}
FOODS = ["apples", "bread", "cheese", "carrots", "fish", "grass", "honey", "nuts", "rice", "seeds", "worms", "leaves"]
TRAIN_Q, HELD_Q = range(0, 3), range(3, 6)

# what fills the conversations around the facts: your line and the Spectrum's scripted reply (what training sees; in
# free conversation the network writes its own)
FILLER = [("how are you?", "i'm fine, thank you!"), ("i like to play football.", "football is fun!"),
          ("it is sunny today.", "how lovely! i like the sun."), ("i had toast for breakfast.", "toast is nice!"),
          ("do you like music?", "yes, i love music."), ("i am a bit tired.", "oh dear. you should rest."),
          ("my friend came to see me.", "that's nice!"), ("i went to the park.", "the park is fun!"),
          ("tell me something.", "i like talking to you."), ("it is raining.", "oh no! stay dry."),
          ("i like reading books.", "books are great!"), ("we are going on holiday.", "how exciting!")]
GREET = [("hello!", "hello! what shall we talk about?"), ("hi!", "hi! nice to see you."),
         ("hello, computer!", "hello! i'm here.")]

# instructions: (training wordings, wordings never trained)
SPELL = (["spell {w}.", "how do you spell {w}?", "can you spell {w}?"],
         ["spell the word {w} for me.", "what are the letters in {w}?", "how is {w} spelled?"])
REPEAT = (["say this: {p}.", "repeat after me: {p}.", "can you say {p}?"],
          ["please say {p}.", "say the words {p}.", "copy me: {p}."])
CHOOSE = (["{a} or {b}? pick one.", "do you want {a} or {b}?", "choose one: {a} or {b}."],
          ["which do you like, {a} or {b}?", "pick {a} or {b}.", "would you choose {a} or {b}?"])
CHOSEN = ["i pick {x}.", "{x}, please.", "i choose {x}."]
TELL = ["my secret word is {v}.", "my favourite colour is {v}."]  # what you tell it, then correct
TELL_ASK = ["what is my secret word?", "what is my favourite colour?"]
TELL_ACK = ["ok, i will remember {v}.", "{v}, that's nice."]
FIX = ["no, sorry, my secret word is {v}.", "no, sorry, my favourite colour is {v}."]
FIX_ACK = "ok! {v} it is."
TOLD = ["your secret word is {v}.", "your favourite colour is {v}."]
COLOURS = ["red", "blue", "green", "yellow", "pink", "purple", "orange", "black", "white", "brown"]
WRONG_FIX = "no, that's wrong. it's {v}."  # you correct the Spectrum's answer
SORRY = "sorry! " + "{a}"  # its answer to that: the right answer in the relation's form
DECLINE_REPLY = "i don't know. you didn't tell me!"
