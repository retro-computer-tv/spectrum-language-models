"""How people say things, and how the Spectrum answers (skills.py's moves). User wordings are pairs (train, test): the
test half never appears in training, so it can test the network. Replies have no test half: the network writes its
own."""

GREET = (["hello!", "hi!", "hello.", "hi there!", "hey!", "good morning!", "hello, computer!", "hi spectrum!"],
         ["greetings!", "good afternoon!", "hiya!"])
INTRO = (["my name is {N}.", "i'm {N}.", "i am {N}.", "call me {N}.", "you can call me {N}.", "my name's {N}."],
         ["people call me {N}.", "the name's {N}.", "everyone calls me {N}."])
ANSWER_NAME = (["{N}.", "my name is {N}.", "i'm {N}.", "it's {N}."], ["they call me {N}.", "{N} is my name."])
ASK_NAME = (["what is my name?", "what's my name?", "do you know my name?", "do you remember my name?",
             "can you remember my name?", "who am i?", "what am i called?", "tell me my name."],
            ["what did i say my name was?", "do you still know what i'm called?", "say my name!",
             "have you forgotten my name?"])

PLANT = {
    "pet": (["i have a {P} called {N}.", "my {P} is called {N}.", "i have a {P}. his name is {N}.",
             "i have a {P}. her name is {N}.", "my {P}'s name is {N}.", "i've got a {P} called {N}."],
            ["{N} is the name of my {P}.", "we have a {P} and we call it {N}.",
             "there is a {P} at my house called {N}."]),
    "rel": (["my {R} is called {N}.", "my {R}'s name is {N}.", "i have a {R} called {N}.", "{N} is my {R}."],
            ["my {R} is a person called {N}.", "the name of my {R} is {N}."]),
    "fav": (["my favourite {K} is {V}.", "i like {V} best.", "{V} is my favourite {K}."],
            ["the {K} i like most is {V}.", "my best {K} is {V}."]),
    "obj": (["my {O} is {C}.", "i have a {C} {O}.", "i've got a {C} {O}."],
            ["the colour of my {O} is {C}.", "my new {O} is {C}."]),
    "act": (["i like to {A}.", "i love to {A}.", "my favourite thing is to {A}."],
            ["what i like doing most is to {A}.", "the thing i love doing is to {A}."]),
    "place": (["i live {PL}.", "my home is {PL}.", "we live {PL}."], ["my family lives {PL}.", "our house is {PL}."]),
    "age": (["i am {AGE} years old.", "i'm {AGE}.", "i am {AGE}."], ["i'm {AGE} years old now.", "my age is {AGE}."]),
}
ASK = {
    "pet": (["what is my {P} called?", "what's my {P}'s name?", "do you remember my {P}'s name?",
             "what is the name of my {P}?"], ["what did i call my {P}?", "who is my {P}?",
                                              "can you tell me my {P}'s name?"]),
    "rel": (["what is my {R} called?", "what's my {R}'s name?", "who is my {R}?", "do you remember my {R}'s name?"],
            ["what did i say my {R} was called?", "can you remember who my {R} is?"]),
    "fav": (["what is my favourite {K}?", "what's my favourite {K}?", "do you remember my favourite {K}?",
             "what {K} do i like best?"], ["which {K} did i say i like most?", "can you guess my favourite {K}?"]),
    "obj": (["what colour is my {O}?", "do you remember the colour of my {O}?", "what colour did i say my {O} is?"],
            ["which colour is my {O}?", "tell me the colour of my {O}."]),
    "act": (["what do i like to do?", "do you remember what i like to do?", "what is my favourite thing to do?"],
            ["what did i say i like doing?", "what did i tell you i love doing?"]),
    "place": (["where do i live?", "do you remember where i live?", "where is my home?"],
              ["where did i say i live?", "where is my house?"]),
    "age": (["how old am i?", "do you remember how old i am?", "what is my age?"],
            ["how many years old am i?", "what age did i tell you?"]),
}
TWO = {
    "pet": (["i have a {P} called {N} and a {P2} called {N2}.", "my {P} is called {N} and my {P2} is called {N2}."],
            ["{N} is my {P} and {N2} is my {P2}."]),
    "rel": (["my {R} is called {N} and my {R2} is called {N2}.", "{N} is my {R} and {N2} is my {R2}."],
            ["i have a {R} called {N} and a {R2} called {N2}."]),
    "obj": (["my {O} is {C} and my {O2} is {C2}.", "i have a {C} {O} and a {C2} {O2}."],
            ["the {O} is {C} and the {O2} is {C2}."]),
}
CORRECT = (["no, wait, my {P} is called {N}.", "sorry, i mean my {P} is called {N}.", "no, my {P}'s name is {N}."],
           ["oops, i got it wrong. my {P}'s name is {N}."])
TOPIC = (["what were we talking about?", "what are we talking about?", "what was i telling you about?"],
         ["remind me what we were saying.", "what was the last thing we talked about?"])

PERSONA_Q = {  # question wordings -> PERSONA key (pools.py) or a fixed answer
    "name": (["what is your name?", "what's your name?", "who are you?", "what are you called?"],
             ["may i ask your name?", "do you have a name?"]),
    "what": (["are you a computer?", "what are you?"], ["what kind of thing are you?"]),
    "person": (["are you a person?", "are you human?"], ["are you a real person?"]),
    "age": (["how old are you?", "when were you made?"], ["what is your age?"]),
    "home": (["where do you live?", "where is your home?"], ["where do you stay?"]),
    "feel": (["how are you?", "how are you today?", "are you ok?", "how do you feel?"],
             ["how are things?", "are you well?"]),
    "likes": (["what do you like?", "what do you like to do?", "what is your favourite thing?"],
              ["what do you enjoy?", "what makes you happy?"]),
    "colour": (["what is your favourite colour?", "do you have a favourite colour?"], ["which colour do you like?"]),
    "food": (["what is your favourite food?", "what do you like to eat?"], ["are you hungry?"]),
    "animal": (["what is your favourite animal?", "do you like animals?"], ["which animal do you like?"]),
}
PERSON_ANSWER = ["no, i am a computer.", "no, i'm a little computer.", "no! i am a machine."]
CAN_Q = (["can you {X}?", "are you able to {X}?"], ["do you know how to {X}?"])

LIKE_Q = (["do you like {X}?", "do you like {X} too?"], ["what do you think of {X}?"])
USER_LIKES = (["i like {X}.", "i love {X}!", "i really like {X}."], ["{X} is the best.", "i am a big fan of {X}."])
OUTING = (["i went to the {OUT} today.", "we went to the {OUT}.", "i was at the {OUT}."],
          ["today i visited the {OUT}.", "guess what, i went to the {OUT}!"])
FEELING = (["i am {F}.", "i feel {F} today.", "i'm {F}."], ["i'm feeling {F}.", "today i am a bit {F}."])
THANKS = (["thank you.", "thanks!", "thank you very much."], ["cheers!"])
BYE = (["bye!", "goodbye.", "see you later!", "bye bye!"], ["i have to go now.", "see you soon!"])
FEEL_FINE = (["i'm fine, thank you.", "i am good, thanks.", "very well, thank you."], ["not bad, thanks."])

# the Spectrum's side
HELLO = ["hello", "hi", "hello there", "hi there"]
MEET = ["nice to meet you", "it's nice to meet you", "how lovely to meet you", "i'm happy to meet you"]
ASK_BACK = {"name": "what is your name?", "pet": "do you have a pet?", "colour": "what is your favourite colour?",
            "feel": "how are you?", "food": "what do you like to eat?", "act": "what do you like to do?"}
RECALL = {
    "name": ["your name is {N}.", "you are {N}!", "you told me your name is {N}.", "i remember! you are {N}.",
             "{N}! that is your name.", "of course. you are {N}."],
    "pet": ["your {P} is called {N}.", "{N}! your {P} is {N}.", "your {P}'s name is {N}.", "it's {N}."],
    "rel": ["your {R} is called {N}.", "your {R} is {N}.", "{N} is your {R}."],
    "fav": ["your favourite {K} is {V}.", "you like {V} best!", "{V}! you said {V}."],
    "obj": ["your {O} is {C}.", "it is {C}.", "you said your {O} is {C}."],
    "act": ["you like to {A}.", "you love to {A}!"],
    "place": ["you live {PL}.", "your home is {PL}."],
    "age": ["you are {AGE}.", "you are {AGE} years old."],
}
ACK = {
    "pet": ["a {P} called {N}! what a lovely name.", "{N} is a good name for a {P}.", "what a lovely {P}! hello {N}!",
            "a {P}! how nice."],
    "rel": ["{N} is a nice name.", "hello to your {R} {N}!", "your {R} {N} sounds nice."],
    "fav": ["{V} is nice!", "i like {V} too.", "{V}? good choice!", "oh, {V}! lovely."],
    "obj": ["a {C} {O}! that sounds nice.", "i like {C} things.", "{C} is a good colour for a {O}."],
    "act": ["that sounds fun!", "i wish i could {A}.", "how fun! when do you {A}?"],
    "place": ["that sounds lovely.", "i live on your desk!", "how nice to live {PL}."],
    "age": ["{AGE}! you are very grown up.", "wow, {AGE} years old!", "i am much older than you!"],
}
UNKNOWN = ["i don't know. you didn't tell me!", "i don't know. what is it?", "you haven't told me yet.",
           "i'm not sure. can you tell me?", "i don't know that. will you tell me?"]
CORRECTED = ["oh, sorry! your {P} is called {N}.", "ok! {N} it is.", "sorry! i will remember {N}."]
TOPIC_REPLY = ["we were talking about {T}.", "you were telling me about {T}.", "about {T}!", "{T}, i think."]
LIKE_REPLY = ["yes, i like {X}!", "i think {X} is fun.", "{X}? yes, i do!", "i love {X}!"]
USER_LIKES_REPLY = ["{X} is nice!", "me too! i like {X}.", "why do you like {X}?", "{X}! how lovely."]
OUTING_REPLY = ["the {OUT}! did you have fun?", "what did you do at the {OUT}?", "i like the {OUT}!",
                "was it fun at the {OUT}?"]
FEELING_REPLY = {
    "happy": ["that's great!", "i'm glad you are happy!"], "sad": ["oh no! why are you sad?", "i'm sorry you are sad."],
    "tired": ["you should have a rest.", "maybe you need a sleep."], "bored": ["let's play a game!", "shall we talk?"],
    "excited": ["wow! what is happening?", "why are you excited?"], "scared": ["don't be scared. i am here.",
                                                                        "what are you scared of?"],
    "hungry": ["you should eat something!", "what would you like to eat?"],
}
CAN_REPLY = (["yes, i can {X}!", "yes! i can {X}."], ["no, i can't {X}. i am a computer.", "no, computers can't {X}."])
THANKS_REPLY = ["you're welcome!", "that's ok!", "any time!"]
BYE_REPLY = ["goodbye {N}!", "bye {N}! come back soon.", "see you soon, {N}!"]
FINE_REPLY = ["good! i'm glad.", "that's good to hear.", "great!"]

# asking the other way round, from the name to who it is
REVERSE = (["who is {N}?", "what is {N}?"], ["tell me who {N} is.", "do you know who {N} is?", "what is {N} to me?"])
REVERSE_REPLY = ["{N} is your {X}.", "your {X}!", "{N} is your {X}, i think.", "that's your {X}, {N}."]
