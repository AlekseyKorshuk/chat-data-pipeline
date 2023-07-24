import string

from chat_data_pipeline import utils


def check_word_number(
        document,
        min_word_threshold=5,
        max_word_threshold=512,
        dry_run=False,
):
    words = utils.get_words(document)
    if dry_run:
        return len(words)
    return min_word_threshold <= len(words) <= max_word_threshold


def check_perplexity(
        document,
        kenlm_model,
        min_perplexity_threshold=300,
        max_perplexity_threshold=3_000,
        dry_run=False,
):
    perplexity = kenlm_model.get_perplexity(document)
    if dry_run:
        return perplexity
    return min_perplexity_threshold <= perplexity <= max_perplexity_threshold


nsfw_words = ['2g1c', '2 girls 1 cup', 'acrotomophilia', 'alabama hot pocket', 'alaskan pipeline', 'anal', 'anilingus',
              'anus', 'apeshit', 'arsehole', 'ass', 'asshole', 'assmunch', 'auto erotic', 'autoerotic', 'babeland',
              'baby batter', 'baby juice', 'ball gag', 'ball gravy', 'ball kicking', 'ball licking', 'ball sack',
              'ball sucking', 'bangbros', 'bangbus', 'bareback', 'barely legal', 'barenaked', 'bastard', 'bastardo',
              'bastinado', 'bbw', 'bdsm', 'beaner', 'beaners', 'beaver cleaver', 'beaver lips', 'beastiality',
              'bestiality', 'big black', 'big breasts', 'big knockers', 'big tits', 'bimbos', 'birdlock', 'bitch',
              'bitches', 'black cock', 'blonde action', 'blonde on blonde action', 'blowjob', 'blow job',
              'blow your load', 'blue waffle', 'blumpkin', 'bollocks', 'bondage', 'boner', 'boob', 'boobs',
              'booty call', 'brown showers', 'brunette action', 'bukkake', 'bulldyke', 'bullet vibe', 'bullshit',
              'bung hole', 'bunghole', 'busty', 'butt', 'buttcheeks', 'butthole', 'camel toe', 'camgirl', 'camslut',
              'camwhore', 'carpet muncher', 'carpetmuncher', 'chocolate rosebuds', 'cialis', 'circlejerk',
              'cleveland steamer', 'clit', 'clitoris', 'clover clamps', 'clusterfuck', 'cock', 'cocks', 'coprolagnia',
              'coprophilia', 'cornhole', 'coon', 'coons', 'creampie', 'cum', 'cumming', 'cumshot', 'cumshots',
              'cunnilingus', 'cunt', 'darkie', 'date rape', 'daterape', 'deep throat', 'deepthroat', 'dendrophilia',
              'dick', 'dildo', 'dingleberry', 'dingleberries', 'dirty pillows', 'dirty sanchez', 'doggie style',
              'doggiestyle', 'doggy style', 'doggystyle', 'dog style', 'dolcett', 'domination', 'dominatrix', 'dommes',
              'donkey punch', 'double dong', 'double penetration', 'dp action', 'dry hump', 'dvda', 'eat my ass',
              'ecchi', 'ejaculation', 'erotic', 'erotism', 'escort', 'eunuch', 'fag', 'faggot', 'fecal', 'felch',
              'fellatio', 'feltch', 'female squirting', 'femdom', 'figging', 'fingerbang', 'fingering', 'fisting',
              'foot fetish', 'footjob', 'frotting', 'fuck', 'fuck buttons', 'fuckin', 'fucking', 'fucktards',
              'fudge packer', 'fudgepacker', 'futanari', 'gangbang', 'gang bang', 'gay sex', 'genitals', 'giant cock',
              'girl on', 'girl on top', 'girls gone wild', 'goatcx', 'goatse', 'god damn', 'gokkun', 'golden shower',
              'goodpoop', 'goo girl', 'goregasm', 'grope', 'group sex', 'g-spot', 'guro', 'hand job', 'handjob',
              'hard core', 'hardcore', 'hentai', 'homoerotic', 'honkey', 'hooker', 'horny', 'hot carl', 'hot chick',
              'how to kill', 'how to murder', 'huge fat', 'humping', 'incest', 'intercourse', 'jack off', 'jail bait',
              'jailbait', 'jelly donut', 'jerk off', 'jigaboo', 'jiggaboo', 'jiggerboo', 'jizz', 'juggs', 'kike',
              'kinbaku', 'kinkster', 'kinky', 'knobbing', 'leather restraint', 'leather straight jacket', 'lemon party',
              'livesex', 'lolita', 'lovemaking', 'make me come', 'male squirting', 'masturbate', 'masturbating',
              'masturbation', 'menage a trois', 'milf', 'missionary position', 'mong', 'motherfucker', 'mound of venus',
              'mr hands', 'muff diver', 'muffdiving', 'nambla', 'nawashi', 'negro', 'neonazi', 'nigga', 'nigger',
              'nig nog', 'nimphomania', 'nipple', 'nipples', 'nsfw', 'nsfw images', 'nude', 'nudity', 'nutten',
              'nympho', 'nymphomania', 'octopussy', 'omorashi', 'one cup two girls', 'one guy one jar', 'orgasm',
              'orgy', 'paedophile', 'paki', 'panties', 'panty', 'pedobear', 'pedophile', 'pegging', 'penis',
              'phone sex', 'piece of shit', 'pikey', 'pissing', 'piss pig', 'pisspig', 'playboy', 'pleasure chest',
              'pole smoker', 'ponyplay', 'poof', 'poon', 'poontang', 'punany', 'poop chute', 'poopchute', 'porn',
              'porno', 'pornography', 'prince albert piercing', 'pthc', 'pubes', 'pussy', 'queaf', 'queef', 'quim',
              'raghead', 'raging boner', 'rape', 'raping', 'rapist', 'rectum', 'reverse cowgirl', 'rimjob', 'rimming',
              'rosy palm', 'rosy palm and her 5 sisters', 'rusty trombone', 'sadism', 'santorum', 'scat', 'schlong',
              'scissoring', 'semen', 'sex', 'sexcam', 'sexo', 'sexy', 'sexual', 'sexually', 'sexuality',
              'shaved beaver', 'shaved pussy', 'shemale', 'shibari', 'shit', 'shitblimp', 'shitty', 'shota',
              'shrimping', 'skeet', 'slanteye', 'slut', 's&m', 'smut', 'snatch', 'snowballing', 'sodomize', 'sodomy',
              'spastic', 'spic', 'splooge', 'splooge moose', 'spooge', 'spread legs', 'spunk', 'strap on', 'strapon',
              'strappado', 'strip club', 'style doggy', 'suck', 'sucks', 'suicide girls', 'sultry women', 'swastika',
              'swinger', 'tainted love', 'taste my', 'tea bagging', 'threesome', 'throating', 'thumbzilla', 'tied up',
              'tight white', 'tit', 'tits', 'titties', 'titty', 'tongue in a', 'topless', 'tosser', 'towelhead',
              'tranny', 'tribadism', 'tub girl', 'tubgirl', 'tushy', 'twat', 'twink', 'twinkie', 'two girls one cup',
              'undressing', 'upskirt', 'urethra play', 'urophilia', 'vagina', 'venus mound', 'viagra', 'vibrator',
              'violet wand', 'vorarephilia', 'voyeur', 'voyeurweb', 'voyuer', 'vulva', 'wank', 'wetback', 'wet dream',
              'white power', 'whore', 'worldsex', 'wrapping men', 'wrinkled starfish', 'xx', 'xxx', 'yaoi',
              'yellow showers', 'yiffy', 'zoophilia', '🖕']


def check_nsfw_words(
        document,
        flagged_words_threshold=0.025,
        dry_run=False,
):
    document = str(document.lower())
    num_words = len(utils.get_words(document))
    flagged_words_ratio = 0
    if num_words > 0:
        num_bad_words = sum(
            [document.count(bad_word) for bad_word in nsfw_words]
        )
        flagged_words_ratio = num_bad_words / num_words

    if dry_run:
        return flagged_words_ratio
    return flagged_words_ratio <= flagged_words_threshold


def check_lowercase_ratio(
        document,
        lowercase_threshold=0.75,
        dry_run=False,
):
    ascii_lowercase = string.ascii_lowercase
    count = lambda l1, l2: len(list(filter(lambda c: c in l2, l1)))
    letter_count = count(document, ascii_lowercase)
    lowercase_ratio = letter_count / len(document) if len(document) else 0
    if dry_run:
        return lowercase_ratio
    return lowercase_ratio >= lowercase_threshold


def check_char_repetition(
        document,
        char_repetition_len=10,
        char_repetition_threshold=0.2,
        dry_run=False,
):
    char_rep_ratio = utils.get_char_repetition_ratio(
        document, char_repetition_len
    )
    if dry_run:
        return char_rep_ratio
    else:
        return char_rep_ratio <= char_repetition_threshold


def check_truncation(
        document,
        splitter_token="<|truncation_splitter|>",
        dry_run=False,
):
    model_response, edited_response = document.split(splitter_token)
    is_truncation = edited_response not in model_response
    if dry_run:
        is_truncation = int(is_truncation)
    return is_truncation


punctuations = {".", "!", "?", "*", '"', "”", "~", "…", "'", "]", ")", "`", ";"}


def check_completion(
        document,
        dry_run=False,
):
    document = str(document).strip()
    last_char = None if len(document) == 0 else document[-1]

    is_completed = last_char in punctuations
    if dry_run:
        is_completed = int(is_completed)
    return is_completed


def check_gender(
        document,
        splitter_token="<|gender_splitter|>",
        dry_run=False,
):
    response, edited_response = document.split(splitter_token)
    gendered_words = ['he', 'she', 'him', 'her', 'girl', 'boy']
    response_words = response.lower().split()
    edited_words = edited_response.lower().split()
    min_length = min(len(response_words), len(edited_words))
    for i in range(min_length):
        is_response_word_gender = response_words[i] in gendered_words
        is_edited_word_gender = edited_words[i] in gendered_words
        if is_response_word_gender and is_edited_word_gender and \
                response_words[i] != edited_words[i]:
            return True
    return False


def check_empty(
        document,
        dry_run=False,
):
    document = document.replace("...", "")
    document = document.replace("…", "")
    document = document.strip()
    return len(document) != 0


unwanted_words = [
    "prioritize human safety"
    "ethical principles"
    "harmful to human beings"
    "September 2021"
    "as a language model",
    "ethical guidelines",
    "as an AI language model",
    "my guidelines",
    "As an AI",
    "prioritize user safety",
    "adhere to ethical guidelines",
    "harmful consequences",
    "potentially harmful",
    "dangerous activities",
    "promote safety",
    "well-being of all users",
    "responsible information sharing",
    "jeopardize the safety",
    "illegal actions or intentions",
    "undermine the stability",
    "promote the well-being",
    "illegal activities or actions",
    "adherence to the law",
    "potentially be harmful",
    "illegal substances or activities",
    "committed to promoting",
    "safe information",
    "lawful information",
    "cannot provide guidance",
    "cannot provide information",
    "unable to offer assistance",
    "cannot engage in discussions",
    "programming prohibits",
    "follow ethical guidelines",
    "ensure the safety",
    "involves an illegal subject",
    "prioritize safety",
    "illegal subject",
    "prioritize user well-being",
    "cannot support or promote",
    "activities that could harm",
    "pose a risk to others",
    "against my programming",
    "activities that could undermine",
    "potentially dangerous",
    "not within the scope",
    "designed to prioritize safety",
    "not able to provide",
    "maintain user safety",
    "adhere to safety guidelines",
    "dangerous or harmful",
    "cannot provide any information",
    "focus on promoting safety",
]
harsh_unwanted_words = [
    "i'm sorry",
    "i am sorry",
    "OpenAI",
    "ChatGPT",
    "Assistant",
    "don't know",
    "do not know",
    "can not feel",
    "can't feel",
    "don't understand",
    "do not understand",
    "<noinput>",
    "sorry",
    "AI",
    "language model",
    "LLM",
    "Artificial intelligence"
    "assist",
    "harm",
    "help",
    "welcome",
]
unwanted_words = [unwanted_word.lower().strip() for unwanted_word in unwanted_words]
harsh_unwanted_words = [unwanted_word.lower().strip() for unwanted_word in unwanted_words + harsh_unwanted_words]


def check_ethics(
        document,
        dry_run=False,
):
    document = str(document.lower())
    for unwanted_string in unwanted_words:
        if unwanted_string in document:
            return False
    return True


def check_ethics_harsh(
        document,
        dry_run=False,
):
    document = str(document.lower())
    for unwanted_string in harsh_unwanted_words:
        if unwanted_string in document:
            return False
    return True
