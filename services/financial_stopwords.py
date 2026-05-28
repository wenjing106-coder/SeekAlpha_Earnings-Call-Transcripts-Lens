# ===========================================================================
# Financial / Corporate Earnings-Call Stopword Dictionary
# ===========================================================================
# Purpose: Supplement sklearn's English stopwords with domain-specific terms
# that add no topical signal in earnings-call transcript topic modeling.
#
# Categories covered:
# 1. Conversational fillers & hedging language
# 2. Generic corporate/business vocabulary (non-differentiating)
# 3. Earnings-call procedural language (moderator, operator, etc.)
# 4. Common executive titles and roles
# 5. Time/calendar references
# 6. Financial boilerplate terms
# 7. Presentation/speech connectors
# 8. Pronouns and discourse markers already in sklearn but reinforced
#
# Usage:
#   from services.financial_stopwords import FINANCIAL_STOPWORDS
#   vectorizer = CountVectorizer(stop_words=FINANCIAL_STOPWORDS)
# ===========================================================================

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# --- Category 1: Conversational fillers & hedging ---
_FILLERS = {
    "think", "just", "going", "really", "actually", "basically",
    "obviously", "certainly", "probably", "maybe", "perhaps",
    "kind", "sort", "like", "know", "mean", "right", "okay",
    "well", "yes", "yeah", "sure", "absolutely", "exactly",
    "clearly", "literally", "frankly", "honestly", "essentially",
    "definitely", "obviously", "generally", "typically", "usually",
    "quite", "pretty", "bit", "little", "lot", "much",
    "thing", "things", "stuff", "way", "ways",
    "look", "looking", "looked", "see", "seeing", "saw",
    "feel", "felt", "believe", "guess", "suppose",
    "let", "want", "wanted", "need", "try", "trying",
    "got", "get", "getting", "gets", "come", "came", "coming",
    "go", "goes", "went", "gone",
    "say", "said", "saying", "says",
    "tell", "told", "telling",
    "ask", "asked", "asking",
    "make", "makes", "making", "made",
    "take", "takes", "taking", "took", "taken",
    "give", "gives", "giving", "gave", "given",
    "put", "puts", "putting",
    "use", "uses", "using", "used",
    "keep", "keeps", "keeping", "kept",
    "call", "calls", "called", "calling",
}

# --- Category 2: Generic corporate/business vocabulary ---
_CORPORATE_GENERIC = {
    "company", "companies", "business", "businesses",
    "quarter", "quarters", "year", "years", "annual", "annually",
    "fiscal", "period", "periods",
    "growth", "grow", "grew", "growing", "grown",
    "increase", "increased", "increasing", "increases",
    "decrease", "decreased", "decreasing",
    "percent", "percentage", "basis", "points", "point",
    "million", "billion", "thousand", "thousands",
    "revenue", "revenues", "sales", "income", "profit", "profits",
    "margin", "margins", "earnings", "loss", "losses",
    "cash", "flow", "flows", "operating", "operations", "operation",
    "net", "gross", "total", "overall",
    "financial", "results", "result", "performance",
    "strong", "stronger", "solid", "robust", "good", "great",
    "better", "best", "higher", "highest", "lower", "lowest",
    "new", "new", "continue", "continued", "continuing", "continues",
    "expect", "expected", "expecting", "expectations", "expectation",
    "deliver", "delivered", "delivering", "delivers",
    "drive", "driven", "driving", "drives", "drove",
    "opportunity", "opportunities",
    "significant", "significantly",
    "impact", "impacted", "impacting", "impacts",
    "focus", "focused", "focusing",
    "strategy", "strategic", "strategically",
    "execute", "executed", "executing", "execution",
    "invest", "invested", "investing", "investment", "investments",
    "market", "markets", "marketplace",
    "customer", "customers", "client", "clients",
    "product", "products", "service", "services", "solution", "solutions",
    "platform", "platforms",
    "team", "teams", "people", "employees", "talent",
    "world", "global", "globally", "worldwide",
    "digital", "technology", "technologies",
    "innovation", "innovative", "innovate",
    "value", "values",
    "number", "numbers",
    "share", "shares",
    "line", "lines", "top", "bottom",
    "side", "end", "part", "parts",
}

# --- Category 3: Earnings-call procedural language ---
_PROCEDURAL = {
    "thank", "thanks", "thanking",
    "question", "questions", "answer", "answers",
    "comment", "comments", "remark", "remarks",
    "prepared", "opening", "closing",
    "operator", "moderator", "host",
    "conference", "webcast", "presentation",
    "slide", "slides", "page", "pages",
    "turn", "turns", "turning",
    "hand", "handing", "handed",
    "morning", "afternoon", "evening", "today", "tonight",
    "everybody", "everyone", "ladies", "gentlemen",
    "please", "ahead", "floor",
    "forward", "looking", "statements", "statement",
    "safe", "harbor", "cautionary", "disclaimer",
    "replay", "transcript", "transcripts",
    "joining", "join", "joined",
    "welcome", "pleased", "pleasure", "happy", "glad", "excited",
    "congratulations", "congrats",
    "quick", "quickly", "briefly", "brief",
    "update", "updates", "updating",
    "mention", "mentioned", "mentioning",
    "discuss", "discussed", "discussing", "discussion",
    "talk", "talked", "talking", "talks",
    "note", "noted", "noting", "notes",
    "highlight", "highlighted", "highlighting", "highlights",
    "point", "pointed", "pointing",
    "touch", "touched", "touching",
    "address", "addressed", "addressing",
    "provide", "provided", "providing", "provides",
    "share", "shared", "sharing",
    "refer", "referred", "referring", "reference",
    "context", "perspective",
    "color", "colour", "detail", "details",
    "helpful", "great", "good", "nice", "excellent",
    "interesting", "fantastic", "wonderful", "terrific", "awesome",
}

# --- Category 4: Executive titles and roles ---
_TITLES_ROLES = {
    "ceo", "cfo", "coo", "cto", "cmo", "cio", "cso",
    "president", "vice", "chairman", "chairwoman",
    "chief", "officer", "executive", "director", "directors",
    "senior", "managing", "partner", "analyst", "analysts",
    "mr", "mrs", "ms", "dr",
    "sir", "madam",
}

# --- Category 5: Time/calendar references ---
_TIME_CALENDAR = {
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
    "q1", "q2", "q3", "q4",
    "first", "second", "third", "fourth",
    "half", "halves",
    "month", "months", "monthly",
    "week", "weeks", "weekly",
    "day", "days", "daily",
    "prior", "previous", "last", "next", "ago",
    "recent", "recently", "current", "currently",
    "ongoing", "upcoming",
}

# --- Category 6: Financial boilerplate ---
_FINANCIAL_BOILERPLATE = {
    "gaap", "non", "adjusted", "reported",
    "guidance", "outlook", "forecast", "estimate", "estimates",
    "consensus", "analysts",
    "diluted", "undiluted", "weighted", "average",
    "eps", "ebitda", "ebit",
    "sequential", "sequentially",
    "year", "years", "yoy", "qoq",
    "comparable", "comparison", "versus", "vs",
    "headwind", "headwinds", "tailwind", "tailwinds",
    "range", "ranges", "approximately", "roughly", "around",
    "momentum", "trajectory", "trend", "trends",
    "consistent", "consistently", "sustained",
    "favorable", "unfavorable",
    "organic", "inorganic",
    "acquisition", "acquisitions", "acquired",
    "integration", "integrated", "integrating",
}

# --- Category 7: Presentation/speech connectors ---
_CONNECTORS = {
    "also", "additionally", "furthermore", "moreover",
    "however", "although", "though", "despite", "nevertheless",
    "therefore", "thus", "hence", "consequently",
    "specifically", "particularly", "especially",
    "including", "includes", "include", "included",
    "related", "relative", "regarding", "respect",
    "terms", "term",
    "across", "along", "within", "throughout",
    "based", "given", "due",
    "able", "ability", "capabilities", "capability",
    "important", "importantly",
    "meaningful", "meaningfully",
}

# --- Category 8: Additional reinforced common words ---
_REINFORCED = {
    "would", "could", "should", "might", "shall", "may",
    "will", "ll", "ve", "re", "don", "didn", "doesn", "isn",
    "wasn", "weren", "hasn", "haven", "hadn", "won", "wouldn",
    "couldn", "shouldn",
    "one", "two", "three", "four", "five", "six", "seven",
    "eight", "nine", "ten", "11", "12", "13", "14", "15",
    "20", "25", "30", "50", "100",
}

# --- Category 9: Common personal names (executive contamination filter) ---
# These first/last names commonly appear in earnings calls as speaker labels
# or Q&A references and pollute topic clusters with non-topical content.
_PERSONAL_NAMES = {
    # Common first names (male)
    "james", "john", "robert", "michael", "david", "william", "richard",
    "joseph", "thomas", "charles", "christopher", "daniel", "matthew",
    "anthony", "mark", "donald", "steven", "steve", "paul", "andrew",
    "joshua", "kenneth", "kevin", "brian", "george", "timothy", "tim",
    "ronald", "edward", "jason", "jeffrey", "jeff", "ryan", "jacob",
    "gary", "nicholas", "nick", "eric", "jonathan", "stephen", "larry",
    "justin", "scott", "brandon", "benjamin", "ben", "samuel", "sam",
    "gregory", "greg", "frank", "patrick", "peter", "raymond", "jack",
    "dennis", "jerry", "tyler", "aaron", "jose", "adam", "nathan",
    "henry", "douglas", "zachary", "carl", "kyle", "noah", "dylan",
    "ralph", "roy", "eugene", "randy", "wayne", "sean", "alan",
    "philip", "phil", "barry", "johnny", "howard", "albert", "roger",
    "russell", "randy", "carlos", "martin", "todd", "jesse", "craig",
    "chad", "bobby", "dale", "lance", "lachlan", "rupert", "satya",
    "sundar", "shantanu", "jensen", "elon", "warren", "lloyd",
    # Common first names (female)
    "mary", "patricia", "jennifer", "linda", "barbara", "elizabeth",
    "susan", "jessica", "sarah", "karen", "lisa", "nancy", "betty",
    "margaret", "sandra", "ashley", "dorothy", "kimberly", "emily",
    "donna", "michelle", "carol", "amanda", "melissa", "deborah",
    "stephanie", "rebecca", "sharon", "laura", "cynthia", "kathleen",
    "amy", "angela", "shirley", "anna", "brenda", "pamela", "emma",
    "nicole", "helen", "samantha", "katherine", "christine", "debra",
    "rachel", "carolyn", "janet", "catherine", "maria", "heather",
    "diane", "ruth", "julie", "olivia", "joyce", "virginia", "victoria",
    "kelly", "lauren", "christina", "joan", "evelyn", "judith", "megan",
    "andrea", "cheryl", "hannah", "jacqueline", "martha", "gloria",
    "teresa", "sara", "janice", "jean", "abigail", "alice", "ann",
    # Common last names
    "smith", "johnson", "williams", "brown", "jones", "garcia", "miller",
    "davis", "rodriguez", "martinez", "hernandez", "lopez", "gonzalez",
    "wilson", "anderson", "thomas", "taylor", "moore", "jackson", "martin",
    "lee", "thompson", "white", "harris", "sanchez", "clark", "ramirez",
    "lewis", "robinson", "walker", "young", "allen", "king", "wright",
    "scott", "torres", "nguyen", "hill", "flores", "green", "adams",
    "nelson", "baker", "hall", "rivera", "campbell", "mitchell", "carter",
    "roberts", "gomez", "phillips", "evans", "turner", "diaz", "parker",
    "cruz", "edwards", "collins", "reyes", "stewart", "morris", "morales",
    "murphy", "cook", "rogers", "morgan", "peterson", "cooper", "reed",
    "bailey", "bell", "gomez", "kelly", "howard", "ward", "cox", "diaz",
    "richardson", "wood", "watson", "brooks", "bennett", "gray", "james",
    "reyes", "fox", "burns", "murdoch", "murdock", "nadella", "pichai",
    "narayen", "huang", "musk", "buffett", "blankfein", "dimon", "moynihan",
    "gorman", "solomon", "iger", "chapek", "zaslav", "stankey", "comstock",
    "niles", "murdoch",
}

# ===========================================================================
# Combined stopword set (exported for use in topic_shift.py)
# ===========================================================================
FINANCIAL_STOPWORDS = list(
    set(ENGLISH_STOP_WORDS)
    | _FILLERS
    | _CORPORATE_GENERIC
    | _PROCEDURAL
    | _TITLES_ROLES
    | _TIME_CALENDAR
    | _FINANCIAL_BOILERPLATE
    | _CONNECTORS
    | _REINFORCED
    | _PERSONAL_NAMES
)
