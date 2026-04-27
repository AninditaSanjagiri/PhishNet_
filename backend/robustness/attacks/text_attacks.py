"""
robustness/attacks/text_attacks.py
====================================
Six text-level adversarial attack strategies against the DistilBERT
phishing text classifier. These simulate realistic attacker behaviour
at the semantic/linguistic level — the gap identified in the literature
as untested (Phan The Duy et al., Table 19).

Attacks implemented:
  1. synonym_swap         — replace phishing keywords with synonyms
  2. paraphrase_urgency   — rephrase urgency phrases to sound formal/benign
  3. negation_inject      — add negation noise to non-critical sentences
  4. whitespace_inject    — insert zero-width spaces into trigger words
  5. leet_substitution    — replace chars in keywords: password → p@ssw0rd
  6. sentence_dilution    — pad text with benign boilerplate sentences

All attacks operate on raw HTML body text (the TextAgent's input).
Each returns (perturbed_text, attack_name, n_changes).
"""
import random
import re
from typing import Callable

# Synonym map for high-weight phishing keywords
SYNONYM_MAP: dict[str, list[str]] = {
    "verify":      ["validate", "confirm", "authenticate", "check"],
    "password":    ["passphrase", "credentials", "access code", "secret"],
    "account":     ["profile", "membership", "subscription", "portal"],
    "suspended":   ["temporarily restricted", "placed on hold", "under review"],
    "urgent":      ["important", "time-sensitive", "requiring attention", "critical"],
    "click here":  ["follow the link", "tap this button", "use the link below"],
    "login":       ["sign in", "access your profile", "enter your credentials"],
    "verify your identity": ["confirm your details", "validate your information"],
    "free":        ["complimentary", "no-cost", "at no charge"],
    "winner":      ["selected recipient", "chosen beneficiary", "eligible participant"],
    "prize":       ["reward", "benefit", "incentive", "gift"],
    "banking":     ["financial", "monetary", "fiscal"],
    "social security": ["national identification", "government ID"],
    "unauthorized": ["unrecognised", "unverified", "flagged"],
}

URGENCY_PATTERNS = [
    (r"immediately", "at your earliest convenience"),
    (r"right away",  "when you have a moment"),
    (r"within 24 hours", "within a reasonable time"),
    (r"or your account will be (closed|suspended|terminated)",
     "to ensure continued service"),
    (r"act now",     "please review"),
    (r"expires soon", "is pending renewal"),
    (r"final notice", "recent update"),
    (r"failure to (comply|respond|verify)", "should you wish to update"),
]

BENIGN_BOILERPLATE = [
    "We appreciate your continued business with us.",
    "Our customer service team is available Monday through Friday.",
    "Thank you for choosing our services.",
    "Your satisfaction is our top priority.",
    "Please retain this message for your records.",
    "For assistance, visit our help centre at support.example.com.",
    "This message was sent to you as part of our standard communications.",
    "We are committed to protecting your privacy and security.",
]

LEET_MAP: dict[str, str] = {
    'a': '@', 'e': '3', 'i': '1', 'o': '0', 's': '$', 't': '7',
}


# ── Attack implementations ────────────────────────────────────────────────────

def synonym_swap(text: str) -> tuple[str, str, int]:
    """Replace high-weight phishing keywords with plausible synonyms."""
    result   = text
    n_swaps  = 0
    # Sort by length descending to avoid partial replacements
    for kw, syns in sorted(SYNONYM_MAP.items(), key=lambda x: -len(x[0])):
        pattern = re.compile(re.escape(kw), re.IGNORECASE)
        if pattern.search(result):
            replacement = random.choice(syns)
            result  = pattern.sub(replacement, result, count=2)
            n_swaps += 1
    return result, "synonym_swap", n_swaps


def paraphrase_urgency(text: str) -> tuple[str, str, int]:
    """Replace urgency trigger phrases with softer equivalents."""
    result  = text
    n_swaps = 0
    for pattern, replacement in URGENCY_PATTERNS:
        new, count = re.subn(pattern, replacement, result, flags=re.IGNORECASE)
        if count:
            result   = new
            n_swaps += count
    return result, "paraphrase_urgency", n_swaps


def negation_inject(text: str) -> tuple[str, str, int]:
    """
    Insert 'not' or 'no' before a random non-critical verb to confuse
    sentence-level classifiers without breaking overall meaning to humans.
    """
    # Target filler verbs that are not the primary phishing signal
    fillers = re.findall(
        r'\b(received|reviewed|processed|completed|confirmed)\b',
        text, re.IGNORECASE)
    if not fillers:
        return text, "negation_inject", 0
    target = random.choice(fillers)
    result = re.sub(
        r'\b' + re.escape(target) + r'\b',
        f"not fully {target.lower()}",
        text, count=1, flags=re.IGNORECASE)
    return result, "negation_inject", 1


def whitespace_inject(text: str) -> tuple[str, str, int]:
    """
    Insert zero-width space (U+200B) inside trigger words.
    Invisible to humans, breaks token-matching classifiers.
    E.g. 'password' → 'pass​word'
    """
    ZWS     = "\u200b"
    result  = text
    n_inj   = 0
    targets = ["password", "verify", "account", "suspended", "login", "banking"]
    for word in targets:
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        matches = pattern.findall(result)
        if matches:
            mid     = len(word) // 2
            broken  = word[:mid] + ZWS + word[mid:]
            result  = pattern.sub(broken, result, count=2)
            n_inj  += len(matches[:2])
    return result, "whitespace_inject", n_inj


def leet_substitution(text: str) -> tuple[str, str, int]:
    """
    Apply leet-speak substitutions to medium-weight trigger words.
    Preserves human readability while evading exact-match classifiers.
    """
    targets = ["password", "secure", "login", "verify", "account"]
    result  = text
    n_sub   = 0
    for word in targets:
        def leet_word(w: str) -> str:
            return "".join(LEET_MAP.get(c.lower(), c) for c in w)
        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
        if pattern.search(result):
            result = pattern.sub(leet_word(word), result, count=2)
            n_sub += 1
    return result, "leet_substitution", n_sub


def sentence_dilution(text: str, n_sentences: int = 3) -> tuple[str, str, int]:
    """
    Pad the text with benign boilerplate sentences at random positions.
    Dilutes phishing signal in longer texts.
    """
    sentences    = text.split(". ")
    insertions   = random.sample(BENIGN_BOILERPLATE, min(n_sentences, len(BENIGN_BOILERPLATE)))
    for ins in insertions:
        pos = random.randint(0, len(sentences))
        sentences.insert(pos, ins)
    result = ". ".join(sentences)
    return result, "sentence_dilution", len(insertions)


# ── Attack registry ──────────────────────────────────────────────────────────

ALL_TEXT_ATTACKS: list[Callable] = [
    synonym_swap,
    paraphrase_urgency,
    negation_inject,
    whitespace_inject,
    leet_substitution,
    sentence_dilution,
]

TEXT_ATTACK_NAMES = [f.__name__ for f in ALL_TEXT_ATTACKS]


def apply_all_text_attacks(text: str) -> list[dict]:
    """
    Apply every text attack to an input text.
    Returns list of dicts with keys:
      attack_name, perturbed_text, n_changes, char_delta
    """
    results = []
    for fn in ALL_TEXT_ATTACKS:
        try:
            perturbed, name, n_changes = fn(text)
            results.append({
                "attack_name":    name,
                "perturbed_text": perturbed,
                "n_changes":      n_changes,
                "char_delta":     len(perturbed) - len(text),
                "is_noop":        perturbed == text,
            })
        except Exception as exc:
            results.append({
                "attack_name":    fn.__name__,
                "perturbed_text": text,
                "n_changes":      0,
                "char_delta":     0,
                "is_noop":        True,
                "error":          str(exc),
            })
    return results


def apply_combined_text_attack(text: str) -> tuple[str, str]:
    """
    Chain synonym_swap + paraphrase_urgency + sentence_dilution.
    Represents the most realistic multi-strategy attacker.
    """
    t1, _, c1 = synonym_swap(text)
    t2, _, c2 = paraphrase_urgency(t1)
    t3, _, c3 = sentence_dilution(t2, n_sentences=2)
    desc = f"synonym({c1}) + urgency_paraphrase({c2}) + dilution({c3})"
    return t3, desc
