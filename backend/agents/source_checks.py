"""Детерминированные проверки формы утверждений для полного режима."""

import re

from backend.agents.state import FactualFinding, ValidationCode

_ALLEGATION = re.compile(r"\b(?:alleg(?:ed|es)|claim(?:ed|s)|accus(?:ed|es))\b", re.I)
_ATTRIBUTION = re.compile(
    r"\b(?:according to|alleg\w*|claim\w*|accus\w*|said|says|reportedly|denied|argued|asserted)\b", re.I,
)
_NUMBERED_DETAIL = re.compile(r"\b\d+[\s-]+[a-z]+\s+[a-z]+\b", re.I)
_VEHICLE_IMPACT = re.compile(
    r"\b(?:hit|struck|run over) by (?:[a-z]+ ){0,3}"
    r"(?:lorry|truck|van|car|bus|train|motorcycle|bicycle|tram|vehicle)\b", re.I,
)
_IMPACT_EVIDENCE = re.compile(
    r"\b(?:hit|struck|collision|collid\w*|crash\w*|ran over|run over|knocked down)\b", re.I,
)
_CAUSAL_CONNECTOR = re.compile(r"\b(?:prompting|causing|leading to|resulting in)\b", re.I)
_CAUSAL_EVIDENCE = re.compile(
    r"\b(?:because|due to|as a result|caus(?:ed|es|ing)|prompt(?:ed|s|ing)|led to|leading to|result(?:ed|s|ing) in)\b",
    re.I,
)
_QUALIFIER = re.compile(
    r"(?P<attribution>\b(?:according to|по данным|по словам|как утвержда\w+)\b)"
    r"|\b(?:allegedly|reportedly|suspect(?:ed|s)|claim(?:ed|s)|accus(?:ed|es)|alleg(?:ed|es)|"
    r"may have|might have|could have|whether|check to see if|по версии|предположительно|якобы|"
    r"возможно|мог(?:ла|ли)?|подозрева\w*|обвиня\w*|утвержда\w*|предполага\w*|"
    r"проверя\w*\s*,?\s*(?:могли|ли))\b", re.I,
)
_CLAUSE = re.compile(r"\s*;\s*|,\s+(?:and|и)\s+|,?\s+\b(?:but|however|whereas|но|однако)\b\s+", re.I)
_CLAIM_STOP = set(
    "a an the of in at on by to from that it he she they his her their have has had was were is are be been "
    "as and or with this there could should would can after before for about into out person people man woman "
    "police officials news government local also then when during stated said say says told "
    "и в во на из при по с со о об от до за у к для что это они она он их его ее был была были быть "
    "который которая которые где когда как люди человек полиция сообщили сказал заявили".split()
)
_WORD = re.compile(r"[^\W_]+")
_REPORT = re.compile(
    r"^(?P<speaker>[^,:.!?]{3,85}?)\s+(?:says?|said|claims?|claimed|confirms?|confirmed|states?|stated)\s+"
    r"(?P<claim>.+)", re.I,
)
_OPEN_QUESTION = re.compile(r"\b(?:check|verify|determine|investigate|establish)\b.{0,42}\b(?:if|whether)\b", re.I)
_HEDGES = re.compile(r"\b(?:might|may|could|possibly|perhaps|suspected|not|denied|disputed)\b", re.I)
_SPEAKER_STOP = {"the", "but", "however", "and", "while"}


def unattributed_allegation(source_sentences: list[str], summary_sentences: list[str]) -> bool:
    def details(text: str) -> set[str]:
        return {re.sub(r"[\s-]+", " ", item.casefold()) for item in _NUMBERED_DETAIL.findall(text)}

    attributed = set()
    rest = set()
    for sentence in source_sentences:
        marker = _ALLEGATION.search(sentence)
        if marker:
            attributed.update(details(sentence[marker.end():]))
        else:
            rest.update(details(sentence))
    return any(not _ATTRIBUTION.search(sentence) and bool(details(sentence) & (attributed - rest))
               for sentence in summary_sentences)


def unsupported_vehicle_impact(source: str, summary: str) -> bool:
    return bool(_VEHICLE_IMPACT.search(summary) and not _IMPACT_EVIDENCE.search(source))


def unsupported_causal_connector(source: str, summary: str) -> bool:
    return bool(_CAUSAL_CONNECTOR.search(summary) and not _CAUSAL_EVIDENCE.search(source))


def _claim_terms(text: str) -> set[str]:
    terms = set()
    for token in _WORD.findall(text.casefold().replace("ё", "е")):
        if token in _CLAIM_STOP or not token.isalpha() or len(token) < 3:
            continue
        if token.isascii():
            token = re.sub(r"ies$", "y", token)
            token = re.sub(r"(ing|ed|s)$", "", token) if len(token) >= 5 else token
        else:
            token = re.sub(r"(ами|ями|ого|ому|ах|ях|ом|ем|а|я|у|ю|е|и|ы)$", "", token) if len(token) >= 6 else token
        terms.add(token)
    return terms


def _clauses(sentences: list[str]) -> list[str]:
    return [clause.strip(" ,") for sentence in sentences for clause in _CLAUSE.split(sentence) if clause.strip(" ,")]


def attribution_findings(source_sentences: list[str], summary_sentences: list[str]) -> list[FactualFinding]:
    """Отметить перенос открытого вопроса в прямое утверждение другого участника."""
    def reports(sentences):
        for clause in _clauses(sentences):
            if match := _REPORT.match(clause):
                speaker = set(_WORD.findall(match["speaker"].casefold())) - _SPEAKER_STOP
                yield clause, speaker, match["claim"]

    source = list(reports(source_sentences))
    questions = [(speaker, _claim_terms(claim[question.end():])) for _, speaker, claim in source
                 if (question := _OPEN_QUESTION.search(claim))]
    if not questions:
        return []
    findings = []
    for clause, speaker, claim in reports(summary_sentences):
        if not speaker or _OPEN_QUESTION.search(claim) or _HEDGES.search(claim):
            continue
        terms = _claim_terms(claim)
        for original_speaker, question_terms in questions:
            shared = terms & question_terms
            if speaker & original_speaker or len(shared) < max(3, len(question_terms) * 0.6):
                continue
            if any(speaker & author and not _OPEN_QUESTION.search(statement)
                   and not _HEDGES.search(statement) and shared <= _claim_terms(statement)
                   for _, author, statement in source):
                continue
            findings.append(FactualFinding(
                code=ValidationCode.UNSUPPORTED_ATTRIBUTION, severity="review", category="claim",
                subject=clause[:160],
            ))
            break
    return findings


def qualification_findings(source_sentences: list[str], summary_sentences: list[str]) -> list[FactualFinding]:
    qualified = []
    assertions = []
    for clause in _clauses(source_sentences):
        marker = _QUALIFIER.search(clause)
        if marker is None:
            assertions.append(_claim_terms(clause))
            continue
        tail = _claim_terms(clause[marker.end():])
        head = _claim_terms(clause[:marker.start()])
        # Для постпозиции «..., по словам ...» предмет утверждения находится перед маркером.
        terms = head if clause[:marker.start()].rstrip().endswith(",") and len(head) >= 2 else tail
        if len(terms) < 2:
            terms = head | tail
        if len(terms) >= 2:
            qualified.append((terms, bool(marker["attribution"])))

    findings = []
    for clause in _clauses(summary_sentences):
        if _QUALIFIER.search(clause) or _ATTRIBUTION.search(clause):
            continue
        asserted = _claim_terms(clause)
        strongest = 0
        for terms, reported_only in qualified:
            shared = terms & asserted
            if len(shared) < 2 or len(shared) / len(terms) < 0.35:
                continue
            if any(shared <= supported for supported in assertions):
                continue
            direct = len(shared) >= min(len(terms), 3) and len(shared) / len(terms) >= 0.65
            strongest = max(strongest, 2 if direct and not reported_only else 1)
        if strongest:
            findings.append(FactualFinding(
                code=ValidationCode.LOST_QUALIFICATION if strongest == 2 else ValidationCode.UNCERTAIN_QUALIFICATION,
                severity="error" if strongest == 2 else "review", category="claim", subject=clause[:160],
            ))
    return findings
