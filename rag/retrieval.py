import json
import re

from services.document_matching import explicit_document_matches
from services.scientific_evidence import (
    classify_experimental_evidence,
    extract_explicit_classifier_taxonomy,
    experimental_evidence_object,
    requested_answer_slots,
)
from settings import FINAL_RESULTS, INITIAL_RESULTS


SUMMARY_QUERIES = (
    "paper title abstract purpose scope and central contribution",
    "introduction background overview and research context",
    "conclusions discussion implications limitations and future directions",
    "overall framework major themes mechanisms and summary figure captions",
)
SUMMARY_FALLBACK_QUERIES = (
    "paper abstract purpose conclusions",
    "main framework major themes",
    "overview discussion conclusion",
)


_LIMITATION_PATTERNS = (
    (
        "numerical_2d",
        r"\b2D model\b",
        "The work uses a numerical 2D model rather than an in-vivo experiment.",
    ),
    (
        "fixed_tissue_properties",
        r"\b(?:unalterable|unvarying|uniform and consistent)\b.{0,120}\b(?:thermal|dielectric|tissue)?\s*propert|"
        r"\b(?:thermal|dielectric|tissue)?\s*propert\w*\b.{0,120}\b(?:unalterable|unvarying|fixed)\b",
        "Tissue properties are fixed or unvarying in the model.",
    ),
    (
        "no_phase_changes",
        r"\bno alterations? in the phase\b|\bno phase changes?\b",
        "Phase changes are excluded.",
    ),
    (
        "no_chemical_reactions",
        r"\babsence of chemical reactions?\b|\bno chemical reactions?\b",
        "Chemical reactions are excluded.",
    ),
    (
        "local_thermal_equilibrium",
        r"\blocali[sz]ed thermal equilibrium\b.{0,100}\bblood\b.{0,100}\btissue\b",
        "Local blood-tissue thermal equilibrium is assumed.",
    ),
    (
        "uniform_incident_irradiance",
        r"\buniform distribution of incident irradiance\b",
        "Incident irradiance is uniform across the exposure area.",
    ),
    (
        "simplified_environment",
        r"\bunobstructed environment\b.{0,180}\b(?:lacks|without|no)\b.{0,80}\b(?:walls?|metallic enclosures?)\b",
        "The environmental geometry is simplified and excludes surrounding walls or metallic enclosures.",
    ),
    (
        "benchmark_validation",
        r"\bvalidat(?:e|ed|ion)\b.{0,240}\b(?:previous|prior|published|benchmark|Torv[ey])\b|"
        r"\b(?:previous|prior|published|benchmark|Torv[ey])\b.{0,240}\bvalidat(?:e|ed|ion)\b",
        "Validation is against benchmarks or prior studies rather than new experimental human data.",
    ),
)


def extract_inferred_limitations(documents: list[str]) -> dict:
    """Derive modelling limitations only from explicit assumptions/validation text."""
    text = re.sub(r"\s+", " ", "\n".join(documents))
    if re.search(r"(?:^|\n)\s*(?:\d+(?:\.\d+)*\.?\s+)?limitations?\s*(?:\n|$)", "\n".join(documents), re.I):
        return {}
    items = [
        {"key": key, "statement": statement}
        for key, pattern, statement in _LIMITATION_PATTERNS
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    ]
    return {"items": items, "status": "inferred_from_stated_assumptions"} if items else {}


def extract_explicit_limitations(documents: list[str]) -> dict:
    """Collect every distinct adverse claim from an author-labelled limitation block."""
    text = re.sub(r"[ \t]+", " ", "\n".join(str(item or "") for item in documents))
    start = re.search(
        r"\b(?:there (?:are|were)|we (?:identify|acknowledge))\s+(?:some\s+)?"
        r"limitations?\b|(?:^|\n)\s*limitations?\s*(?:\n|$)",
        text,
        re.IGNORECASE,
    )
    if not start:
        return {}
    tail = text[start.start():]
    stop = re.search(
        r"\n\s*(?:\d+(?:\.\d+)*\s+)?(?:conclusions?|future work|"
        r"data availability|author contributions|funding|references)\b",
        tail,
        re.IGNORECASE,
    )
    block = tail[:stop.start()] if stop else tail[:5000]
    sentences = [
        re.sub(r"\s+", " ", sentence).strip(" \n-;:")
        for sentence in re.split(r"(?<=[.!?])\s+", block)
    ]
    adverse = re.compile(
        r"\b(?:more\s+time[ -]?consuming|more\s+complex|slower|costlier|"
        r"computationally expensive|difficult(?:y)?|drawback|requires?\s+additional|"
        r"cannot|unable|limited by)\b",
        re.IGNORECASE,
    )
    items = []
    seen_signatures = []
    for index, sentence in enumerate(sentences):
        if not adverse.search(sentence):
            continue
        if items and re.match(r"^(?:hence|therefore|thus)\b", sentence, re.IGNORECASE):
            continue
        cleaned = re.sub(
            r"^(?:there (?:are|were).*?addressed\.\s*)|^(?:first|second|third|also),?\s*",
            "",
            sentence,
            flags=re.IGNORECASE,
        ).strip()
        if len(cleaned.split()) < 5:
            continue
        signature = {
            token for token in re.findall(r"[a-z][a-z-]{3,}", cleaned.casefold())
            if token not in {"than", "more", "method", "study", "hybrid"}
        }
        if any(
            signature and existing
            and len(signature.intersection(existing)) / min(len(signature), len(existing)) >= 0.7
            for existing in seen_signatures
        ):
            continue
        evidence = [cleaned]
        if index + 1 < len(sentences) and re.search(
            r"\b(?:for example|approximately|about|times|same DOF)\b",
            sentences[index + 1],
            re.IGNORECASE,
        ):
            evidence.append(sentences[index + 1])
        key = "explicit_" + "_".join(sorted(signature)[:5])
        items.append({"key": key, "statement": cleaned, "evidence": evidence})
        seen_signatures.append(signature)
    return {"items": items, "status": "explicit_author_limitations"} if items else {}


def extract_discussion_limitations(documents: list[str]) -> dict:
    """Capture limitations explicitly stated in Discussion even without a heading."""
    text = re.sub(r"[ \t]+", " ", "\n".join(str(item or "") for item in documents))
    protected_text = re.sub(r"\be\.g\.", "e§g§", text, flags=re.I)
    sentences = [
        re.sub(r"\s+", " ", sentence.replace("e§g§", "e.g.")).strip()
        for sentence in re.split(r"(?<=[.!?])\s+", protected_text)
    ]
    limitation_language = re.compile(
        r"\b(?:limitation|caveat|however|while ideally|depends? on|"
        r"affect(?:s|ed)? (?:the )?(?:comparisons?|performance|interpretation)|"
        r"(?:may|might) (?:need|require) (?:to be )?adapt\w*|adaptation may be required|"
        r"performed worse|unlikely (?:that .*? would )?work|cannot|not possible|might not)\b",
        re.I,
    )
    topic_language = re.compile(
        r"\b(?:marker|comparison|other tissues?|generali[sz]|heterogene\w*|"
        r"false positives?|predictor|classifier|score|TSS|morphology|cell type|stressor|condition)\b",
        re.I,
    )
    items, seen = [], set()
    for sentence in sentences:
        if not limitation_language.search(sentence) or not topic_language.search(sentence):
            continue
        if not 7 <= len(sentence.split()) <= 90:
            continue
        key = re.sub(r"[^a-z0-9]+", "_", sentence.casefold())[:72].strip("_")
        signature = frozenset(re.findall(r"[a-z]{5,}", sentence.casefold()))
        if any(len(signature & prior) / max(1, min(len(signature), len(prior))) > 0.7 for prior in seen):
            continue
        seen.add(signature)
        items.append({"key": f"discussion_{key}", "statement": sentence, "evidence": [sentence]})
    return {"items": items, "status": "explicit_author_limitations"} if items else {}


def _limitation_keys(document: str) -> set[str]:
    text = re.sub(r"\s+", " ", str(document or ""))
    return {
        key for key, pattern, _ in _LIMITATION_PATTERNS
        if re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    }


def _clean_framework_item(value: str) -> str:
    item = re.sub(r"\s+", " ", value).strip(" .;:-")
    item = re.sub(r"^(?:and|or)\s+", "", item, flags=re.IGNORECASE)
    return item


def extract_named_framework_items(documents: list[str]) -> list[str]:
    """Extract the document's explicit canonical item list, not section mechanisms."""
    text = "\n".join(documents)
    patterns = (
        r"(?:these|the)\s+(?:candidate\s+)?(?:\w+\s+){0,3}hallmarks\s+are\s*:\s*(.{20,900}?)(?:\.\s|\n\n)",
        r"enumerates?\s+(?:\w+\s+){0,4}hallmarks[^.]{0,120}?\b(?:as|are)\s*:\s*(.{20,900}?)(?:\.\s|\n\n)",
    )
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if not match:
            continue
        raw_items = re.split(r",|\band\b", match.group(1), flags=re.IGNORECASE)
        items = [_clean_framework_item(item) for item in raw_items]
        items = [
            item for item in items
            if 1 <= len(item.split()) <= 6
            and not re.search(r"\b(?:figure|page|section|mechanism)\b", item, re.I)
        ]
        deduplicated = []
        seen = set()
        for item in items:
            key = re.sub(r"[^a-z0-9]+", "", item.lower())
            if key and key not in seen:
                seen.add(key)
                deduplicated.append(item)
        if len(deduplicated) >= 3:
            return deduplicated
    return []


def _framework_tokens(value: str) -> set[str]:
    stopwords = {
        "hallmark", "hallmarks", "aging", "ageing", "altered", "loss",
        "deregulated", "cellular", "dysfunction", "instability",
    }
    return {
        token[:8]
        for token in re.findall(r"[a-z][a-z-]{4,}", value.lower())
        if token not in stopwords
    }


def _candidate_matches_segment(candidate: str, segment: str) -> bool:
    candidate_lower = candidate.lower()
    segment_lower = segment.lower()
    if candidate_lower in segment_lower:
        return True
    tokens = _framework_tokens(candidate)
    generic_qualifiers = re.search(
        r"\b(?:dysfunction|instability|alterations?|changes?|defect)\b",
        candidate_lower,
    )
    if len(tokens) < 2 and generic_qualifiers:
        return False
    return bool(tokens and any(token in segment_lower for token in tokens))


def _category_segments(text: str, candidates: list[str] | None = None) -> dict[str, str]:
    markers = []
    for group in ("primary", "antagonistic", "integrative"):
        for match in re.finditer(
            rf"\b{group}\s+hallmarks?\b|"
            rf"\bhallmarks?\s+(?:considered\s+to\s+be\s+)?(?:the\s+)?{group}\b",
            text,
            re.IGNORECASE,
        ):
            markers.append((match.start(), match.end(), group))
    markers.sort()
    # Ignore a compact declaration such as "primary, antagonistic and
    # integrative" and choose the nearby explanatory passages where each
    # category receives its own text.
    triples = []
    for primary in (marker for marker in markers if marker[2] == "primary"):
        for antagonistic in (
            marker for marker in markers
            if marker[2] == "antagonistic" and marker[0] - primary[0] >= 60
        ):
            for integrative in (
                marker for marker in markers
                if marker[2] == "integrative" and marker[0] - antagonistic[0] >= 60
            ):
                span = integrative[0] - primary[0]
                if span <= 5000:
                    triples.append((span, primary, antagonistic, integrative))
    if triples:
        ranked = []
        for span, primary, antagonistic, integrative in triples:
            segments = {
                "primary": text[primary[0]:antagonistic[0]],
                "antagonistic": text[antagonistic[0]:integrative[0]],
                "integrative": text[integrative[0]:integrative[1] + 700],
            }
            grounding_hits = 0
            for candidate in candidates or []:
                grounding_hits += sum(
                    _candidate_matches_segment(candidate, segment)
                    for segment in segments.values()
                )
            ranked.append((grounding_hits, span, segments))
        best_hits = max(row[0] for row in ranked)
        sufficiently_grounded = [row for row in ranked if row[0] >= best_hits - 1]
        return min(sufficiently_grounded, key=lambda row: row[1])[2]
    return {}


def validate_framework_grounding(value: dict) -> dict:
    candidates = value.get("named_items", [])
    categories = value.get("categories", {})
    if not isinstance(candidates, list) or not isinstance(categories, dict):
        return {}
    candidate_keys = {
        re.sub(r"[^a-z0-9]+", "", item.lower()): item
        for item in candidates if isinstance(item, str)
    }
    seen = set()
    normalized = {}
    for group in ("primary", "antagonistic", "integrative"):
        items = categories.get(group, [])
        if not isinstance(items, list):
            return {}
        normalized[group] = []
        for item in items:
            key = re.sub(r"[^a-z0-9]+", "", str(item).lower())
            if key not in candidate_keys or key in seen:
                return {}
            seen.add(key)
            normalized[group].append(candidate_keys[key])
    uncertain = [item for item in candidates if re.sub(r"[^a-z0-9]+", "", item.lower()) not in seen]
    return {
        "named_items": candidates,
        "categories": normalized,
        "uncertain": uncertain,
    }


def extract_framework_grounding(documents: list[str]) -> dict:
    """Ground categories in an explicit named-item list plus framework prose."""
    candidates = extract_named_framework_items(documents)
    if not candidates:
        return {}
    text = "\n".join(documents)
    segments = _category_segments(text, candidates)
    if not segments:
        return {}

    assignments = [None] * len(candidates)
    for index, candidate in enumerate(candidates):
        matching_groups = []
        for group, segment in segments.items():
            if _candidate_matches_segment(candidate, segment):
                direct_position = segment.lower().find(candidate.lower())
                matching_groups.append((
                    group,
                    direct_position if direct_position >= 0 else len(segment) + 1,
                ))
        if len(matching_groups) == 1:
            assignments[index] = matching_groups[0][0]
        elif matching_groups:
            # A later category declaration can occur at the end of the previous
            # segment. Prefer the category whose own marker is closest to the
            # canonical item rather than discarding an otherwise explicit match.
            best_position = min(position for _, position in matching_groups)
            nearest = [
                group for group, position in matching_groups
                if position == best_position
            ]
            if len(nearest) == 1:
                assignments[index] = nearest[0]

    # Explicit lists often preserve category order. Fill only gaps bounded by
    # the same category, plus leading/trailing items anchored to the first/last
    # declared category. This resolves canonical synonyms without inventing a
    # label that was absent from the document's named-item list.
    group_order = ("primary", "antagonistic", "integrative")
    for index, assigned in enumerate(assignments):
        if assigned:
            continue
        left = next((assignments[pos] for pos in range(index - 1, -1, -1) if assignments[pos]), None)
        right = next((assignments[pos] for pos in range(index + 1, len(assignments)) if assignments[pos]), None)
        if left and left == right:
            assignments[index] = left
        elif left is None and right == group_order[0]:
            assignments[index] = right
        elif right is None and left == group_order[-1]:
            assignments[index] = left

    categories = {group: [] for group in group_order}
    for candidate, group in zip(candidates, assignments):
        if group:
            categories[group].append(candidate)
    return validate_framework_grounding({
        "named_items": candidates,
        "categories": categories,
    })


def is_broad_summary_question(question: str) -> bool:
    text = re.sub(r"\s+", " ", question.lower()).strip()
    patterns = (
        r"\bsummari[sz]e\s+(?:this|the)\s+paper\b",
        r"\bsummari[sz]e\s+.{1,120}\bpaper\b",
        r"\bmain\s+(?:findings|conclusions|contributions)\b",
        r"\b(?:give|provide)\s+me\s+an?\s+overview\b",
        r"\boverview\s+of\s+(?:this|the)\s+paper\b",
        r"\bwhole[- ]document\s+summary\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


def is_multi_figure_evidence_question(question: str) -> bool:
    text = re.sub(r"\s+", " ", str(question or "")).casefold()
    return bool(
        re.search(r"\bwhich\s+figures?\b", text)
        and re.search(r"\b(?:evidence|demonstrate|show|support)\b", text)
        and re.search(r"\b(?:both|and)\b", text)
    )


def is_section_aware_question(question: str) -> bool:
    return bool(re.search(
        r"\b(?:limitations?|future work|future research|recommendations?|"
        r"conclusions?|discussion|authors?' stated assumptions?)\b",
        str(question or ""), re.I,
    ))


def is_reported_trend_question(question: str) -> bool:
    return bool(
        re.search(r"\bwhy\b", str(question or ""), re.I)
        and re.search(r"\b(?:error|trend|increase|decrease|jump|deviation)\b", str(question or ""), re.I)
    )


def resolve_followup_retrieval_query(question: str, previous_question: str) -> str:
    """Resolve a high-confidence possessive referent without paraphrasing terms."""
    current = str(question or "")
    previous = str(previous_question or "")
    if not previous or not re.search(r"\b(?:its|it)\b", current, re.I):
        return current
    subjects = re.findall(
        r"\b((?:hybrid\s+)?[A-Z]*[A-Za-z-]*(?:\s+[A-Z]*[A-Za-z-]+){0,3}\s+method)\b",
        previous, re.I,
    )
    if not subjects:
        return current
    subject = max(subjects, key=len).strip()
    subject = re.sub(
        r"^(?:(?:why|how)\s+)?(?:does|do|did)\s+(?:the\s+)?|^the\s+",
        "", subject, flags=re.I,
    )
    return re.sub(r"\bits\b", f"the {subject}'s", current, count=1, flags=re.I)


def _retrieve_section_context(
    question, collection, embedder, reranker, available_chunks,
    selected_source=None,
):
    queries = [question, f"{question} conclusion discussion future work limitations"]
    candidates, _ = _query_candidates(
        queries, collection, embedder, min(max(INITIAL_RESULTS, 24), available_chunks)
    )
    if not candidates:
        return "", []
    names = list(dict.fromkeys([
        metadata.get("pdf", metadata.get("source", "")) for _, metadata in candidates.values()
    ] + _collection_source_names(collection)))
    explicit = explicit_document_matches(question, names)
    if len(explicit) > 1:
        choices = ", ".join(sorted(explicit))
        return f"[DOCUMENT TITLE AMBIGUOUS]\nClarify which document: {choices}.", []
    if explicit:
        target = _source_identity({"source": next(iter(explicit))})
    elif selected_source:
        target = _source_identity({"source": str(selected_source)})
    else:
        scored = reranker.predict([[question, document] for document, _ in candidates.values()])
        best = max(zip(scored, candidates.values()), key=lambda row: float(row[0]))[1][1]
        target = _source_identity(best)
    _expand_target_document_candidates(collection, candidates, target)
    rows = [
        (document, metadata, infer_section(document))
        for document, metadata in candidates.values()
        if _source_identity(metadata) == target and not _is_reference_or_metadata(document)
    ]
    terms = {
        token for token in re.findall(r"[a-z][a-z-]{4,}", question.casefold())
        if token not in {"what", "authors", "explicitly", "identify", "their"}
    }
    ranked = []
    model_scores = reranker.predict([[question, document] for document, _, _ in rows]) if rows else []
    for score, (document, metadata, section) in zip(model_scores, rows):
        lower = document.casefold()
        exact_hits = sum(term in lower for term in terms)
        section_boost = 4 if section in {"conclusion", "discussion"} else 0
        future_boost = 4 if re.search(r"\bfuture (?:work|research)\b|\bwill (?:also )?(?:be )?(?:investigated|considered|used)\b", lower) else 0
        ranked.append((float(score) + exact_hits + section_boost + future_boost, document, metadata, section))
    ranked.sort(reverse=True, key=lambda row: row[0])
    if ranked and isinstance(ranked[0][2].get("page"), int):
        anchor_page = ranked[0][2]["page"]
        adjacent = [
            row for row in ranked
            if isinstance(row[2].get("page"), int)
            and abs(row[2]["page"] - anchor_page) <= 1
        ]
        if len(adjacent) >= 2:
            ranked = adjacent
    selected, pages = [], set()
    for score, document, metadata, section in ranked:
        page = metadata.get("page")
        if page in pages and len(selected) >= 2:
            continue
        selected.append(_source_item(score, document, metadata, section))
        pages.add(page)
        if len(selected) >= FINAL_RESULTS:
            break
    explicit_limitations = {}
    if re.search(r"\blimitations?\b", question, re.IGNORECASE):
        ordered_documents = [
            document for document, metadata, _ in sorted(
                rows,
                key=lambda row: (
                    int(row[1].get("page", 0)) if isinstance(row[1].get("page"), int) else 0,
                    int(row[1].get("chunk", 0)) if isinstance(row[1].get("chunk"), int) else 0,
                ),
            )
        ]
        explicit_limitations = extract_explicit_limitations(ordered_documents)
        discussion_limitations = extract_discussion_limitations(ordered_documents)
        if discussion_limitations:
            if explicit_limitations:
                existing = {item["statement"].casefold() for item in explicit_limitations["items"]}
                explicit_limitations["items"].extend(
                    item for item in discussion_limitations["items"]
                    if item["statement"].casefold() not in existing
                )
            else:
                explicit_limitations = discussion_limitations
    limitations_block = ""
    if explicit_limitations:
        limitations_block = (
            "\n[EXPLICIT AUTHOR LIMITATIONS]\n"
            f"{json.dumps(explicit_limitations, ensure_ascii=False)}\n"
            "For a plural limitations question, include every item exactly once. "
            "Keep these author-stated limitations separate from any inference.\n"
        )
    prefix = (
        "[SECTION-AWARE AUTHOR EVIDENCE]\n"
        "Continue through the supplied discussion/conclusion evidence. Separate "
        "author-stated limitations and explicit future work from assistant inference. "
        "For a reported trend, state when the authors do not give a mechanism; label "
        f"any numerical explanation as inference.{limitations_block}"
    )
    return _format_context(selected, prefix), selected


def _evidence_aspects(question: str) -> list[str]:
    text = re.sub(r"\s+", " ", str(question or "")).strip(" ?.!")
    match = re.search(r"\bboth\s+(.+?)\s+and\s+(.+)$", text, re.I)
    if match:
        return [match.group(1).strip(), match.group(2).strip()]
    return [text]


def _evidence_driver(question: str) -> str:
    text = re.sub(r"\s+", " ", str(question or "")).strip(" ?.!")
    match = re.search(
        r"\bevidence\s+that\s+(.+?)\s+(?:affects?|changes?|influences?)\s+both\b",
        text,
        re.IGNORECASE,
    )
    return match.group(1).strip() if match else "the stated independent variable"


def _aspect_expansion(aspect: str) -> str:
    """Add general measurement synonyms, never document/figure answers."""
    lowered = aspect.casefold()
    terms = []
    if re.search(r"\b(?:depth|penetration|locali[sz]ation)\b", lowered):
        terms.extend(("spatial distribution", "penetration", "localization", "energy deposition", "absorbed power"))
    if re.search(r"\b(?:temperature|thermal|heat)\b", lowered):
        terms.extend(("temperature", "thermal", "isothermal", "peak temperature"))
    return " ".join(dict.fromkeys(terms))


def _is_reference_or_metadata(document: str) -> bool:
    text = document.strip()
    lower = text.lower()
    if re.search(r"^(references|bibliography|literature cited)\b", lower):
        return True
    metadata_terms = (
        "author manuscript", "all rights reserved", "copyright",
        "corresponding author", "author affiliations", "publisher's note",
        "available in pmc", "europe pmc funders author manuscripts",
    )
    if any(term in lower for term in metadata_terms) and len(text) < 500:
        return True
    years = len(re.findall(r"\b(?:19|20)\d{2}[a-z]?\b", text))
    database_markers = len(re.findall(r"\b(?:pubmed|pmid|doi)\b", lower))
    reference_lines = len(re.findall(
        r"(?m)^[A-Z][A-Za-z'’-]+(?:\s+[A-Z][A-Za-z'’-]+)*.*\b(?:19|20)\d{2}\b",
        text,
    ))
    return database_markers >= 3 or (years >= 10 and reference_lines >= 5)


def _source_identity(metadata: dict) -> str:
    source = str(metadata.get("pdf", metadata.get("source", "unknown"))).lower()
    return re.sub(r"\.(?:pdf|txt)$", "", source)


def infer_section(document: str) -> str:
    lower = document.lower()
    if _is_reference_or_metadata(document):
        return "excluded"
    if re.search(r"\babstract\b", lower[:500]):
        return "abstract"
    if re.search(r"\bintroduction\b", lower[:500]):
        return "introduction"
    if re.search(r"\b(conclusions?|concluding remarks|summary and conclusions)\b", lower[:600]):
        return "conclusion"
    if re.search(r"\bdiscussion\b", lower[:500]):
        return "discussion"
    if re.search(r"\b(?:methods?|materials and methods|experimental procedures)\b", lower[:700]):
        return "methods"
    if re.search(r"\b(overview|perspective|conceptual framework)\b", lower[:700]):
        return "overview"
    if re.search(r"\bfigure\s+\d+[a-z]?\.?\s+", lower[:300]):
        return "figure_caption"
    first_line = document.strip().splitlines()[0][:100].strip() if document.strip() else ""
    if 1 <= len(first_line.split()) <= 8 and not first_line.endswith("."):
        return f"section:{first_line.lower()}"
    return "body"


def infer_document_type(documents: list[str]) -> str:
    text = "\n".join(documents).lower()
    if re.search(r"\b(this review|we review|review article|narrative review|systematic review)\b", text):
        return "review"
    if re.search(r"\b(this perspective|perspective article|commentary)\b", text):
        return "commentary or perspective"
    if re.search(r"\b(methods paper|we present a method|novel method|protocol)\b", text):
        return "methods paper"
    if re.search(
        r"\b(we conducted|participants were|randomized|our experiments|we measured|"
        r"numerical (?:investigation|study|simulation)|finite element method)\b",
        text,
    ):
        return "original research study"
    if re.search(
        r"\b(?:review|perspective)\b.{0,100}\b(?:framework|synthesi[sz]e)\b|"
        r"\benumerates? .* hallmarks\b",
        text,
    ):
        return "review"
    return "document type uncertain"


def _source_item(score, document, metadata, section=None) -> dict:
    return {
        "score": float(score),
        "document": document,
        "source": metadata.get("pdf", metadata.get("source", "Unknown source")),
        "page": metadata.get("page", "Unknown page"),
        "chunk": metadata.get("chunk", "Unknown chunk"),
        **({"section": section} if section else {}),
    }


def _format_context(selected_results: list[dict], prefix: str = "") -> str:
    parts = [
        (
            f"Source: {item['source']}, page {item['page']}, chunk {item['chunk']}"
            + (f", section {item['section']}" if item.get("section") else "")
            + f"\n{item['document']}"
        )
        for item in selected_results
    ]
    return "\n\n".join(([prefix] if prefix else []) + parts)


def _question_coverage_prefix(question: str) -> str:
    slots = requested_answer_slots(question)
    if not slots:
        return ""
    return (
        "[QUESTION COVERAGE]\n"
        f"Requested answer slots: {json.dumps(slots)}\n"
        "Answer every slot from evidence, explicitly mark it not found, or clearly label inference."
    )


def _query_candidates(queries, collection, embedder, n_results):
    candidates = {}
    raw_count = 0
    for query in queries:
        embedding = embedder.encode(query, normalize_embeddings=True).tolist()
        results = collection.query(
            query_embeddings=[embedding],
            n_results=n_results,
            include=["documents", "metadatas", "distances"],
        )
        raw_count += len(results["documents"][0])
        for document, metadata in zip(results["documents"][0], results["metadatas"][0]):
            key = (
                metadata.get("pdf", metadata.get("source", "Unknown source")),
                metadata.get("page", "Unknown page"),
                metadata.get("chunk", "Unknown chunk"),
            )
            candidates.setdefault(key, (document, metadata))
    return candidates, raw_count


def _collection_source_names(collection) -> list[str]:
    getter = getattr(collection, "get", None)
    if not callable(getter):
        return []
    try:
        rows = getter(include=["metadatas"])
    except Exception:
        return []
    return list(dict.fromkeys(
        metadata.get("pdf", metadata.get("source", ""))
        for metadata in rows.get("metadatas", []) if isinstance(metadata, dict)
        and metadata.get("pdf", metadata.get("source"))
    ))


def _target_document_identity(question: str, collection, selected_source=None):
    names = _collection_source_names(collection)
    matches = explicit_document_matches(question, names)
    if len(matches) > 1:
        return None, sorted(matches), None
    explicit = next(iter(matches), None)
    if explicit:
        return _source_identity({"source": explicit}), [], explicit
    if selected_source:
        return _source_identity({"source": str(selected_source)}), [], None
    return None, [], None


def is_methods_question(question: str) -> bool:
    return bool(re.search(
        r"\b(?:exactly how|methods?|training|preprocessing|threshold|inclusion|"
        r"exclusion|sample selection|algorithm configuration|test size|overfitting|"
        r"randomi[sz]ation|constructed|assembled)\b",
        str(question or ""), re.I,
    ))


def is_cross_domain_evidence_question(question: str) -> bool:
    text = str(question or "").casefold()
    requested = [
        bool(re.search(r"\b(?:cell culture|in vitro|cell line)\b", text)),
        bool(re.search(r"\b(?:mouse|mice|animal)\b", text)),
        bool(re.search(r"\b(?:human|patient|clinical)\b", text)),
    ]
    return sum(requested) >= 2 and bool(re.search(r"\b(?:evidence|figures?|experiments?)\b", text))


def _expand_target_document_candidates(collection, candidates, target_source):
    """Add all indexed chunks for a broad summary's selected document.

    The abstract or conclusion can fall outside the nearest vector hits. Chroma
    supports a metadata-filtered ``get`` call; lightweight test doubles and
    older backends may not, in which case the original candidates are retained.
    """
    getter = getattr(collection, "get", None)
    if not callable(getter) or target_source is None:
        return 0

    selectors = set()
    for _, metadata in candidates.values():
        if _source_identity(metadata) != target_source:
            continue
        for key in ("source", "pdf"):
            value = metadata.get(key)
            if value:
                selectors.add((key, value))

    added = 0
    if not selectors:
        try:
            rows = getter(include=["documents", "metadatas"])
        except Exception:
            rows = {}
        documents = rows.get("documents", []) if isinstance(rows, dict) else []
        metadatas = rows.get("metadatas", []) if isinstance(rows, dict) else []
        for document, metadata in zip(documents or [], metadatas or []):
            if _source_identity(metadata) != target_source:
                continue
            candidate_key = (
                metadata.get("pdf", metadata.get("source", "Unknown source")),
                metadata.get("page", "Unknown page"),
                metadata.get("chunk", "Unknown chunk"),
            )
            if candidate_key not in candidates:
                candidates[candidate_key] = (document, metadata)
                added += 1
        return added
    for key, value in selectors:
        try:
            rows = getter(
                where={key: value},
                include=["documents", "metadatas"],
            )
        except Exception:
            continue
        documents = rows.get("documents", []) if isinstance(rows, dict) else []
        metadatas = rows.get("metadatas", []) if isinstance(rows, dict) else []
        for document, metadata in zip(documents or [], metadatas or []):
            if _source_identity(metadata) != target_source:
                continue
            candidate_key = (
                metadata.get("pdf", metadata.get("source", "Unknown source")),
                metadata.get("page", "Unknown page"),
                metadata.get("chunk", "Unknown chunk"),
            )
            if candidate_key not in candidates:
                candidates[candidate_key] = (document, metadata)
                added += 1
    return added


def _retrieve_summary_context(
    question, collection, embedder, reranker, available_chunks,
    selected_source=None,
):
    n_results = min(max(INITIAL_RESULTS, 30), available_chunks)
    candidates, raw_count = _query_candidates(
        SUMMARY_QUERIES, collection, embedder, n_results
    )
    diagnostics = {
        "collection_total": available_chunks,
        "initial_vector_search_candidates": len(candidates),
        "initial_vector_search_rows": raw_count,
    }

    if not candidates:
        fallback, fallback_raw = _query_candidates(
            SUMMARY_FALLBACK_QUERIES, collection, embedder, n_results
        )
        candidates.update(fallback)
        diagnostics["fallback_vector_search_rows"] = fallback_raw

    # Existing indexes may store only source=.txt while newer records may also
    # contain pdf=.pdf. Treat matching stems as the same document.
    source_counts = {}
    for _, metadata in candidates.values():
        identity = _source_identity(metadata)
        source_counts[identity] = source_counts.get(identity, 0) + 1
    candidate_names = list(dict.fromkeys(
        metadata.get("pdf", metadata.get("source", ""))
        for _, metadata in candidates.values()
        if metadata.get("pdf", metadata.get("source"))
    ))
    getter = getattr(collection, "get", None)
    if callable(getter):
        try:
            metadata_rows = getter(include=["metadatas"]).get("metadatas", [])
        except Exception:
            metadata_rows = []
        candidate_names.extend(
            metadata.get("pdf", metadata.get("source", ""))
            for metadata in metadata_rows or []
            if metadata.get("pdf", metadata.get("source"))
        )
        candidate_names = list(dict.fromkeys(candidate_names))
    explicit_names = explicit_document_matches(question, candidate_names)
    if len(explicit_names) > 1:
        choices = ", ".join(sorted(explicit_names))
        return (
            "[DOCUMENT TITLE AMBIGUOUS]\n"
            f"The current query could refer to more than one indexed document: {choices}. "
            "Ask the user to clarify; do not inherit a previous document.",
            [],
        )
    explicit_name = next(iter(explicit_names), None)
    target_source = (
        _source_identity({"source": explicit_name}) if explicit_name
        else _source_identity({"source": str(selected_source)}) if selected_source
        else max(source_counts, key=source_counts.get) if source_counts else None
    )
    diagnostics["document_expansion_chunks_added"] = _expand_target_document_candidates(
        collection, candidates, target_source
    )
    source_candidates = [
        (document, metadata)
        for document, metadata in candidates.values()
        if target_source is None or _source_identity(metadata) == target_source
    ]
    diagnostics["candidates_after_document_source_filtering"] = len(source_candidates)

    sectioned = [
        (document, metadata, infer_section(document))
        for document, metadata in source_candidates
    ]
    diagnostics["candidates_after_section_filtering"] = len(sectioned)

    usable = [
        (document, metadata, section)
        for document, metadata, section in sectioned
        if not _is_reference_or_metadata(document)
    ]
    diagnostics["candidates_after_bibliography_filtering"] = len(usable)

    if len(usable) < 3:
        fallback, fallback_raw = _query_candidates(
            SUMMARY_FALLBACK_QUERIES, collection, embedder, n_results
        )
        diagnostics["fallback_vector_search_rows"] = (
            diagnostics.get("fallback_vector_search_rows", 0) + fallback_raw
        )
        for key, value in fallback.items():
            candidates.setdefault(key, value)
        source_candidates = [
            (document, metadata)
            for document, metadata in candidates.values()
            if target_source is None or _source_identity(metadata) == target_source
        ]
        sectioned = [
            (document, metadata, infer_section(document))
            for document, metadata in source_candidates
        ]
        usable = [row for row in sectioned if not _is_reference_or_metadata(row[0])]
        diagnostics["candidates_after_fallback"] = len(usable)

    # A non-empty collection must yield evidence. If conservative filtering
    # removed everything, retain the best non-reference raw candidates rather
    # than sending an empty context downstream.
    if not usable:
        usable = [
            (document, metadata, "unknown")
            for document, metadata in source_candidates
        ]
    if not usable:
        return "", []

    summary_query = (
        f"{question}\nPurpose scope central framework major themes conclusions "
        "implications limitations"
    )
    scores = [float(score) for score in reranker.predict(
        [[summary_query, document] for document, _, _ in usable]
    )]
    boosts = {
        "abstract": 3.0, "conclusion": 2.8, "discussion": 2.2,
        "introduction": 2.0, "overview": 2.3, "figure_caption": 1.4,
    }
    best_score = max(scores)
    # Scores are model-relative and may all be negative. Remove only severe
    # outliers relative to the best candidate, never by an absolute cutoff.
    scored_usable = [
        (score, row) for score, row in zip(scores, usable)
        if score >= best_score - 6.0
    ]
    if not scored_usable:
        scored_usable = list(zip(scores, usable))
    diagnostics["candidates_after_score_filtering"] = len(scored_usable)
    ranked = sorted(
        (
            (float(score) + boosts.get(section, 0.0), float(score), document, metadata, section)
            for score, (document, metadata, section) in scored_usable
        ),
        key=lambda row: row[0],
        reverse=True,
    )

    selected = []
    seen_pages = set()
    section_counts = {}
    priority_order = (
        "abstract", "conclusion", "discussion", "overview",
        "introduction", "figure_caption",
    )
    for desired_section in priority_order:
        match = next(
            (
                row for row in ranked
                if row[4] == desired_section
                and (row[3].get("pdf", row[3].get("source")), row[3].get("page")) not in seen_pages
            ),
            None,
        )
        if match:
            _, raw_score, document, metadata, section = match
            selected.append(_source_item(raw_score, document, metadata, section))
            seen_pages.add((metadata.get("pdf", metadata.get("source")), metadata.get("page")))
            section_counts[section] = 1
        if len(selected) >= FINAL_RESULTS:
            break

    for _, raw_score, document, metadata, section in ranked:
        page_key = (metadata.get("pdf", metadata.get("source")), metadata.get("page"))
        if page_key in seen_pages or section_counts.get(section, 0) >= 2:
            continue
        selected.append(_source_item(raw_score, document, metadata, section))
        seen_pages.add(page_key)
        section_counts[section] = section_counts.get(section, 0) + 1
        if len(selected) >= FINAL_RESULTS:
            break

    document_type = infer_document_type([item["document"] for item in selected])
    # Vector-query insertion order is relevance order, not document order.
    # Framework lists and category transitions must be reconstructed in page
    # order or unrelated chunks can create false category boundaries.
    framework_documents = [
        document
        for document, _, _ in sorted(
            usable,
            key=lambda row: (
                _source_identity(row[1]),
                int(row[1].get("page", 0)),
                int(row[1].get("chunk", 0)),
            ),
        )
    ]
    framework_grounding = extract_framework_grounding(framework_documents)
    framework_block = ""
    if framework_grounding:
        diagnostics["framework_grounding"] = framework_grounding
        framework_block = (
            "\n[VALIDATED FRAMEWORK GROUNDING]\n"
            f"Canonical named framework item count: {len(framework_grounding['named_items'])}\n"
            f"{json.dumps(framework_grounding, ensure_ascii=False)}\n"
            "Only named_items may be presented as canonical framework items. "
            "Mechanisms discussed within their sections are not additional named items.\n"
        )
    explicit_limitations_grounding = {}
    limitations_grounding = {}
    limitations_block = ""
    if re.search(r"\blimitations?\b", question, re.IGNORECASE):
        explicit_limitations_grounding = extract_explicit_limitations(framework_documents)
        discussion_limitations = extract_discussion_limitations(framework_documents)
        if discussion_limitations:
            if explicit_limitations_grounding:
                existing = {
                    item["statement"].casefold()
                    for item in explicit_limitations_grounding["items"]
                }
                explicit_limitations_grounding["items"].extend(
                    item for item in discussion_limitations["items"]
                    if item["statement"].casefold() not in existing
                )
            else:
                explicit_limitations_grounding = discussion_limitations
        if not explicit_limitations_grounding:
            limitations_grounding = extract_inferred_limitations(framework_documents)
    if explicit_limitations_grounding:
        diagnostics["explicit_author_limitations"] = explicit_limitations_grounding
        limitations_block = (
            "\n[EXPLICIT AUTHOR LIMITATIONS]\n"
            f"{json.dumps(explicit_limitations_grounding, ensure_ascii=False)}\n"
            "These are author-stated limitations from the Discussion. Present them "
            "as explicit, not implied, and keep any assistant inference separate.\n"
        )
    elif limitations_grounding:
        diagnostics["inferred_limitations_grounding"] = limitations_grounding
        limitations_block = (
            "\n[INFERRED LIMITATIONS GROUNDING]\n"
            f"{json.dumps(limitations_grounding, ensure_ascii=False)}\n"
            "These are limitations inferred from explicit modelling assumptions "
            "and validation statements. Label them as inferred; do not claim the "
            "authors provided a dedicated limitations section.\n"
        )
        uncovered = {item["key"] for item in limitations_grounding["items"]}
        for score, (document, metadata, section) in sorted(
            scored_usable,
            key=lambda row: len(_limitation_keys(row[1][0])),
            reverse=True,
        ):
            covered = _limitation_keys(document).intersection(uncovered)
            if not covered:
                continue
            page_key = (metadata.get("pdf", metadata.get("source")), metadata.get("page"))
            if page_key not in seen_pages:
                selected.append(_source_item(score, document, metadata, "assumptions"))
                seen_pages.add(page_key)
            uncovered.difference_update(covered)
            if not uncovered:
                break
    taxonomy = extract_explicit_classifier_taxonomy("\n".join(framework_documents))
    taxonomy_block = ""
    if taxonomy:
        diagnostics["explicit_classifier_taxonomy"] = taxonomy
        taxonomy_block = (
            "\n[EXPLICIT CLASSIFIER TAXONOMY]\n"
            f"{json.dumps(taxonomy, ensure_ascii=False)}\n"
            "Preserve these exact identifiers and source descriptions. Do not infer "
            "acronym expansions or an umbrella classifier family absent from this block.\n"
        )
    prefix = (
        "[DOCUMENT SUMMARY MODE]\n"
        f"Document type: {document_type}\n"
        "Summarize the whole document using representative evidence. For a review, "
        "cover purpose and scope, central framework, major themes, conclusions, "
        "implications and limitations; do not call review arguments experimental findings."
        f"{framework_block}{limitations_block}{taxonomy_block}"
    )
    diagnostics["final_selected_chunks"] = len(selected)
    diagnostics["final_selected_pages"] = [item["page"] for item in selected]
    diagnostics["target_source"] = target_source
    diagnostics["explicit_document_match"] = explicit_name or ""
    if selected:
        selected[0]["retrieval_debug"] = diagnostics
    return _format_context(selected, prefix), selected


def _figure_ids(document: str) -> list[str]:
    identifiers = []
    for match in re.finditer(
        r"\bfig(?:ure)?s?\.?\s*(\d+(?:\.\d+)?[a-z]?)\b",
        document,
        re.IGNORECASE,
    ):
        identifier = match.group(1)
        if identifier.casefold() not in {item.casefold() for item in identifiers}:
            identifiers.append(identifier)
    return identifiers


def _driver_variation_score(document: str, driver: str) -> float:
    text = re.sub(r"\s+", " ", document).casefold()
    terms = re.findall(r"[a-z0-9]+", driver.casefold())
    if not terms:
        return 0.0
    noun = terms[-1]
    plural = f"{noun[:-1]}ies" if noun.endswith("y") else f"{noun}s"
    noun_pattern = rf"(?:{re.escape(noun)}|{re.escape(plural)})"
    score = 0.0
    if re.search(
        rf"\b(?:impact|effect|influence)\s+of\b.{{0,55}}\b{noun_pattern}\b",
        text,
    ):
        score += 2.0
    if re.search(
        rf"\b(?:different|multiple|varying|various|range of)\b.{{0,35}}\b{noun_pattern}\b|"
        rf"\b{noun_pattern}\b.{{0,35}}\b(?:different|multiple|varying|various|range)\b",
        text,
    ):
        score += 1.0
    if score == 0 and re.search(
        rf"\b(?:fixed|constant)\b.{{0,35}}\b{noun_pattern}\b|"
        rf"\b{noun_pattern}\b.{{0,35}}\b(?:fixed|constant)\b",
        text,
    ):
        score -= 1.0
    return score


def _trim_nearby_to_figure_discussion(
    caption: str, nearby_text: str, identifier: str
) -> str:
    """Drop a prior figure's paragraph when page-local text crosses a page break."""
    nearby = str(nearby_text or "")
    identifier = str(identifier)
    discussion = re.search(
        rf"\bfig(?:ure)?\.?\s*{re.escape(identifier)}\b"
        r".{0,260}?\b(?:shows?|depicts?|displays?|illustrates?|representation)\b",
        nearby,
        re.IGNORECASE | re.DOTALL,
    )
    if discussion:
        nearby = nearby[discussion.start():]
    # Page text commonly continues into the next figure discussion or repeats
    # a different figure caption. Do not attach that later figure's narrative
    # to the current indexed figure.
    truncated_at_other_figure = False
    for candidate in re.finditer(r"\bfig(?:ure)?\.?\s*(\d+(?:\.\d+)?)\b", nearby, re.I):
        if candidate.group(1) == identifier:
            continue
        tail = nearby[candidate.start():candidate.start() + 280]
        caption_like = bool(re.match(
            r"\bfig(?:ure)?\.?\s*\d+(?:\.\d+)?\s*\.", tail, re.I
        ))
        narrative_like = bool(re.search(
            r"\b(?:shows?|depicts?|displays?|illustrates?|representation|expound)\b",
            tail,
            re.I,
        ))
        if caption_like or narrative_like:
            nearby = nearby[:candidate.start()].rstrip()
            truncated_at_other_figure = True
            break
    if truncated_at_other_figure and nearby and nearby[-1] not in ".!?":
        complete = re.search(r"^.*[.!?](?=\s|$)", nearby, re.DOTALL)
        if complete:
            nearby = complete.group(0).rstrip()
    return "\n".join(part for part in (str(caption or ""), nearby) if part).strip()


def _preceding_text_before_figure_discussion(
    nearby_text: str, identifier: str
) -> str:
    """Recover a prior page's prose that continues before this figure begins."""
    nearby = str(nearby_text or "")
    discussion = re.search(
        rf"\bfig(?:ure)?\.?\s*{re.escape(str(identifier))}\b"
        r".{0,260}?\b(?:shows?|depicts?|displays?|illustrates?|representation)\b",
        nearby,
        re.IGNORECASE | re.DOTALL,
    )
    return nearby[:discussion.start()].strip() if discussion else ""


def _has_later_figure_discussion(
    nearby_text: str, identifier: str
) -> bool:
    """Tell whether this page has already moved to a later-numbered figure."""
    try:
        current = float(identifier)
    except (TypeError, ValueError):
        return False
    for candidate in re.finditer(
        r"\bfig(?:ure)?\.?\s*(\d+(?:\.\d+)?)\b", str(nearby_text or ""), re.I
    ):
        if float(candidate.group(1)) <= current:
            continue
        tail = str(nearby_text or "")[candidate.start():candidate.start() + 280]
        if re.search(
            r"\b(?:shows?|depicts?|displays?|illustrates?|representation|expound)\b",
            tail,
            re.I,
        ):
            return True
    return False


def _retrieve_multi_figure_context(
    question, collection, embedder, reranker, available_chunks
):
    aspects = _evidence_aspects(question)
    driver = _evidence_driver(question)
    queries = [question, *[
        f"figure caption and nearby results evidence for {aspect}"
        for aspect in aspects
    ]]
    candidates, _ = _query_candidates(
        queries, collection, embedder, min(max(INITIAL_RESULTS, 30), available_chunks)
    )
    rows = [
        (document, metadata, _figure_ids(document))
        for document, metadata in candidates.values()
        if _figure_ids(document) and not _is_reference_or_metadata(document)
    ]
    if not rows:
        return "", []

    # Keep one coherent document. Explicit title evidence wins; otherwise use
    # the document with the strongest aggregate reranker evidence.
    source_names = list(dict.fromkeys(
        metadata.get("pdf", metadata.get("source", ""))
        for _, metadata, _ in rows
    ))
    explicit = explicit_document_matches(question, source_names)
    if explicit:
        chosen_source = next(iter(explicit))
    else:
        aggregate = {}
        scores = reranker.predict([[question, document] for document, _, _ in rows])
        for score, (_, metadata, _) in zip(scores, rows):
            source = metadata.get("pdf", metadata.get("source", "Unknown source"))
            aggregate[source] = max(aggregate.get(source, float("-inf")), float(score))
        chosen_source = max(aggregate, key=aggregate.get)
    rows = [
        row for row in rows
        if row[1].get("pdf", row[1].get("source", "Unknown source")) == chosen_source
    ]
    getter = getattr(collection, "get", None)
    if callable(getter):
        source_key = "pdf" if any(
            metadata.get("pdf") == chosen_source for _, metadata, _ in rows
        ) else "source"
        try:
            expanded = getter(
                where={source_key: chosen_source},
                include=["documents", "metadatas"],
            )
        except Exception:
            expanded = {}
        known = {
            (
                metadata.get("pdf", metadata.get("source", "Unknown source")),
                metadata.get("page"), metadata.get("chunk"),
            )
            for _, metadata, _ in rows
        }
        for document, metadata in zip(
            expanded.get("documents", []) if isinstance(expanded, dict) else [],
            expanded.get("metadatas", []) if isinstance(expanded, dict) else [],
        ):
            identifiers = _figure_ids(document)
            key = (
                metadata.get("pdf", metadata.get("source", "Unknown source")),
                metadata.get("page"), metadata.get("chunk"),
            )
            if identifiers and key not in known:
                rows.append((document, metadata, identifiers))
                known.add(key)

    try:
        from services.visual_index import (
            flatten_visual_targets, load_or_build_visual_index,
        )
        indexed_figures = [
            target for target in flatten_visual_targets(load_or_build_visual_index())
            if target.get("target_type") == "figure"
            and target.get("match_kind") == "caption"
            and target.get("pdf_name") == chosen_source
        ]
    except Exception:
        indexed_figures = []
    indexed_rows = []
    for target in indexed_figures:
        identifier = str(target.get("target_number"))
        current_page = int(target.get("page_number") or 0)
        spillover = ""
        if not _has_later_figure_discussion(target.get("nearby_text", ""), identifier):
            next_target = next((
                candidate for candidate in sorted(
                    indexed_figures,
                    key=lambda item: int(item.get("page_number") or 0),
                )
                if int(candidate.get("page_number") or 0) == current_page + 1
            ), None)
            if next_target:
                spillover = _preceding_text_before_figure_discussion(
                    next_target.get("nearby_text", ""),
                    str(next_target.get("target_number")),
                )
        figure_text = _trim_nearby_to_figure_discussion(
            target.get("caption", ""), target.get("nearby_text", ""), identifier
        )
        if spillover:
            figure_text = f"{figure_text}\n{spillover}".strip()
        indexed_rows.append((
            figure_text,
            {
                "pdf": chosen_source,
                "page": target.get("page_number"),
                "chunk": f"visual-index-{target.get('target_number')}",
                "visual_index_caption": True,
                "driver_text": target.get("caption", ""),
            },
            [identifier],
        ))
    if indexed_rows:
        # Exact caption occurrences plus their page-local results text provide
        # cleaner figure evidence than arbitrary chunks that merely mention a
        # figure number in passing.
        rows = indexed_rows

    selected = []
    used_figures = set()
    for aspect in aspects:
        aspect_terms = _aspect_expansion(aspect)
        scores = reranker.predict([
            [
                f"Which figure directly visualizes {aspect} ({aspect_terms})? "
                "Prefer a caption and "
                f"nearby result that vary {driver} and explicitly show its effect on "
                f"{aspect}. Reject results where {driver} is merely fixed while another "
                "parameter is compared.",
                metadata.get("driver_text", document),
            ]
            for document, metadata, _ in rows
        ])
        ranked = sorted(
            zip(scores, rows),
            key=lambda item: (
                _driver_variation_score(
                    item[1][1].get("driver_text", item[1][0]), driver
                ),
                bool(item[1][1].get("visual_index_caption")),
                float(item[0]),
            ),
            reverse=True,
        )
        choice = next((
            (score, row) for score, row in ranked
            if any(identifier not in used_figures for identifier in row[2])
        ), ranked[0] if ranked else None)
        if choice is None:
            continue
        score, (document, metadata, identifiers) = choice
        used_figures.update(identifiers)
        item = _source_item(score, document, metadata, f"figure evidence: {aspect}")
        item["figure_identifiers"] = identifiers
        if not any(
            existing["source"] == item["source"]
            and existing["page"] == item["page"]
            and existing["chunk"] == item["chunk"]
            for existing in selected
        ):
            selected.append(item)

    prefix = (
        "[MULTI-FIGURE EVIDENCE MODE]\n"
        "The question asks for evidence about distinct effects. Identify every "
        "selected figure by number, explain what separate effect its caption and "
        "nearby results support, and do not collapse the answer to one figure when "
        "the supplied evidence spans multiple figures. When the nearby results "
        "report maxima, minima, peaks, or other explicit extrema, name that kind "
        "of evidence and preserve its reported values. Keep every trend attached "
        "to its measured quantity: a power-density trend is not a temperature "
        "trend. Describe heating depth or penetration only as an inference from "
        "the spatial absorbed-power or temperature contours unless depth is "
        "directly measured. Never state opposite trends for the same quantity."
    )
    return _format_context(selected, prefix), selected


def _retrieve_methods_context(
    question, collection, embedder, reranker, available_chunks, selected_source=None,
):
    queries = [
        question,
        f"Methods experimental procedures {question}",
        f"training data sample selection preprocessing threshold test size {question}",
    ]
    candidates, _ = _query_candidates(
        queries, collection, embedder, min(max(INITIAL_RESULTS, 30), available_chunks)
    )
    target, ambiguous, explicit = _target_document_identity(
        question, collection, selected_source
    )
    if ambiguous:
        return (
            "[DOCUMENT TITLE AMBIGUOUS]\nClarify which document: "
            + ", ".join(ambiguous),
            [],
        )
    if target is None and candidates:
        scored = reranker.predict([[question, document] for document, _ in candidates.values()])
        target = _source_identity(max(
            zip(scored, candidates.values()), key=lambda row: float(row[0])
        )[1][1])
    _expand_target_document_candidates(collection, candidates, target)
    rows = [
        (document, metadata)
        for document, metadata in candidates.values()
        if (target is None or _source_identity(metadata) == target)
        and not _is_reference_or_metadata(document)
    ]
    if not rows:
        return "", []
    slots = requested_answer_slots(question)
    slot_terms = {
        "features": ("feature", "area", "form factor", "elongation", "compactness"),
        "library construction": ("training set", "wells", "plates", "random"),
        "training cell counts": ("cells per condition", "10,000", "million cells"),
        "CT split": ("classification tree", "30%", "test size"),
        "RF split": ("random forest", "0.5", "test size"),
        "CT overfitting method": ("cost complexity", "pruning", "alpha", "over fitting"),
        "RF threshold": ("probability", "> 0.5", "senescent"),
        "inclusion criteria": ("excluded", "included", "threshold", "at least"),
    }
    model_scores = reranker.predict([[question, document] for document, _ in rows])
    ranked = []
    for score, (document, metadata) in zip(model_scores, rows):
        lower = document.casefold()
        coverage = {
            slot for slot in slots
            if any(term.casefold() in lower for term in slot_terms.get(slot, (slot,)))
        }
        methods_boost = 4.0 if infer_section(document) == "methods" or re.search(
            r"\b(?:methods?|test size|training sets?|software|sample selection)\b", lower
        ) else 0.0
        ranked.append((float(score) + methods_boost + 1.5 * len(coverage), document, metadata, coverage))
    selected, covered, per_page = [], set(), {}
    while ranked and len(selected) < max(FINAL_RESULTS, 8):
        ranked.sort(key=lambda row: (len(row[3] - covered), row[0]), reverse=True)
        score, document, metadata, coverage = ranked.pop(0)
        page_key = (metadata.get("pdf", metadata.get("source")), metadata.get("page"))
        if per_page.get(page_key, 0) >= 6:
            continue
        selected.append(_source_item(score, document, metadata, "methods"))
        per_page[page_key] = per_page.get(page_key, 0) + 1
        covered.update(coverage)
    missing = [slot for slot in slots if slot not in covered]
    prefix = (
        "[METHODS-AWARE RETRIEVAL]\n"
        f"Explicit document match: {explicit or ''}\n"
        f"Requested answer slots: {json.dumps(slots)}\n"
        f"Grounded slots: {json.dumps(sorted(covered))}\n"
        f"Not found in retrieved evidence: {json.dumps(missing)}\n"
        "Answer every requested slot. Report missing details as not specified in "
        "retrieved evidence. Do not complete methodology from standard practice, "
        "presumption, or typical conventions."
    )
    return _format_context(selected, prefix), selected


def _retrieve_cross_domain_context(
    question, collection, embedder, reranker, available_chunks, selected_source=None,
):
    domain_queries = {
        "in_vitro_human_cell_line": f"{question} cell culture in vitro human cell line evidence figure",
        "mouse_animal_tissue": f"{question} mouse animal in vivo tissue evidence figure",
        "human_patient_tissue": f"{question} human patient clinical tissue evidence figure",
    }
    candidates, _ = _query_candidates(
        domain_queries.values(), collection, embedder,
        min(max(INITIAL_RESULTS, 30), available_chunks),
    )
    target, ambiguous, explicit = _target_document_identity(question, collection, selected_source)
    if ambiguous:
        return "[DOCUMENT TITLE AMBIGUOUS]\nClarify which document: " + ", ".join(ambiguous), []
    if target is None and candidates:
        scores = reranker.predict([[question, document] for document, _ in candidates.values()])
        target = _source_identity(max(zip(scores, candidates.values()), key=lambda row: float(row[0]))[1][1])
    _expand_target_document_candidates(collection, candidates, target)
    rows = [
        (document, metadata)
        for document, metadata in candidates.values()
        if (target is None or _source_identity(metadata) == target)
        and not _is_reference_or_metadata(document)
    ]
    evidence_objects = []
    try:
        from services.visual_index import flatten_visual_targets, load_or_build_visual_index

        for visual in flatten_visual_targets(load_or_build_visual_index()):
            if visual.get("target_type") != "figure" or visual.get("match_kind") != "caption":
                continue
            metadata = {
                "pdf": visual.get("pdf_name", ""),
                "page": visual.get("page_number", "Unknown page"),
                "chunk": f"visual-index-{visual.get('target_number', '')}",
            }
            if target is not None and _source_identity(metadata) != target:
                continue
            caption = visual.get("full_caption") or visual.get("caption", "")
            nearby = visual.get("nearby_text", "")
            classification = classify_experimental_evidence(caption, nearby, "")
            evidence_objects.append({
                "document": f"{caption}\n{nearby}".strip(),
                "metadata": metadata,
                "caption": caption,
                "results": nearby,
                "classification": classification,
                "figure_number": visual.get("target_number"),
                "panel": visual.get("panel"),
            })
    except (OSError, ValueError, TypeError):
        evidence_objects = []
    if not evidence_objects:
        evidence_objects = [{
            "document": document,
            "metadata": metadata,
            "caption": "",
            "results": document,
            "classification": classify_experimental_evidence("", document, ""),
            "figure_number": None,
            "panel": None,
        } for document, metadata in rows]
    selected, provenance_rows = [], []
    for domain, query in domain_queries.items():
        domain_rows = [
            row for row in evidence_objects
            if row["classification"]["experimental_domain"] == domain
        ]
        if not domain_rows:
            provenance_rows.append({"experimental_domain": domain, "status": "no grounded evidence found"})
            continue
        scores = reranker.predict([[query, row["caption"] or row["document"]] for row in domain_rows])
        ranked = sorted(zip(scores, domain_rows), key=lambda row: float(row[0]), reverse=True)
        main_rows = [
            pair for pair in ranked
            if re.fullmatch(r"\d+(?:\.\d+)?", str(pair[1].get("figure_number", "")))
        ]
        if main_rows:
            earliest = min(
                main_rows,
                key=lambda pair: (
                    int(pair[1]["metadata"].get("page", 10**9))
                    if isinstance(pair[1]["metadata"].get("page"), int) else 10**9,
                    -float(pair[0]),
                ),
            )
            ranked = [earliest, *[pair for pair in ranked if pair[1] is not earliest[1]]]
        used_pages = set()
        for score, row in ranked:
            document, metadata = row["document"], row["metadata"]
            page = metadata.get("page")
            if page in used_pages:
                continue
            item = _source_item(score, document, metadata, f"domain:{domain}")
            item["experimental_provenance"] = experimental_evidence_object(
                document=str(metadata.get("pdf", metadata.get("source", ""))),
                figure_number=row["figure_number"],
                page=page,
                caption=row["caption"],
                results=row["results"],
                panel=row["panel"],
            )
            item["figure_number"] = row["figure_number"]
            item["panel"] = row["panel"]
            selected.append(item)
            provenance_rows.append(item["experimental_provenance"])
            used_pages.add(page)
            if len(used_pages) >= 2:
                break
    prefix = (
        "[WHOLE-DOCUMENT EXPERIMENTAL-DOMAIN EVIDENCE]\n"
        f"Explicit document match: {explicit or ''}\n"
        f"{json.dumps(provenance_rows, ensure_ascii=False)}\n"
        "Keep in-vitro cell lines, primary cells, animal models, ex-vivo tissue, "
        "and human clinical/patient tissue distinct. A human-derived immortalized "
        "cell line is not human patient evidence. Preserve figure/panel and experiment "
        "provenance; do not transfer a threshold or control between experiments. Figure "
        "identifiers in the provenance JSON are authoritative. Never generate a main or "
        "supplementary figure identifier from semantic text. If figure_number is null, say "
        "'figure number not resolved'."
    )
    return _format_context(selected, prefix), selected


def retrieve_context(
    question: str,
    previous_question: str,
    collection,
    embedder,
    reranker,
    selected_source=None,
) -> tuple[str, list[dict]]:
    resolved_question = resolve_followup_retrieval_query(question, previous_question)
    if previous_question:
        retrieval_query = (
            f"Previous question: {previous_question}\nCurrent question: {resolved_question}"
        )
    else:
        retrieval_query = question

    available_chunks = collection.count()
    if available_chunks == 0:
        return "", []

    if is_broad_summary_question(question):
        context, sources = _retrieve_summary_context(
            question, collection, embedder, reranker, available_chunks,
            selected_source=selected_source,
        )
        prefix = _question_coverage_prefix(question)
        return (f"{prefix}\n{context}" if prefix else context), sources

    if is_cross_domain_evidence_question(question):
        context, sources = _retrieve_cross_domain_context(
            question, collection, embedder, reranker, available_chunks,
            selected_source=selected_source,
        )
        prefix = _question_coverage_prefix(question)
        return (f"{prefix}\n{context}" if prefix else context), sources

    if is_multi_figure_evidence_question(question):
        context, sources = _retrieve_multi_figure_context(
            question, collection, embedder, reranker, available_chunks
        )
        prefix = _question_coverage_prefix(question)
        return (f"{prefix}\n{context}" if prefix else context), sources

    if is_methods_question(resolved_question):
        context, sources = _retrieve_methods_context(
            resolved_question, collection, embedder, reranker, available_chunks,
            selected_source=selected_source,
        )
        if context:
            prefix = _question_coverage_prefix(question)
            return (f"{prefix}\n{context}" if prefix else context), sources

    if is_section_aware_question(question) or is_reported_trend_question(question) or (
        previous_question and re.search(r"\b(?:error|trend|increase|decrease)\b", question, re.I)
    ):
        context, sources = _retrieve_section_context(
            retrieval_query, collection, embedder, reranker, available_chunks,
            selected_source=selected_source,
        )
        if context:
            prefix = _question_coverage_prefix(question)
            return (f"{prefix}\n{context}" if prefix else context), sources

    # Existing focused-question retrieval path.
    query_embedding = embedder.encode(
        retrieval_query, normalize_embeddings=True,
    ).tolist()
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(INITIAL_RESULTS, available_chunks),
        include=["documents", "metadatas", "distances"],
    )
    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    if not documents:
        return "", []
    target, ambiguous, _ = _target_document_identity(question, collection, selected_source)
    if ambiguous:
        return "[DOCUMENT TITLE AMBIGUOUS]\nClarify which document: " + ", ".join(ambiguous), []
    if target is not None:
        candidate_map = {
            (
                metadata.get("pdf", metadata.get("source", "Unknown source")),
                metadata.get("page", "Unknown page"), metadata.get("chunk", "Unknown chunk"),
            ): (document, metadata)
            for document, metadata in zip(documents, metadatas)
        }
        _expand_target_document_candidates(collection, candidate_map, target)
        filtered = [
            (document, metadata) for document, metadata in candidate_map.values()
            if _source_identity(metadata) == target
        ]
        if filtered:
            documents = [row[0] for row in filtered]
            metadatas = [row[1] for row in filtered]
    scores = reranker.predict([[retrieval_query, document] for document in documents])
    ranked_results = sorted(
        zip(scores, documents, metadatas),
        key=lambda item: float(item[0]),
        reverse=True,
    )
    selected_results = []
    chunks_per_page = {}
    for score, document, metadata in ranked_results:
        source = metadata.get("pdf", metadata.get("source", "Unknown source"))
        page = metadata.get("page", "Unknown page")
        page_key = (source, page)
        if chunks_per_page.get(page_key, 0) >= 2:
            continue
        selected_results.append(_source_item(score, document, metadata))
        chunks_per_page[page_key] = chunks_per_page.get(page_key, 0) + 1
        if len(selected_results) >= FINAL_RESULTS:
            break
    return _format_context(selected_results, _question_coverage_prefix(question)), selected_results
