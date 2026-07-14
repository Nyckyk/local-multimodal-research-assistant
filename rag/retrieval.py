import json
import re

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
        r"\bmain\s+(?:findings|conclusions|contributions)\b",
        r"\b(?:give|provide)\s+me\s+an?\s+overview\b",
        r"\boverview\s+of\s+(?:this|the)\s+paper\b",
        r"\bwhole[- ]document\s+summary\b",
    )
    return any(re.search(pattern, text) for pattern in patterns)


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
    if re.search(r"\b(we conducted|participants were|randomized|our experiments|we measured)\b", text):
        return "original research study"
    if re.search(r"\bframework|synthesi[sz]e|enumerates? .* hallmarks\b", text):
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


def _retrieve_summary_context(question, collection, embedder, reranker, available_chunks):
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
    target_source = max(source_counts, key=source_counts.get) if source_counts else None
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
    prefix = (
        "[DOCUMENT SUMMARY MODE]\n"
        f"Document type: {document_type}\n"
        "Summarize the whole document using representative evidence. For a review, "
        "cover purpose and scope, central framework, major themes, conclusions, "
        "implications and limitations; do not call review arguments experimental findings."
        f"{framework_block}"
    )
    diagnostics["final_selected_chunks"] = len(selected)
    diagnostics["final_selected_pages"] = [item["page"] for item in selected]
    diagnostics["target_source"] = target_source
    if selected:
        selected[0]["retrieval_debug"] = diagnostics
    return _format_context(selected, prefix), selected


def retrieve_context(
    question: str,
    previous_question: str,
    collection,
    embedder,
    reranker,
) -> tuple[str, list[dict]]:
    if previous_question:
        retrieval_query = (
            f"Previous question: {previous_question}\nCurrent question: {question}"
        )
    else:
        retrieval_query = question

    available_chunks = collection.count()
    if available_chunks == 0:
        return "", []

    if is_broad_summary_question(question):
        return _retrieve_summary_context(
            question, collection, embedder, reranker, available_chunks
        )

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
    return _format_context(selected_results), selected_results
