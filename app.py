from pathlib import Path
import re

import streamlit as st

from rag.database import get_collection
from rag.embeddings import load_embedder, load_reranker
from rag.ingestion import index_pdf
from rag.retrieval import retrieve_context
from services.ollama_service import generate_answer
from services.query_rewriter import rewrite_question
from services.visual_fallback import resolve_with_visual_fallback
from services.visual_index import load_or_build_visual_index, visual_index_needs_rebuild
from services.visual_locator import (
    clear_visual_target_state,
    format_resolution_problem,
    manual_visual_resolution,
    resolve_visual_target,
    should_activate_automatic_vision,
)
from services.visual_reference_parser import has_visual_reference
from services.visual_runtime import analyse_resolved_visual
from services.vision_service import COULD_NOT_VERIFY_MESSAGE
from settings import MAX_HISTORY_MESSAGES, PAPERS_FOLDER


st.set_page_config(
    page_title="Local Research Assistant",
    page_icon="📚",
    layout="wide",
)

st.title("Local Research Assistant")
st.caption(
    "Private research-paper chat powered by Ollama, "
    "ChromaDB and local embeddings"
)


# ---------------------------------------------------------
# Load database and models
# ---------------------------------------------------------

@st.cache_resource
def load_components():
    collection = get_collection()
    embedder = load_embedder()
    reranker = load_reranker()

    return collection, embedder, reranker


collection, embedder, reranker = load_components()


@st.cache_data(show_spinner=False)
def load_visual_library(library_signature):
    # The signature is a Streamlit cache key; the index service independently
    # validates and rebuilds only changed local PDFs.
    del library_signature
    return load_or_build_visual_index()


# ---------------------------------------------------------
# Session state
# ---------------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

if "last_user_question" not in st.session_state:
    st.session_state.last_user_question = ""

if "pending_visual_resolution" not in st.session_state:
    st.session_state.pending_visual_resolution = None

if st.session_state.get("pending_preferred_pdf_name"):
    st.session_state.preferred_pdf_name = st.session_state.pop(
        "pending_preferred_pdf_name"
    )


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------

with st.sidebar:
    st.header("Library")

    uploaded_files = st.file_uploader(
        "Upload research papers",
        type=["pdf"],
        accept_multiple_files=True,
        help=(
            "Uploaded PDFs are saved locally, processed "
            "and added to the research database."
        ),
    )

    if uploaded_files:
        if st.button(
            "Add PDFs to library",
            type="primary",
            use_container_width=True,
        ):
            progress_bar = st.progress(0)
            status_box = st.empty()

            successful = []
            failed = []
            total_files = len(uploaded_files)

            for index, uploaded_file in enumerate(
                uploaded_files,
                start=1,
            ):
                status_box.write(
                    f"Processing {uploaded_file.name}..."
                )

                try:
                    result = index_pdf(
                        collection=collection,
                        embedder=embedder,
                        pdf_filename=uploaded_file.name,
                        pdf_bytes=uploaded_file.getvalue(),
                    )

                    successful.append(result)

                except Exception as error:
                    failed.append(
                        {
                            "filename": uploaded_file.name,
                            "error": str(error),
                        }
                    )

                progress_bar.progress(index / total_files)

            progress_bar.empty()
            status_box.empty()

            for result in successful:
                st.success(
                    f"Indexed {result['filename']}: "
                    f"{result['pages']} pages and "
                    f"{result['chunks']} chunks."
                )

            for result in failed:
                st.error(
                    f"Failed to index {result['filename']}: "
                    f"{result['error']}"
                )

    st.divider()

    pdf_files = sorted(PAPERS_FOLDER.glob("*.pdf"))

    st.metric(
        "Indexed chunks",
        collection.count(),
    )

    st.metric(
        "PDF files",
        len(pdf_files),
    )

    if pdf_files:
        st.subheader("Papers")

        for pdf_file in pdf_files:
            st.write(f"• {pdf_file.name}")
    else:
        st.info("No PDFs have been added.")

    st.divider()
    st.subheader("Vision analysis")

    automatic_visual_detection = st.checkbox(
        "Automatic visual target detection",
        value=True,
        help=(
            "Automatically locate explicit figure/table references and "
            "route them through validated local vision analysis."
        ),
    )

    preferred_pdf_name = st.selectbox(
        "Preferred PDF (optional)",
        options=["No preference", *[pdf_file.name for pdf_file in pdf_files]],
        key="preferred_pdf_name",
        help="A preference for automatic resolution, not an unconditional choice.",
    )
    preferred_pdf = (
        PAPERS_FOLDER / preferred_pdf_name
        if preferred_pdf_name != "No preference" else None
    )

    manual_visual_override = st.checkbox(
        "Use manual vision override",
        value=False,
        help="Use the selected PDF and page instead of automatic resolution.",
    )
    manual_pdf = None
    manual_page_number = 1
    if manual_visual_override and pdf_files:
        manual_pdf_name = st.selectbox(
            "Manual PDF",
            options=[pdf_file.name for pdf_file in pdf_files],
        )
        manual_pdf = PAPERS_FOLDER / manual_pdf_name
        manual_page_number = st.number_input(
            "Manual PDF page number", min_value=1, value=1, step=1,
        )

    vision_debug_enabled = st.checkbox(
        "Save vision debug crops",
        value=False,
        help="Keep temporary inference crops and show their coordinates.",
    )

    pending = st.session_state.get("pending_visual_resolution")
    if pending and pending.get("candidates"):
        st.warning("A visual reference needs a paper choice.")
        candidate_names = list(dict.fromkeys(
            candidate["pdf_name"] for candidate in pending["candidates"]
        ))
        pending_choice = st.selectbox(
            "Choose the intended paper",
            options=candidate_names,
            key="visual_candidate_choice",
        )
        if st.button("Use this paper as preference", use_container_width=True):
            st.session_state.pending_preferred_pdf_name = pending_choice
            st.session_state.pending_visual_resolution = None
            st.rerun()

    if st.button(
        "Clear conversation",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.session_state.last_user_question = ""
        clear_visual_target_state(st.session_state)
        st.rerun()


library_signature = tuple(
    (path.name, path.stat().st_size, path.stat().st_mtime_ns)
    for path in pdf_files
)
if visual_index_needs_rebuild():
    with st.spinner("Updating the local figure/table index..."):
        visual_index = load_visual_library(library_signature)
else:
    visual_index = load_visual_library(library_signature)


# ---------------------------------------------------------
# Display previous messages
# ---------------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        evidence = message.get("evidence")

        if message.get("sources") or evidence:
            with st.expander("Sources"):
                if evidence:
                    st.write(
                        f"- Visual analysis: {evidence['pdf']}, "
                        f"page {evidence['page']}"
                    )

                for source in message.get("sources", []):
                    st.write(
                        f"- Retrieved text: {source['source']}, "
                        f"page {source['page']} "
                        f"(score: {source['score']:.3f})"
                    )


# ---------------------------------------------------------
# Chat input
# ---------------------------------------------------------

question = st.chat_input(
    "Ask a question about your papers"
)

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(
            "Understanding question, searching papers "
            "and generating answer..."
        ):
            recent_messages = st.session_state.messages[
                -MAX_HISTORY_MESSAGES:
            ]

            conversation_history = []

            for message in recent_messages[:-1]:
                conversation_history.append(
                    {
                        "role": message["role"],
                        "content": message["content"],
                    }
                )

            retrieval_question = rewrite_question(
                question=question,
                conversation_history=conversation_history,
            )

            context, sources = retrieve_context(
                question=retrieval_question,
                previous_question="",
                collection=collection,
                embedder=embedder,
                reranker=reranker,
            )

            visual_answer = ""
            vision_error = ""
            visual_context = ""
            vision_debug = {"_save_crops": vision_debug_enabled}
            resolution_fallback_debug = {}
            previous_visual_evidence = next(
                (
                    message.get("evidence")
                    for message in reversed(st.session_state.messages[:-1])
                    if isinstance(message.get("evidence"), dict)
                    and message["evidence"].get("vision_result")
                ),
                None,
            )
            previous_sources = [
                source.get("pdf", source.get("source", ""))
                for message in st.session_state.messages[:-1]
                for source in message.get("sources", [])
            ]
            if manual_visual_override and manual_pdf is not None:
                resolution = manual_visual_resolution(
                    manual_pdf, int(manual_page_number), question
                )
            elif automatic_visual_detection:
                resolution = resolve_visual_target(
                    question=question,
                    index=visual_index,
                    selected_pdf=preferred_pdf,
                    conversation_messages=st.session_state.messages[:-1],
                    current_source_names=previous_sources,
                    embedder=embedder,
                )
                if has_visual_reference(question) and (
                    resolution.status == "not_found"
                    or (
                        resolution.status == "ambiguous"
                        and "confidence" in resolution.reason.casefold()
                    )
                ):
                    with st.spinner("Checking unresolved pages with the local vision model..."):
                        resolution = resolve_with_visual_fallback(
                            question=question,
                            base_resolution=resolution,
                            index=visual_index,
                            selected_pdf=preferred_pdf,
                            debug_info=resolution_fallback_debug,
                        )
            else:
                resolution = resolve_visual_target(question, {"files": {}})

            use_vision = manual_visual_override or should_activate_automatic_vision(
                question, resolution
            )
            selected_pdf = Path(resolution.pdf_path) if resolution.pdf_path else None
            vision_page_number = resolution.page_number or 1
            if use_vision and selected_pdf is not None:
                if not manual_visual_override:
                    st.info(
                        "Automatically detected:\n\n"
                        f"{resolution.target_type.title()} {resolution.target_number}"
                        + (f", panel {resolution.panel}" if resolution.panel else "")
                        + f"\n\n{resolution.pdf_name}\n\nPDF page {resolution.page_number}"
                    )
                try:
                    visual_answer = analyse_resolved_visual(
                        question=question,
                        resolution=resolution,
                        debug_info=vision_debug,
                        text_evidence=context,
                    )
                except Exception as error:
                    vision_error = str(error)

                if visual_answer:
                    visual_context = (
                        "[VISUAL ANALYSIS - VALID FIGURE/TABLE EVIDENCE]\n"
                        f"Source: {selected_pdf.name}\n"
                        f"PDF page: {int(vision_page_number)}\n"
                        f"Result: {visual_answer}"
                    )

                    # Put visual evidence first so incomplete text extraction
                    # does not override a clear reading of a figure or table.
                    if context:
                        context = (
                            f"{visual_context}\n\n"
                            f"[EXTRACTED PDF TEXT]\n{context}"
                        )
                    else:
                        context = visual_context

            # When the user explicitly enables vision and the vision model
            # returns an answer, use that answer directly. This prevents the
            # separate text model from contradicting a correct reading of the
            # selected figure/table/page.
            evidence = None
            visual_reference_requested = has_visual_reference(question)

            if visual_answer:
                cross_visual_comparison = bool(
                    previous_visual_evidence
                    and re.search(r"\b(?:compare|versus|vs\.?|difference)\b", question, re.I)
                    and re.search(r"\b(?:it|that|previous|them)\b", question, re.I)
                )
                if cross_visual_comparison:
                    comparison_context = (
                        "[PREVIOUS VALIDATED VISUAL ANALYSIS]\n"
                        f"Source: {previous_visual_evidence['pdf']}\n"
                        f"PDF page: {previous_visual_evidence['page']}\n"
                        f"Result: {previous_visual_evidence['vision_result']}\n\n"
                        f"{context}"
                    )
                    answer = generate_answer(
                        question=question,
                        context=comparison_context,
                        conversation_history=conversation_history,
                    )
                else:
                    answer = (
                        f"{visual_answer}\n\n"
                        f"Source: **{selected_pdf.name}**, "
                        f"PDF page **{int(vision_page_number)}**."
                    )
                evidence = {
                    "summary": (
                        "Local visual analysis of the selected page."
                    ),
                    "pdf": selected_pdf.name,
                    "page": int(vision_page_number),
                    "vision_result": visual_answer,
                    "visual_target": resolution.to_dict(),
                }
            elif resolution.status == "ambiguous" and visual_reference_requested:
                answer = format_resolution_problem(resolution)
                st.session_state.pending_visual_resolution = resolution.to_dict()
            elif resolution.status == "not_found" and visual_reference_requested:
                answer = format_resolution_problem(resolution)
            elif use_vision and selected_pdf is not None and vision_error:
                answer = COULD_NOT_VERIFY_MESSAGE
            elif not context:
                answer = (
                    "No relevant information was found "
                    "in the indexed papers."
                )
            else:
                answer = generate_answer(
                    question=question,
                    context=context,
                    conversation_history=conversation_history,
                )

            st.markdown(answer)

            if sources or evidence:
                with st.expander("Sources"):
                    if evidence:
                        st.write(
                            f"- Visual analysis: {evidence['pdf']}, "
                            f"page {evidence['page']}"
                        )

                    for source in sources:
                        st.write(
                            f"- Retrieved text: {source['source']}, "
                            f"page {source['page']} "
                            f"(score: {source['score']:.3f})"
                        )

            with st.expander("Debug retrieval details"):
                st.markdown("### Original question")
                st.code(question)

                st.markdown("### Rewritten retrieval query")
                st.code(retrieval_question)

                if visual_reference_requested or manual_visual_override:
                    st.markdown("### Visual target resolution")
                    st.json(resolution.to_dict())
                    if resolution_fallback_debug:
                        st.markdown("### Local vision resolution fallback")
                        st.json(resolution_fallback_debug)

                if use_vision and selected_pdf is not None:
                    st.markdown("### Vision analysis")

                    st.write(
                        f"PDF: {selected_pdf.name}"
                    )

                    st.write(
                        f"Page: {int(vision_page_number)}"
                    )

                    if vision_error:
                        st.error(f"Vision analysis failed: {vision_error}")
                    else:
                        st.text_area(
                            "Vision result",
                            value=visual_answer,
                            height=180,
                            key=(
                                f"vision_result_"
                                f"{len(st.session_state.messages)}"
                            ),
                        )

                    if vision_debug and any(
                        key != "_save_crops" for key in vision_debug
                    ):
                        st.markdown("### Vision crops")
                        st.code(vision_debug.get("crop_folder", ""))
                        for crop in vision_debug.get("crops", []):
                            st.write(
                                f"{crop['name']}: {crop['coordinates']}"
                            )
                            st.code(crop["path"])
                        if vision_debug.get("detections"):
                            st.markdown("### Spatial detections")
                            st.json(vision_debug["detections"])
                        if vision_debug.get("assignments"):
                            st.markdown("### Assignment provenance")
                            st.json(vision_debug["assignments"])
                        if vision_debug.get("verification"):
                            st.markdown("### Disputed-label verification")
                            st.json(vision_debug["verification"])
                        if vision_debug.get("normalized_json"):
                            st.markdown("### Normalized vision JSON")
                            st.json(vision_debug["normalized_json"])
                        if vision_debug.get("validated_json"):
                            st.markdown("### Typed vision JSON")
                            st.json(vision_debug["validated_json"])
                        if vision_debug.get("validation_error"):
                            st.markdown("### Vision validation error")
                            st.code(vision_debug["validation_error"])
                        if vision_debug.get("raw_vision_response"):
                            st.markdown("### Raw vision response")
                            st.text_area(
                                "Raw structured output",
                                value=vision_debug["raw_vision_response"],
                                height=220,
                                key=f"raw_vision_{len(st.session_state.messages)}",
                            )
                        if vision_debug.get("final_answer_path"):
                            st.markdown("### Final answer code path")
                            st.code(vision_debug["final_answer_path"])

                if sources:
                    retrieval_debug = sources[0].get("retrieval_debug")
                    if retrieval_debug:
                        st.markdown("### Summary retrieval counts")
                        st.json(retrieval_debug)

                    st.markdown("### Retrieved chunks")

                    for index, source in enumerate(
                        sources,
                        start=1,
                    ):
                        st.markdown(
                            f"**Result {index}: "
                            f"{source['source']}, "
                            f"page {source['page']} "
                            f"(score: {source['score']:.3f})**"
                        )

                        st.text_area(
                            label=f"Chunk {index}",
                            value=source["document"],
                            height=180,
                            key=(
                                f"debug_chunk_"
                                f"{len(st.session_state.messages)}_"
                                f"{index}"
                            ),
                        )
                else:
                    st.write("No text chunks were retrieved.")

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "retrieval_question": retrieval_question,
            "vision_answer": visual_answer,
            "vision_error": vision_error,
            "evidence": evidence,
            "visual_target": (
                resolution.to_dict() if resolution.status == "resolved" else None
            ),
        }
    )

    st.session_state.last_user_question = question 
