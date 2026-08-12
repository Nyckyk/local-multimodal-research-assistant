from pathlib import Path

import streamlit as st

from rag.database import get_collection
from rag.embeddings import load_embedder, load_reranker
from rag.ingestion import index_pdf
from services.research_assistant import ResearchAssistant
from services.visual_index import load_or_build_visual_index, visual_index_needs_rebuild
from services.visual_locator import (
    clear_visual_conversation_state,
)
from services.visual_reference_parser import has_visual_reference
from settings import PAPERS_FOLDER


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
        clear_visual_conversation_state(st.session_state)
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
    if not isinstance(message, dict):
        continue
    role = message.get("role")
    content = message.get("content")
    if not isinstance(role, str) or not isinstance(content, str):
        continue
    with st.chat_message(role):
        st.markdown(content)

        evidence = message.get("evidence")
        if not isinstance(evidence, dict):
            evidence = None
        message_sources = message.get("sources")
        if not isinstance(message_sources, list):
            message_sources = []

        if message_sources or evidence:
            with st.expander("Sources"):
                if evidence:
                    evidence_kind = evidence.get("analysis_kind", "Visual analysis")
                    st.write(
                        f"- {evidence_kind}: {evidence.get('pdf', 'unknown source')}, "
                        f"page {evidence.get('page', 'unknown')}"
                    )

                for source in message_sources:
                    if not isinstance(source, dict):
                        continue
                    source_score = source.get("score")
                    if (
                        not isinstance(source_score, (int, float))
                        or isinstance(source_score, bool)
                    ):
                        source_score = 0.0
                    st.write(
                        f"- Retrieved text: {source.get('source', 'unknown source')}, "
                        f"page {source.get('page', 'unknown')} "
                        f"(score: {source_score:.3f})"
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
            assistant = ResearchAssistant(
                collection=collection,
                embedder=embedder,
                reranker=reranker,
                visual_index=visual_index,
            )
            response = assistant.ask(
                question,
                conversation_messages=st.session_state.messages[:-1],
                preferred_pdf=preferred_pdf,
                automatic_visual_detection=automatic_visual_detection,
                manual_pdf=manual_pdf if manual_visual_override else None,
                manual_page_number=int(manual_page_number),
                save_vision_crops=vision_debug_enabled,
            )
            answer = response.answer
            sources = response.sources
            retrieval_question = response.retrieval_question
            visual_answer = response.visual_answer
            vision_error = response.vision_error
            equation_error = response.equation_error
            evidence = response.evidence
            resolution = response.resolution
            multi_resolutions = response.multi_resolutions
            use_vision = response.used_vision
            selected_pdf = Path(resolution.pdf_path) if resolution.pdf_path else None
            vision_page_number = resolution.page_number or 1
            vision_debug = response.debug["vision"]
            equation_debug = response.debug["equation"]
            multi_debug = response.debug["multi_target"]
            text_generation_debug = response.debug["text_generation"]
            resolution_fallback_debug = response.debug["resolution_fallback"]
            visual_reference_requested = has_visual_reference(question)

            if use_vision and selected_pdf is not None and not manual_visual_override:
                st.info(
                    "Automatically detected:\n\n"
                    f"{resolution.target_type.title()} {resolution.target_number}"
                    + (f", panel {resolution.panel}" if resolution.panel else "")
                    + f"\n\n{resolution.pdf_name}\n\nPDF page {resolution.page_number}"
                )
            if resolution.status == "ambiguous" and visual_reference_requested:
                st.session_state.pending_visual_resolution = resolution.to_dict()

            st.markdown(answer)

            if sources or evidence:
                with st.expander("Sources"):
                    if evidence:
                        evidence_kind = evidence.get("analysis_kind", "Visual analysis")
                        st.write(
                            f"- {evidence_kind}: {evidence['pdf']}, "
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
                    st.json(
                        [item.to_dict() for item in multi_resolutions]
                        if len(multi_resolutions) > 1 else resolution.to_dict()
                    )
                    if len(multi_resolutions) > 1:
                        st.markdown("### Multi-target analysis")
                        st.json(multi_debug)
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
                        final_structured = vision_debug.get("final_structured_output")
                        if final_structured is not None:
                            st.markdown("### Final typed structured output")
                            st.json(final_structured)
                        elif vision_debug.get("validated_json"):
                            st.markdown("### Typed vision JSON")
                            st.json(vision_debug["validated_json"])
                        if vision_debug.get("initial_validation_errors"):
                            st.markdown("### Initial raw-output validation errors")
                            st.code("\n".join(
                                vision_debug["initial_validation_errors"]
                            ))
                            if vision_debug.get("initial_response"):
                                st.text_area(
                                    "Initial raw structured output",
                                    value=vision_debug["initial_response"],
                                    height=220,
                                    key=(
                                        f"initial_raw_"
                                        f"{len(st.session_state.messages)}"
                                    ),
                                )
                            if vision_debug.get("repaired_validation_result") == "passed":
                                st.markdown("### Repaired-output validation: passed")
                        if vision_debug.get("validation_error"):
                            st.markdown("### Vision validation error")
                            st.code(vision_debug["validation_error"])
                        if vision_debug.get("raw_vision_response"):
                            st.markdown("### Raw vision response")
                            st.text_area(
                                "Raw model structured output",
                                value=vision_debug["raw_vision_response"],
                                height=220,
                                key=f"raw_vision_{len(st.session_state.messages)}",
                            )
                        final_answer_code_path = (
                            vision_debug.get("final_answer_code_path")
                            or vision_debug.get("final_answer_path")
                        )
                        if final_answer_code_path:
                            st.markdown("### Final answer code path")
                            st.code(final_answer_code_path)

                if equation_debug:
                    st.markdown("### Equation analysis")
                    if equation_error:
                        st.error(equation_error)
                    st.json(equation_debug)

                if text_generation_debug:
                    st.markdown("### Text generation")
                    st.json(text_generation_debug)

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
