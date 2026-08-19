# Graph Report - src  (2026-08-19)

## Corpus Check
- Corpus is ~11,875 words - fits in a single context window. You may not need a graph.

## Summary
- 390 nodes · 753 edges · 16 communities (15 shown, 1 thin omitted)
- Extraction: 86% EXTRACTED · 14% INFERRED · 0% AMBIGUOUS · INFERRED: 103 edges (avg confidence: 0.69)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Web Identity and Auth|Web Identity and Auth]]
- [[_COMMUNITY_Cross-Channel Ask Flow|Cross-Channel Ask Flow]]
- [[_COMMUNITY_Bot API Client|Bot API Client]]
- [[_COMMUNITY_Telegram Media Interface|Telegram Media Interface]]
- [[_COMMUNITY_Document Ingestion|Document Ingestion]]
- [[_COMMUNITY_Identity Trust Semantics|Identity Trust Semantics]]
- [[_COMMUNITY_RAG Security Boundary|RAG Security Boundary]]
- [[_COMMUNITY_Vector Runtime Lifecycle|Vector Runtime Lifecycle]]
- [[_COMMUNITY_Bot Persistence Lifecycle|Bot Persistence Lifecycle]]
- [[_COMMUNITY_Conversation Memory|Conversation Memory]]
- [[_COMMUNITY_RAG Generation Pipeline|RAG Generation Pipeline]]
- [[_COMMUNITY_Server Runtime Configuration|Server Runtime Configuration]]
- [[_COMMUNITY_Streamlit Presentation|Streamlit Presentation]]
- [[_COMMUNITY_Bot Runtime Configuration|Bot Runtime Configuration]]

## God Nodes (most connected - your core abstractions)
1. `UnauthorizedAPIKeyError` - 16 edges
2. `index_directory()` - 12 edges
3. `WebUser` - 12 edges
4. `ParsedQuestion` - 12 edges
5. `AuthError` - 11 edges
6. `normalize_text()` - 10 edges
7. `UsernameAlreadyExistsError` - 10 edges
8. `accept_invite()` - 10 edges
9. `ConversationMemoryStore` - 10 edges
10. `process_question()` - 10 edges

## Surprising Connections (you probably didn't know these)
- `Browser Text-Safe Answer Rendering` --semantically_similar_to--> `Telegram HTML Output Encoding`  [INFERRED] [semantically similar]
  src/server/templates/index.html → src/bot/handlers/all_handlers.py
- `Browser Source Presentation Pipeline` --semantically_similar_to--> `Streamlit Source Presentation Pipeline`  [INFERRED] [semantically similar]
  src/server/templates/index.html → src/client/app/streamlit_app.py
- `AskAPIClient Adapter` --semantically_similar_to--> `Streamlit get_response API Adapter`  [INFERRED] [semantically similar]
  src/bot/api_client.py → src/client/app/streamlit_app.py
- `JSON /ask Request-Response Contract` --conceptually_related_to--> `Browser /web/ask Fetch Flow`  [INFERRED]
  src/bot/api_client.py → src/server/templates/index.html
- `Browser /web/ask Fetch Flow` --conceptually_related_to--> `Unscreened Untrusted Prompt Forwarding`  [INFERRED]
  src/server/templates/index.html → src/bot/handlers/common.py

## Hyperedges (group relationships)
- **Document-to-Vector Indexing Flow** — document_ingestion_untrusted_document_boundary, document_ingestion_parser_pipeline, document_ingestion_chunking_pipeline, document_ingestion_index_directory, vector_store_trust_boundary [INFERRED 0.90]
- **Authenticated RAG Request Flow** — main_api_key_auth, main_web_session_auth, main_question_validation, main_question_orchestration, rag_ask_question, vector_similarity_search, rag_invoke_llm [EXTRACTED 1.00]
- **Prompt Injection Attack Surface** — document_ingestion_untrusted_document_boundary, rag_build_context, rag_build_conversation_history, rag_build_retrieval_query, rag_prompt_injection_boundary, rag_indirect_prompt_injection_risk, rag_memory_poisoning_risk, main_session_collision_risk [INFERRED 0.89]
- **Multi-Channel Ask Client Flow** — api_ask_client, streamlit_get_response, web_ask_flow, ask_http_contract [INFERRED 0.90]
- **Telegram Media Extraction and Confirmation Flow** — handlers_photo_ingress, handlers_pdf_ingress, common_prepare_question, handlers_question_fsm, handlers_confirmation, handlers_send_answer [EXTRACTED 1.00]
- **Direct and Indirect Prompt Injection Attack Surface** — handlers_text_ingress, handlers_photo_ingress, handlers_pdf_ingress, streamlit_get_response, web_ask_flow, prompt_injection_gap, indirect_prompt_injection [INFERRED 0.90]

## Communities (16 total, 1 thin omitted)

### Community 0 - "Web Identity and Auth"
Cohesion: 0.08
Nodes (60): accept_invite(), authenticate_user(), AuthError, BootstrapAlreadyConfiguredError, create_bootstrap_admin(), create_invite(), create_web_session(), ExpiredInviteError (+52 more)

### Community 1 - "Cross-Channel Ask Flow"
Cohesion: 0.05
Nodes (55): AskAPIClient.ask, AskAPIClient Adapter, AskAPIClient._build_headers, AskAPIClient._extract_error_details, Typed Ask API Failure Taxonomy, X-API-Key Service Authentication Boundary, AskResult and AskSource Response Contract, AskAPIClient._ask_with_client (+47 more)

### Community 2 - "Bot API Client"
Cohesion: 0.1
Nodes (28): AskAPIClient, AskAPIError, AskAPIResponseError, AskAPITimeoutError, AskAPIUnauthorizedError, AskAPIUnavailableError, AskResult, AskSource (+20 more)

### Community 3 - "Telegram Media Interface"
Cohesion: 0.1
Nodes (33): back_keyboard(), build_confirmation_preview(), cb_about(), cb_ask_question(), cb_back_to_menu(), cb_confirm_no(), cb_confirm_yes(), cmd_help() (+25 more)

### Community 4 - "Document Ingestion"
Cohesion: 0.12
Nodes (30): build_chunk_records(), build_document_id(), _build_sections(), chunk_text(), ChunkRecord, create_vector_store(), _delete_document_chunks(), _delete_stale_document_chunks() (+22 more)

### Community 5 - "Identity Trust Semantics"
Cohesion: 0.08
Nodes (35): authenticate_user, Bootstrap and Invite Concurrency Risk, create_bootstrap_admin, One-time Invite Workflow, Database-backed Authentication Pattern, Password and Token Hashing, Revocable Web Session Workflow, Authentication Database Lifecycle (+27 more)

### Community 6 - "RAG Security Boundary"
Cohesion: 0.08
Nodes (35): LLM Prompt Template, Environment-driven Runtime Configuration, Container-aware Storage Path Resolution, validate_runtime_config, ChunkRecord, Overlapping Text Chunking Pipeline, create_vector_store, index_directory (+27 more)

### Community 7 - "Vector Runtime Lifecycle"
Cohesion: 0.19
Nodes (17): build_parser(), main(), _log_startup_indexing_summary(), _prepare_rag_runtime(), clear_vector_cache(), EmptyVectorStoreError, _ensure_index_ready(), ensure_vector_store_ready() (+9 more)

### Community 8 - "Bot Persistence Lifecycle"
Cohesion: 0.19
Nodes (10): main(), create_request(), get_or_create_user(), init_db(), Core database layer for the bot., Base, Base declarative model for bot tables., Request (+2 more)

### Community 9 - "Conversation Memory"
Cohesion: 0.21
Nodes (7): dispose_auth_db(), init_auth_db(), ConversationMemoryStore, _normalize_key(), _normalize_message(), DB-backed windowed memory with TTL and session eviction for server deployments., lifespan()

### Community 10 - "RAG Generation Pipeline"
Cohesion: 0.29
Nodes (13): ask_question(), build_context(), build_conversation_history(), build_empty_answer(), build_fallback_answer(), build_retrieval_query(), compute_confidence(), deduplicate_sources() (+5 more)

### Community 11 - "Server Runtime Configuration"
Cohesion: 0.33
Nodes (8): _is_running_in_container(), _resolve_default_web_auth_db_url(), _resolve_documents_dir(), resolve_sqlite_path_from_url(), _resolve_vector_db_dir(), _resolve_web_auth_database_url(), validate_chunk_settings(), validate_runtime_config()

### Community 12 - "Streamlit Presentation"
Cohesion: 0.5
Nodes (7): _format_source_excerpt(), _format_source_meta(), _format_source_title(), _humanize_source_label(), _normalize_source_field(), _render_sources(), _truncate_text()

## Ambiguous Edges - Review These
- `Bootstrap Login Invite Authentication UI` → `State-Changing Web Form CSRF Posture Requires Verification`  [AMBIGUOUS]
  src/server/templates/index.html · relation: conceptually_related_to

## Knowledge Gaps
- **13 isolated node(s):** `render_metrics`, `ParsedDocument`, `ChunkRecord`, `ConversationMemoryMessage`, `Document Indexing CLI` (+8 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **1 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **What is the exact relationship between `Bootstrap Login Invite Authentication UI` and `State-Changing Web Form CSRF Posture Requires Verification`?**
  _Edge tagged AMBIGUOUS (relation: conceptually_related_to) - confidence is low._
- **Why does `MediaProcessingError` connect `Telegram Media Interface` to `Web Identity and Auth`?**
  _High betweenness centrality (0.149) - this node is a cross-community bridge._
- **Why does `UnauthorizedAPIKeyError` connect `Web Identity and Auth` to `Conversation Memory`, `Document Ingestion`, `Vector Runtime Lifecycle`?**
  _High betweenness centrality (0.078) - this node is a cross-community bridge._
- **Are the 10 inferred relationships involving `UnauthorizedAPIKeyError` (e.g. with `BootstrapAlreadyConfiguredError` and `ExpiredInviteError`) actually correct?**
  _`UnauthorizedAPIKeyError` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 8 inferred relationships involving `WebUser` (e.g. with `AuthError` and `BootstrapAlreadyConfiguredError`) actually correct?**
  _`WebUser` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 10 inferred relationships involving `ParsedQuestion` (e.g. with `BootstrapAlreadyConfiguredError` and `ExpiredInviteError`) actually correct?**
  _`ParsedQuestion` has 10 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `AuthError` (e.g. with `WebInvite` and `WebSession`) actually correct?**
  _`AuthError` has 3 INFERRED edges - model-reasoned connections that need verification._
