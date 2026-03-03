document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const uploadProgress = document.getElementById("upload-progress");
    const progressBar = document.getElementById("progress-bar");
    const progressText = document.getElementById("progress-text");
    const documentList = document.getElementById("document-list");
    const noDocsMsg = document.getElementById("no-docs-msg");
    const searchForm = document.getElementById("search-form");
    const queryInput = document.getElementById("query-input");
    const searchBtn = document.getElementById("search-btn");
    const resultsContainer = document.getElementById("results-container");
    const answerBox = document.getElementById("answer-box");
    const loadingIndicator = document.getElementById("loading-indicator");
    const sourcesContainer = document.getElementById("sources-container");
    const sourcesList = document.getElementById("sources-list");
    const graphRelationsContainer = document.getElementById("graph-relations-container");
    const graphRelationsList = document.getElementById("graph-relations-list");
    const refreshStatsBtn = document.getElementById("refresh-stats-btn");

    // --- Document Management ---

    async function loadDocuments() {
        const res = await fetch("/api/documents");
        const data = await res.json();
        documentList.innerHTML = "";

        if (data.documents.length === 0) {
            noDocsMsg.style.display = "";
            documentList.appendChild(noDocsMsg);
            return;
        }

        noDocsMsg.style.display = "none";
        data.documents.forEach((doc) => {
            const el = document.createElement("div");
            el.className = "doc-item flex justify-between items-center p-2 rounded-lg border text-sm";
            el.innerHTML = `
                <div class="flex-1 min-w-0">
                    <p class="font-medium text-gray-700 truncate">${doc.filename}</p>
                    <p class="text-xs text-gray-400">${doc.page_count}p / ${doc.chunk_count} chunks</p>
                </div>
                <button onclick="deleteDocument('${doc.document_id}')"
                        class="ml-2 text-red-400 hover:text-red-600 text-xs flex-shrink-0">삭제</button>
            `;
            documentList.appendChild(el);
        });
    }

    window.deleteDocument = async function (docId) {
        if (!confirm("이 문서를 삭제하시겠습니까?")) return;
        await fetch(`/api/documents/${docId}`, { method: "DELETE" });
        loadDocuments();
        loadGraphStats();
    };

    // --- File Upload ---

    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("drag-over");
    });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("drag-over");
        if (e.dataTransfer.files.length) uploadFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length) uploadFile(fileInput.files[0]);
    });

    async function uploadFile(file) {
        if (!file.name.toLowerCase().endsWith(".pdf")) {
            alert("PDF 파일만 업로드 가능합니다.");
            return;
        }

        uploadProgress.classList.remove("hidden");
        progressBar.style.width = "10%";
        progressText.textContent = "업로드 중...";

        const formData = new FormData();
        formData.append("file", file);

        progressBar.style.width = "30%";
        progressText.textContent = "PDF 분석 + 그래프 구축 중...";

        try {
            const res = await fetch("/api/documents/upload", {
                method: "POST",
                body: formData,
            });

            progressBar.style.width = "90%";

            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || "업로드 실패");
            }

            const data = await res.json();
            progressBar.style.width = "100%";
            progressText.textContent = `완료! (${data.chunk_count} chunks, ${data.entity_count} entities)`;

            loadDocuments();
            loadGraphStats();
        } catch (err) {
            progressText.textContent = `오류: ${err.message}`;
            progressBar.style.width = "0%";
        }

        setTimeout(() => uploadProgress.classList.add("hidden"), 3000);
        fileInput.value = "";
    }

    // --- Search with SSE ---

    searchForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;

        searchBtn.disabled = true;
        resultsContainer.classList.remove("hidden");
        answerBox.textContent = "";
        sourcesList.innerHTML = "";
        sourcesContainer.classList.add("hidden");
        graphRelationsList.innerHTML = "";
        graphRelationsContainer.classList.add("hidden");
        loadingIndicator.classList.remove("hidden");

        try {
            const res = await fetch("/api/search/stream", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query, k: 5 }),
            });

            const reader = res.body.getReader();
            const decoder = new TextDecoder();
            let buffer = "";

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split("\n");
                buffer = lines.pop();

                for (const line of lines) {
                    if (!line.startsWith("data: ")) continue;
                    const payload = JSON.parse(line.slice(6));

                    if (payload.type === "sources") {
                        renderSources(payload.content);
                    } else if (payload.type === "graph") {
                        renderGraphRelations(payload.content);
                    } else if (payload.type === "chunk") {
                        loadingIndicator.classList.add("hidden");
                        answerBox.textContent += payload.content;
                    } else if (payload.type === "error") {
                        loadingIndicator.classList.add("hidden");
                        answerBox.textContent = payload.content;
                    } else if (payload.type === "done") {
                        loadingIndicator.classList.add("hidden");
                    }
                }
            }
        } catch (err) {
            loadingIndicator.classList.add("hidden");
            answerBox.textContent = `오류가 발생했습니다: ${err.message}`;
        }

        searchBtn.disabled = false;
    });

    function renderSources(sources) {
        sourcesContainer.classList.remove("hidden");
        sourcesList.innerHTML = "";
        sources.forEach((src) => {
            const el = document.createElement("div");
            el.className = "flex items-start gap-2 p-2 bg-gray-50 rounded-lg text-sm";

            const typeBadge = `<span class="source-badge ${src.content_type}">${src.content_type}</span>`;
            const methodBadge = `<span class="method-badge ${src.retrieval_method}">${src.retrieval_method}</span>`;

            el.innerHTML = `
                <div class="flex-1">
                    <div class="flex items-center gap-1 mb-1">
                        <span class="font-medium text-gray-600">${src.document}</span>
                        <span class="text-gray-400">p.${src.page}</span>
                        ${typeBadge}${methodBadge}
                    </div>
                    <p class="text-xs text-gray-500 line-clamp-2">${src.snippet}</p>
                </div>
            `;
            sourcesList.appendChild(el);
        });
    }

    function renderGraphRelations(relations) {
        if (!relations || relations.length === 0) return;
        graphRelationsContainer.classList.remove("hidden");
        graphRelationsList.innerHTML = "";
        relations.forEach((rel) => {
            const el = document.createElement("div");
            el.className = "graph-relation";
            el.innerHTML = `
                <span class="entity">${rel.source}</span>
                <span class="rel-type">${rel.type}</span>
                <span>&rarr;</span>
                <span class="entity">${rel.target}</span>
            `;
            graphRelationsList.appendChild(el);
        });
    }

    // --- Graph Stats ---

    async function loadGraphStats() {
        try {
            const res = await fetch("/api/graph/stats");
            const data = await res.json();
            document.getElementById("stat-documents").textContent = data.document_count;
            document.getElementById("stat-chunks").textContent = data.chunk_count;
            document.getElementById("stat-entities").textContent = data.entity_count;
            document.getElementById("stat-relationships").textContent = data.relationship_count;
        } catch {
            // ignore
        }
    }

    refreshStatsBtn.addEventListener("click", loadGraphStats);

    // --- Init ---
    loadDocuments();
    loadGraphStats();
});
