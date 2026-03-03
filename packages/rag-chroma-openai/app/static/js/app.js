document.addEventListener("DOMContentLoaded", () => {
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const uploadProgress = document.getElementById("upload-progress");
    const progressBar = document.getElementById("progress-bar");
    const progressText = document.getElementById("progress-text");
    const searchForm = document.getElementById("search-form");
    const queryInput = document.getElementById("query-input");
    const searchBtn = document.getElementById("search-btn");
    const resultsContainer = document.getElementById("results-container");
    const answerBox = document.getElementById("answer-box");
    const loadingIndicator = document.getElementById("loading-indicator");
    const sourcesContainer = document.getElementById("sources-container");
    const sourcesList = document.getElementById("sources-list");

    // --- Document List ---
    async function loadDocuments() {
        const res = await fetch("/api/documents");
        const data = await res.json();
        const list = document.getElementById("document-list");
        const noDocsMsg = document.getElementById("no-docs-msg");

        if (data.documents.length === 0) {
            noDocsMsg.classList.remove("hidden");
            list.querySelectorAll(".doc-item").forEach(el => el.remove());
            return;
        }
        noDocsMsg.classList.add("hidden");

        // Clear existing items
        list.querySelectorAll(".doc-item").forEach(el => el.remove());

        data.documents.forEach(doc => {
            const el = document.createElement("div");
            el.className = "doc-item flex items-center justify-between p-3 bg-gray-50 rounded-lg";
            el.innerHTML = `
                <div class="min-w-0 flex-1">
                    <p class="text-sm font-medium text-gray-700 truncate">${doc.filename}</p>
                    <p class="text-xs text-gray-400">${doc.page_count}페이지 · ${doc.chunk_count}청크</p>
                </div>
                <button onclick="deleteDocument('${doc.document_id}')"
                        class="ml-2 text-red-400 hover:text-red-600 text-sm flex-shrink-0" title="삭제">
                    <svg class="h-4 w-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M6 18L18 6M6 6l12 12"/>
                    </svg>
                </button>
            `;
            list.appendChild(el);
        });
    }

    window.deleteDocument = async function(docId) {
        if (!confirm("이 문서를 삭제하시겠습니까?")) return;
        await fetch(`/api/documents/${docId}`, { method: "DELETE" });
        loadDocuments();
    };

    // --- File Upload ---
    dropZone.addEventListener("click", () => fileInput.click());
    dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
    dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
    dropZone.addEventListener("drop", e => {
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
        progressBar.style.width = "30%";
        progressText.textContent = "업로드 중...";

        const formData = new FormData();
        formData.append("file", file);

        try {
            progressBar.style.width = "60%";
            progressText.textContent = "PDF 처리 중...";

            const res = await fetch("/api/documents/upload", { method: "POST", body: formData });
            const data = await res.json();

            if (!res.ok) {
                alert(data.detail || "업로드 실패");
                return;
            }

            progressBar.style.width = "100%";
            progressText.textContent = "완료!";

            setTimeout(() => {
                uploadProgress.classList.add("hidden");
                progressBar.style.width = "0%";
            }, 1500);

            loadDocuments();
        } catch (err) {
            alert("업로드 중 오류가 발생했습니다.");
            uploadProgress.classList.add("hidden");
        }

        fileInput.value = "";
    }

    // --- Search with SSE Streaming ---
    searchForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const query = queryInput.value.trim();
        if (!query) return;

        searchBtn.disabled = true;
        resultsContainer.classList.remove("hidden");
        answerBox.textContent = "";
        sourcesContainer.classList.add("hidden");
        sourcesList.innerHTML = "";
        loadingIndicator.classList.remove("hidden");

        try {
            const res = await fetch("/api/search/stream", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ query }),
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
                    const jsonStr = line.slice(6);
                    if (!jsonStr) continue;

                    try {
                        const event = JSON.parse(jsonStr);

                        if (event.type === "sources") {
                            loadingIndicator.classList.add("hidden");
                            renderSources(event.content);
                        } else if (event.type === "chunk") {
                            loadingIndicator.classList.add("hidden");
                            answerBox.textContent += event.content;
                        } else if (event.type === "error") {
                            loadingIndicator.classList.add("hidden");
                            answerBox.textContent = event.content;
                        } else if (event.type === "done") {
                            // stream complete
                        }
                    } catch {}
                }
            }
        } catch (err) {
            answerBox.textContent = "검색 중 오류가 발생했습니다.";
            loadingIndicator.classList.add("hidden");
        }

        searchBtn.disabled = false;
    });

    function renderSources(sources) {
        if (!sources || sources.length === 0) return;
        sourcesContainer.classList.remove("hidden");
        sourcesList.innerHTML = "";

        sources.forEach(src => {
            const typeLabel = { text: "텍스트", table: "테이블", image_ocr: "이미지 OCR" };
            const el = document.createElement("div");
            el.className = "p-3 bg-gray-50 rounded-lg";
            el.innerHTML = `
                <div class="flex items-center gap-2 mb-1">
                    <span class="source-badge ${src.content_type}">${typeLabel[src.content_type] || src.content_type}</span>
                    <span class="text-xs text-gray-500">${src.document} · ${src.page}페이지</span>
                </div>
                <p class="text-xs text-gray-500 line-clamp-2">${src.snippet}</p>
            `;
            sourcesList.appendChild(el);
        });
    }

    // Initial load
    loadDocuments();
});
