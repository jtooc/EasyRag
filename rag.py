import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
os.environ.setdefault("LAZYLLM_HOME", str(PROJECT_ROOT / ".temp" / "lazyllm"))
import lazyllm


DEFAULT_KB_PATH = PROJECT_ROOT / "data_kb"
PROMPT = (
    "You will act as an AI question-answering assistant and complete a dialogue task. "
    "In this task, you need to provide your answers based on the given context and questions."
)


def create_retriever(dataset_path=DEFAULT_KB_PATH, query="", topk=3):
    if not query:
        return []

    documents = lazyllm.Document(dataset_path=str(dataset_path))
    retriever = lazyllm.Retriever(
        doc=documents, group_name="CoarseChunk", similarity="bm25_chinese", topk=3
    )
    results = retriever(query=query)
    return results if isinstance(results, list) else list(results)


def main():
    llm = lazyllm.OnlineChatModule(
        source="deepseek",
        model="deepseek-v4-flash",
        stream=False,
    )
    llm.prompt(lazyllm.ChatPrompter(instruction=PROMPT, extra_keys=["context_str"]))

    query = "为我们介绍下短叶水蜈蚣"
    doc_node_list = create_retriever(query=query)
    context_str = "".join([node.get_content() for node in doc_node_list])
    res = llm({"query": query, "context_str": context_str})

    print(f"With RAG Answer: {res}")


if __name__ == "__main__":
    main()
