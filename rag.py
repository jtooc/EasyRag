import lazyllm
from pathlib import Path

# 文档加载
kb_path = Path(__file__).resolve().parent / "data_kb"
documents = lazyllm.Document(dataset_path=str(kb_path))

# 检索组件定义
retriever = lazyllm.Retriever(
    doc=documents, group_name="CoarseChunk", similarity="bm25_chinese", topk=3
)

# 生成组件定义
llm = lazyllm.OnlineChatModule(source="deepseek", model="deepseek-v4-flash")

# prompt 设计
prompt = "You will act as an AI question-answering assistant and complete a dialogue task. In this task, you need to provide your answers based on the given context and questions."
llm.prompt(lazyllm.ChatPrompter(instruction=prompt, extra_keys=["context_str"]))

# 推理
query = "为我介绍下短叶水蜈蚣"
# 将Retriever组件召回的节点全部存储到列表doc_node_list中
doc_node_list = retriever(query=query)
# 将query和召回节点中的内容组成dict，作为大模型的输入
res = llm(
    {
        "query": query,
        "context_str": "".join([node.get_content() for node in doc_node_list]),
    }
)

print(f"With RAG Answer: {res}")


# llm_without_rag = lazyllm.OnlineChatModule(source="deepseek", model="deepseek-v4-flash")
# query = "为我介绍下短叶水蜈蚣"
# res = llm_without_rag(query)
# print(f'Without RAG Answer: {res}')
