from rag import create_retriever

# 测试文档路径
TEST_PATH = "./data_kb"


def test_retriever_contains_keyword():
    test_query = "为我介绍一下2008年北京奥运会"
    expected_keyword = "奥运比赛"

    results = create_retriever(TEST_PATH, test_query)
    retrieved_content = "\n".join([node.get_content() for node in results])

    assert expected_keyword in retrieved_content, f"检索结果中未找到关键词 '{expected_keyword}'"


def test_retriever_empty_query():
    results = create_retriever(TEST_PATH, "")
    assert isinstance(results, list), "结果应该是列表类型"
