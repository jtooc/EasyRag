import lazyllm

llm2 = lazyllm.OnlineChatModule(source="deepseek", model="deepseek-v4-flash").prompt(
    "根据给出的文段回答问题，文段：{content}"
)

passage = (
    "孙悟空，是在小说《西游记》当中唐僧的四个徒弟之一，排行第一，别名孙行者、孙猴子。"
    "自封美猴王、齐天大圣。因曾在天庭掌管御马监而又被称为弼马温，在取经完成后被如来佛祖授封为斗战胜佛"
)

# 下面打印promt_content仅做展示，实际不需要：
prompt_content = llm2._prompt.generate_prompt(
    {"input": "孙悟空有哪些名字？", "content": passage}, return_dict=True
)
print(prompt_content)

# 模型推理：
print(llm2({"input": "孙悟空有哪些名字？", "content": passage}))
