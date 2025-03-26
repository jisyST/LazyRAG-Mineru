# -*- coding: utf-8 -*-
# flake8: noqa: F821

import lazyllm
from lazyllm import LOG
from lazyllm import pipeline, parallel, bind, OnlineEmbeddingModule, SentenceSplitter, Document, Retriever, Reranker
from src.reader.magic_pdf_reader import MagicPDFReader
from src.parser.magic_pdf_transform import MagicPDFTransform


import jieba
jieba.setLogLevel(jieba.logging.WARNING)

# Before running, set the environment variable:
#
# 1. `export LAZYLLM_GLM_API_KEY=xxxx`: the API key of Zhipu AI, default model "glm-4", `source="glm"`.
#     You can apply for the API key at https://open.bigmodel.cn/
#     Also supports other API keys:
#       - LAZYLLM_OPENAI_API_KEY: the API key of OpenAI, default model "gpt-3.5-turbo", `source="openai"`.
#           You can apply for the API key at https://openai.com/index/openai-api/
#       - LAZYLLM_KIMI_API_KEY: the API key of Moonshot AI, default model "moonshot-v1-8k", `source="kimi"`.
#           You can apply for the API key at https://platform.moonshot.cn/console
#       - LAZYLLM_QWEN_API_KEY: the API key of Alibaba Cloud, default model "qwen-plus", `source="qwen"`.
#           You can apply for the API key at https://home.console.aliyun.com/
#       - LAZYLLM_SENSENOVA_API_KEY: the API key of SenseTime, default model "SenseChat-5", `source="sensenova"`.
#                                  You also have to set LAZYLLM_SENSENOVA_SECRET_KEY` togather.
#           You can apply for the API key at https://platform.sensenova.cn/home
#     * `source` needs to be specified for multiple API keys, but it does not need to be set for a single API key.
#
# 2. `export LAZYLLM_DATA_PATH=path/to/docs/folder/`: The parent folder of the document folder `rag_master`.
#                                                    Alternatively, you can set the `dataset_path` of the Document
#                                                    to `path/to/docs/folder/rag_master` to replace
#                                                    the setting of this environment variable.

prompt = 'You will play the role of an AI Q&A assistant and complete a dialogue task. In this task, you need to provide your answer based on the given context and question.'

documents = Document(dataset_path="/home/mnt/jisiyuan/projects/LazyRAG-Mineru/data/金融知识库", embed=OnlineEmbeddingModule(), manager=False)

documents.add_reader("**/*.pdf", MagicPDFReader)

documents.create_node_group(name="magic-pdf", transform=MagicPDFTransform)
documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=2048, chunk_overlap=100, parent="magic-pdf")


with pipeline() as ppl:
    with parallel().sum as ppl.prl:
        prl.retriever1 = Retriever(documents, group_name="sentences", similarity="cosine", topk=3)
        prl.retriever2 = Retriever(documents, group_name="sentences", similarity="bm25_chinese", topk=3)
    ppl.reranker = Reranker("ModuleReranker", model=OnlineEmbeddingModule(type="rerank"), topk=1, output_format='content', join=True) | bind(query=ppl.input)
    ppl.formatter = (lambda nodes, query: dict(context_str=nodes, query=query)) | bind(query=ppl.input)
    ppl.llm = lazyllm.OnlineChatModule(stream=False).prompt(lazyllm.ChatPrompter(prompt, extro_keys=["context_str"]))


if __name__ == "__main__":
    while True:
        query = input("input your query: ")
        if query == 'exit':
            break
        print(f"recieve your query: {query} \n")

        answer = ppl(query)
        print(f"{'='*50} answer {'='*50} \n")
        print(answer)
        print('='*110 + "\n")
