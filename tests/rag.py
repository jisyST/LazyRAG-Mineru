# -*- coding: utf-8 -*-
# flake8: noqa: F821

import lazyllm
from lazyllm import LOG
from lazyllm import pipeline, parallel, bind, Document, Retriever, Reranker, SentenceSplitter

from lazyrag.components.llm.lazy_llm_factory import LazyLlmFactory
from lazyrag.modules.rag.query_pipeline.components.reranker import RerankComponent
from lazyrag.components.embedding import LazyEmbeddingFactory
from src.rag.utils import (
    LLMChatFormatter,
    RAGContextFormatter,
    RerankComponent
)
from src.reader.magic_pdf_reader import MagicPDFReader
from src.parser.magic_pdf_parser import MagicPDFTransform

prompt = (
    "作为国学大师，你将扮演一个人工智能国学问答助手的角色，完成一项对话任务。在这个任务中，你需要根据给定的已知国学篇章以及问题，给出你的结论。请注意，你的回答应基于给定的国学篇章，而非你的先验知识，且注意你回答的前后逻辑不要出现"
    "重复，且不需要提到具体文件名称。\n任务示例如下：\n示例国学篇章：《礼记 大学》大学之道，在明明德，在亲民，在止于至善。\n问题：什么是大学？\n回答：“大学”在《礼记》中代表的是一种理想的教育和社会实践过程，旨在通过个人的"
    "道德修养和社会实践达到最高的善治状态。\n注意以上仅为示例，禁止在下面任务中提取或使用上述示例已知国学篇章。\n现在，请对比以下给定的国学篇章和给出的问题。如果已知国学篇章中有该问题相关的原文，请提取相关原文出来。\n"
    "已知国学篇章：{context_str}\n问题: {query}\n回答：\n"
)

# Three ways to specify the model:
#   1. Specify the model name (e.g. 'internlm2-chat-7b'):
#           the model will be automatically downloaded from the Internet;
#   2. Specify the model name (e.g. 'internlm2-chat-7b') ​​+ set
#      the environment variable `export LAZYLLM_MODEL_PATH="/path/to/modelzoo"`:
#           the model will be found in `path/to/modelazoo/internlm2-chat-7b/`
#   3. Directly pass the absolute path to TrainableModule:
#           `path/to/modelazoo/internlm2-chat-7b`

## lazy model 
# embed_model = lazyllm.TrainableModule("/home/mnt/share_server/models/BAAI--bge-large-zh-v1.5").start()
# rerank_model = Reranker("ModuleReranker", model="/home/mnt/share_server/models/BAAI--bge-reranker-large", topk=1, output_format='content', join=True).start()

## Custom Model
embed_model = LazyEmbeddingFactory.create_from(LazyEmbeddingFactory.BGE_M3_DENSE)
rerank_model = RerankComponent()
# llm_mode = 

documents = Document(dataset_path="/home/mnt/jisiyuan/projects/LazyRAG-Mineru/data/test14", embed=embed_model, manager=False)
documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=1024, chunk_overlap=100)
# documents.create_node_group(name="magicpdf", transform=MagicPDFTransform)

# register reader
# documents.add_reader("**/*.pdf", MagicPDFReader)


with pipeline() as ppl:
    with parallel().sum as ppl.prl:
        prl.retriever1 = Retriever(documents, group_name="sentences", similarity="cosine", topk=6)
        # prl.retriever2 = Retriever(documents, "CoarseChunk", "bm25_chinese", 0.003, topk=3, chunk_size=10240)
    # ppl.reranker = rerank_model | bind(query=ppl.input, topk=3)
    # ppl.formatter = (lambda nodes, query: dict(context_str=nodes, query=query)) | bind(query=ppl.input)
    # ppl.formatter = (lambda nodes, query: query + "\n".join([node.text for node in nodes])) | bind(query=ppl.input)
    # ppl.print_node = print_node
    ppl.fomatter = RAGContextFormatter() | bind(query=ppl.input)
    # ppl.print_query = print_query
    ppl.llm = LazyLlmFactory.create_from(LazyLlmFactory("Qwen/Qwen2.5-32B")) | bind(temperature=0.0)
    # ppl.llm = lazyllm.TrainableModule("/home/mnt/share_server/models/internlm2-chat-7b").prompt(lazyllm.ChatPrompter(prompt, extro_keys=["context_str"]))  # internlm2-chat-7b 


if __name__ == "__main__":

    
    answer = ppl('宁德时代股权激励或员工持股计划是怎么样的')  # 3月15日政府发行国债的净融资量是多少
    LOG.info('answer:')
    LOG.info(answer)
