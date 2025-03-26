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
# from lazyrag.servers.doc_parser.doc_parser_client import DocParserClient
from src.reader.magic_pdf_reader import MagicPDFReader
from src.parser.magic_pdf_transform import MagicPDFTransform


## Custom Model
embed_dense_model = LazyEmbeddingFactory.create_from(LazyEmbeddingFactory.BGE_M3_DENSE)
embed_sparse_model = LazyEmbeddingFactory.create_from(LazyEmbeddingFactory.BGE_M3_SPARSE)
rerank_model = RerankComponent()


documents = Document(dataset_path="/home/mnt/jisiyuan/projects/LazyRAG-Mineru/data/test15", embed={"dense": embed_dense_model, "sparse":embed_sparse_model}, manager=False)
# documents.create_node_group(name="sentences", transform=SentenceSplitter, chunk_size=10240, chunk_overlap=100)
documents.create_node_group(name="magicpdf", transform=MagicPDFTransform)

# register reader
documents.add_reader("**/*.pdf", MagicPDFReader)  # DocParserClient


with pipeline() as ppl:
    with parallel().sum as ppl.prl:
        prl.retriever1 = Retriever(documents, group_name="sentences", embed_keys=["dense"], similarity="cosine", topk=3)
        prl.retriever2 = Retriever(documents, group_name="sentences", embed_keys=["sparse"], similarity="bm25_chinese", topk=3)
    ppl.fomatter = RAGContextFormatter() | bind(query=ppl.input)
    ppl.llm = LazyLlmFactory.create_from(LazyLlmFactory("Qwen/Qwen2.5-32B")) | bind(temperature=0.0)


if __name__ == "__main__":

    while True:
        query = input("input your query: ")
        if query == 'exit':
            break
        print(f"recieve your query: {query} \n")

        answer = ppl(query)  # 3月15日政府发行国债的净融资量是多少
        print(f"{'='*50} answer {'='*50}")
        print(answer)
        print('='*110)
