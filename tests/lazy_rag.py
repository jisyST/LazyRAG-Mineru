import os
import sys

curt_file_path = os.path.realpath(__file__) if "__file__" in globals() else os.getcwd()
sys.path.append(curt_file_path[:curt_file_path.index("src") + len("src")])
# sys.path = ["/home/mnt/jisiyuan/LazyLLM"] + sys.path

from lazyllm import ModuleBase, pipeline, parallel, ifs, switch, warp, bind, Document, Retriever, Reranker
from lazyllm import OnlineEmbeddingModule, TrainableModule
from lazyllm.tools.rag import DocNode
from itertools import product
from lazyllm.tools.rag.transform import NodeTransform
from lazyllm.tools.rag.transform import SentenceSplitter
from lazyllm.module import ModuleBase
from lazyllm import LOG
from typing import Any, List


from lazyrag.modules.rag.query_pipeline.components.join import RRFJoinComponent
from lazyrag.modules.rag.query_pipeline.components.reranker import RerankComponent
from lazyrag.components.llm.lazy_llm_factory import LazyLlmFactory
from lazyrag.config import BaseConfig, Constants
from lazyrag.components.embedding import LazyEmbeddingFactory
from lazyrag.modules.rag.schemas.base import Component
from lazyrag.modules.rag.query_pipeline.components.formatter.prompt_formatter import (
    LLMChatFormatter,
    RAGContextFormatter,
)

from src.rag.utils import LLMChatFormatter, generate_retrieve_configs
# from reader.magic_pdf_reader import MagicPDFReader


embed_model = TrainableModule("bge-large-zh-v1.5")
rerank_model = Reranker("ModuleReranker", model="bge-reranker-large", topk=6, output_format='content', join=True)


class RetrieverComponent(Component):
    '''input kb path， config, query；output list[nodes]'''
    def __init__(self, return_trace: bool = False, **kwargs):
        super().__init__(return_trace=return_trace, **kwargs)
        
    def _create_doc(self, dataset_path: str, name: str = "default", manager: bool = False, node_groups: dict = {}):
        doc = Document(
                dataset_path=dataset_path,
                embed=embed_model,
                name=name,
                # doc_fields=DocumentManager.CUSTOM_DOC_FIELDS,
                manager=manager,
                # store_conf=get_milvus_store_conf(kb_group_name=DocumentManager.DEDAULT_GROUP_NAME),
            )
        # 普通data pipeline vs 自定义 data pipeline，需要重写NodeTransform
        doc.create_node_group(name="sentence", transform=SentenceSplitter, chunk_size=2048, chunk_overlap=100)
        # 重点写
        # doc.add_reader("**/*.pdf", MagicPDFReader)
        return doc

    def _create_retriever(self, kb_path: str, group_name: str, similarity_type: str, top_k: int) -> Retriever:
        """Create retriever."""
        doc = self._create_doc(kb_path)

        if similarity_type == "semantic":
            embed_keys = ["bge_m3_dense"]
            similarity = "cosine"
        elif similarity_type == "keyword":
            embed_keys = ["bge_m3_sparse"]
            similarity = "bm25"  # "bm25_chinese"
        else:
            raise ValueError(f"invaild similarity_type {similarity_type}")
        retriever = Retriever(doc=doc, group_name=group_name, embed_keys=embed_keys, topk=top_k, similarity=similarity)
        return retriever

    def _run_component(self, input, **kwargs) -> List[Any]:
        """Run component."""
        """
        1. 获取对应的retrievers ppl
        2. 运行检索
        """
        query = input.get('query')
        kb_path = input.get('kb_path')
        top_k = input.get('recall_top_k', 5)
        group_name = input.get('recall_node_group', "sentence")
        similarity_type = input.get('recall_similarity_type', "semantic")
        LOG.error(kb_path)
        # filters: dict = input.get('filters', {}).copy()
        # filters.update({"kb_id": [kb_id], "disabled": [0]})
        
        retriever = self._create_retriever(kb_path, group_name, similarity_type, top_k)
        nodes = retriever(query)
        for node in nodes:
            LOG.info(node.text)
            LOG.info("==================================")
        return nodes
        

with pipeline() as search_ppl:
    # search_ppl.config_list = generate_retrieve_configs | bind(
    #             kb_ids=search_ppl.input["kb_ids"], config=search_ppl.input["recall_strategy"], filters_list=search_ppl.input["filters"]
    #         )
    search_ppl.retrieve = RetrieverComponent() # | bind(query=search_ppl.input["query"])
    # search_ppl.retrieve = warp(RetrieverComponent()).aslist() | bind(query=search_ppl.input["query"])
    # search_ppl.retrieval_join = RRFJoinComponent()
    # search_ppl.rerank = RerankComponent()

def tmp(input):
    # LOG.error(input)
    return {'search_nodes': input}

with pipeline() as rag_ppl:
    rag_ppl.search = search_ppl
    rag_ppl.tmp = tmp
    rag_ppl.fomatter = RAGContextFormatter() | bind(query=rag_ppl.input["query"])
    rag_ppl.llm = LazyLlmFactory.create_from(LazyLlmFactory("Qwen/Qwen2.5-32B"))


# input = dict(
#     query = 'hello world',
#     kbs = ["path1", "path2"],
#     config = {
#         "recall_top_k": 5,
#         "recall_node_group": ["groupA", "groupB"],
#         "recall_similarity_type": ["cosine", "euclidean"]
#     }
# )

input = dict(
    query='需求侧居民消费占全球GDP份额来看，2024年增速和2023年增速的对比情况是怎么样的',
    kb_path="/home/mnt/jisiyuan/projects/LazyRAG-Mineru/data/test3",
)

# ans = search_ppl(input)
# for node in ans:
#     print(node.text)

ans = rag_ppl(input)
print(ans)
