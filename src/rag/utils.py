import asyncio
from typing import Any, Dict, List
from itertools import product

import lazyllm
from lazyllm import LOG
from lazyllm.module import OnlineChatModuleBase
from lazyllm.tools.rag.transform import NodeTransform

from lazyrag.modules.rag.schemas.base import Component
from lazyrag.modules.rag.schemas import LlmStrategy


class SimpleLlmComponent(Component):

    def __init__(
        self,
        llm: OnlineChatModuleBase,
        prompter,
        return_trace: bool = False,
        stream: bool = False,
        chat_history: List[str] = [],
        **kwargs,
    ):
        super().__init__(return_trace=return_trace)
        self.stream = stream
        self.llm = llm.share()
        self.llm.prompt(prompter)
        self.chat_history = chat_history

    @property
    def series(self):
        return "LlmComponent"

    @property
    def type(self):
        return "LLM"
    
    async def astream_iterator(self, input, llm, llm_strategy):
        lazyllm.globals._init_sid()
        with lazyllm.ThreadPoolExecutor(1) as executor:
            future = executor.submit(llm, input, llm_chat_history=[], stream_output=True, **llm_strategy)
            while True:
                if value := lazyllm.FileSystemQueue().dequeue():
                    yield ''.join(value)
                elif future.done():
                    break
                else:
                    await asyncio.sleep(0.1)
            llm = None

    def _run_component(self, params, stream, **kwargs: Any) -> Any:
        try:
            query = params

            llm_strategy = kwargs.get("llm_strategy", {})
            if isinstance(llm_strategy, LlmStrategy):
                llm_strategy = llm_strategy.model_dump()

            # 优先使用llm_strategy中的system_prompt
            # TODO fix system prompt changed by user's input
            llm = self.llm.share(llm_strategy.pop("system_prompt")) if llm_strategy.get("system_prompt") else self.llm
            
            filtered_kw = {k: v for k, v in llm_strategy.items() if v is not None and k in custom_params_list}
            if stream:
                response = self.astream_iterator(query, llm, filtered_kw)
            else:
                response = llm.forward(
                    query, stream_output=False, llm_chat_history=self.chat_history, **filtered_kw
                )
                llm = None
            return response
        except Exception as e:
            llm = None
            raise e

llm_chat_instruction_cn = """## 人物描述
你是一个由商汤科技自主研发的智能助手，请回答用户的问题。

## 要求
- 请使用自己的先验知识，全面、准确、专业地回答问题。
- 如果问题存在多个角度，请按顺序依次回答。
"""

llm_chat_input_cn = """## 用户的问题
{query}
"""

class LLMChatFormatter(Component):
    def __init__(self, return_trace: bool = False, **kwargs) -> None:
        super().__init__(return_trace=return_trace, **kwargs)

    def _run_component(self, input, **kwargs) -> Any:
        query = kwargs.get("query")
        if isinstance(query, dict):
            query = query.get("query")
        return llm_chat_input_cn.format(query=query)
    
    
    
def generate_retrieve_configs(kb_ids: List[str], config: Any, filters_list: List[List[str]]) -> List[Dict[str, Any]]:
    """
    根据知识库 ID、配置和筛选条件，生成召回配置列表。

    Args:
        kb_ids (List[str]): 知识库 ID 列表。
        config (Any): 配置对象，包含 recall_top_k、recall_node_group、recall_similarity_type 等字段。
        filters_list (List[List[str]]): 与 kb_ids 对应的筛选条件列表。

    Returns:
        List[Dict[str, Any]]: 生成的召回配置列表。
    """
    result = []
    for kb_id, filters in zip(kb_ids, filters_list):
        for node_group, similarity_type, f in product(config.recall_node_group, config.recall_similarity_type, filters):
            result.append({
                "kb_id": kb_id,
                "recall_top_k": config.recall_top_k,
                "recall_node_group": node_group,
                "recall_similarity_type": similarity_type,
                "filters": f,
            })
    return result


from typing import Any

from lazyllm.tools.rag.doc_node import MetadataMode
# from lazyrag.common import log
from lazyrag.modules.rag.schemas.base import Component
from lazyrag.modules.rag.query_pipeline.components.utils import regroup_nodes
from lazyrag.modules.rag.prompt.chat_prompt_cn import standard_rag_input_cn, llm_chat_input_cn

# logger = log.getLogger(__name__)
RAG_CITING_CLAIM = (
    "- 回答中需标注引用的召回节点来源，以确保信息的可信度，不要引用没有帮助的召回节点。"
    "规定引用的索引形式为`[[id]]`，如果有多个索引，则用多个[[]]表示，如`[[id_1]][[id_2]]`。\n"
    "- 如果引用节点为文字类型（text），引用索引添加在句子末尾，例如`明天是晴天。[[id_1]]\\n`。请忽略文字类型节点中出现的图片url。\n"
    "- 如果引用节点为图片类型（image），请只引用与回答内容相关的图片，并采用图文并茂的方式回答问题，例如`段落1\\n[[id_1]]\\n段落2`。\n"
    "- 引用文字类型节点的句子不要并列引用图片节点。\n"
)
CHAIN_OF_THOUGHTS = (
    "请生成具有写作逻辑的专业回答，严格遵循回答思路，从多个角度回答问题。"
    "你的答案中可以包括多级标题，这可以体现你的专业性。\n"
    "## 回答思路\n{cot}\n")



class RAGContextFormatter(Component):
    def __init__(self, return_trace: bool = False, **kwargs) -> None:
        super().__init__(return_trace=return_trace, **kwargs)

    def _create_context_str(self, nodes: dict) -> str:
        nodes = regroup_nodes(nodes)
        node_str_list = []
        for index, node in nodes.items():
            file_name = node.metadata.get('file_name')
            node_type = 'image' if node.metadata.get('type') == 'image' else 'text'

            node_str = (
                f"召回节点[{index}]\n"
                f"来源文件：{file_name}\n"
                f"节点类型：{node_type}\n"
                f"节点内容：{node.text}\n"
            )
            node_str_list.append(node_str)
        context_str = "\n".join(node_str_list)
        return context_str
    
    def _run_component(self, input, **kwargs) -> Any:
        nodes = input
        query = kwargs.get("query")
        if isinstance(query, dict):
            query = query.get("query")

        if len(nodes):
            context_str = self._create_context_str(nodes)
            context_str += f"\n## 引用格式要求\n{RAG_CITING_CLAIM}\n"
        else:
            context_str = "使用你的先验知识回答用户的问题"
        # return {"context": context_str, "query": query}
        query = standard_rag_input_cn.format(context=context_str, query=query)
        LOG.info(" ============ query ============= ")
        LOG.info(query)
        return query



# class DataPipelineTransform(NodeTransform):
#     def __init__(self, 
#                  group_name,
#                  embed_metadata_keys: List[str] = ['file_name'],
#                  redis_store: RedisDocStore = RedisDocStore()
#                  ):
#         super().__init__()
#         self.group_name = group_name
#         self.embed_metadata_keys = embed_metadata_keys
#         self.redis_store = redis_store

#         self._build_transform_pipeline()
    
#     def _set_excluded_embed_metadata_keys(self, nodes: Union[DocNode, List[DocNode]]):
#         nodes = nodes if isinstance(nodes, List) else [nodes]
#         for node in nodes:
#             excluded_keys = [key for key in node.metadata if key not in self.embed_metadata_keys]
#             node.excluded_embed_metadata_keys = excluded_keys
#         return nodes

#     def batch_forward(
#         self, documents: Union[DocNode, List[DocNode]], node_group: str, **kwargs
#     ) -> List[DocNode]:
#         nodes = super().batch_forward(documents=documents, node_group=node_group, **kwargs)
#         if node_group == "block":
#             persist_nodes = filter(lambda node: not node.metadata.get("cache"), nodes)
#             for file_id, file_group in groupby(persist_nodes, key=lambda node: node.global_metadata.get("docid")):
#                 file_group = list(file_group)
#                 kb_id = file_group[0].global_metadata.get("kb_id")
#                 if self.redis_store:
#                     assert kb_id is not None, "kb_id cannot be None."    
#                     self.redis_store.delete_segments(kb_id=kb_id, file_id=file_id, group_name=node_group)
#             if self.redis_store:
#                 self.redis_store.add_nodes(nodes=nodes, group_name=node_group)
#         return nodes
            
#     def transform(self, document: DocNode, **kwargs) -> List[DocNode]:
#         # restore from redis
#         if self.group_name == "block":
#             file_hash = document.metadata.get('file_hash')
#             kb_id = document.global_metadata.get('kb_id')
#             file_id = document.global_metadata.get('docid')
#             if self.redis_store:
#                 cache_file_hash = self.redis_store.get_file_hash(kb_id=kb_id, file_id=file_id)
#                 if file_hash == cache_file_hash:
#                     return self.redis_store.get_nodes(kb_id=kb_id, file_id=file_id, group_name=self.group_name)

#         # transform
#         file_type = os.path.splitext(document.metadata['file_name'])[1].lstrip('.').lower()
#         content = document._content if isinstance(document._content, List) else [document._content]
#         segments = []
#         for item in content:
#             segment = Segment(**item) if isinstance(item, dict) else Segment(text=item, metadata=document.metadata)
#             segments.append(segment)
#         nodes = self.transform_ppl(segments, global_metadata=document.global_metadata, file_type=file_type, **kwargs)
#         return nodes

#     def _build_transform_pipeline(self, **kwargs):
#         with pipeline() as trans_ppl:
#             trans_ppl.splitter = switch(
#                 (lambda _, group_name, **kwargs: group_name == "block") | bind(group_name=self.group_name), ParserFactory(),
#                 (lambda _, group_name, **kwargs: group_name == "image") | bind(group_name=self.group_name), ImageNodeGroupParser(),
#                 (lambda _, group_name, **kwargs: group_name == "summary") | bind(group_name=self.group_name), SummaryParser(),
#                 'default', LineSplitter()
#             )
#             trans_ppl.convert = warp(lambda x, global_metadata: DocNode(
#                 text=x.text,
#                 group=self.group_name,
#                 metadata=x.metadata,
#                 global_metadata=global_metadata
#             )).aslist | bind(global_metadata=trans_ppl.kwargs["global_metadata"])
#             trans_ppl.excluded_embed_metadata_keys = self._set_excluded_embed_metadata_keys
#         self.transform_ppl = trans_ppl




from typing import Any, List, Union

from lazyllm import warp, pipeline, parallel, bind

from lazyrag.common import log
from lazyrag.components.reranker import LazyRerankerFactory
from lazyrag.modules.rag.schemas.base import Component
# from lazyrag.modules.rag.query_pipeline.components.schema import NodeWithScore
from lazyrag.modules.rag.schemas import RerankStrategy

# logger = log.getLogger(__name__)

DEFAULT_TEMPLATE = "{file_name}\n{title_1}\n{title_2}\n{text}"
DEFAULT_RERANK_MODEL_NAME = "rerank_v2_m3"
DEFAULT_RRF_K = 60


class RerankComponent(Component):
    """
    RerankerModule implements the logic for re-ranking retrieval nodes, supporting scoring
    nodes using multiple templates with different metadata.
    """

    def __init__(
        self,
        top_k: int = 6,
        weights: List[float] = None,
        text_templates: Union[List[str], str] = DEFAULT_TEMPLATE,
        **kwargs,
    ):
        super().__init__()
        self.top_k = top_k
        self.text_templates = [text_templates] if isinstance(text_templates, str) else text_templates
        self.node_group = []
        templates_num = len(self.text_templates)
        self.weights = (
            weights if (weights and templates_num > 1) or (templates_num == 1) else [1 / templates_num] * templates_num
        )

    def _get_model(self, model_name: str):
        rerank_model = LazyRerankerFactory.create_from(LazyRerankerFactory(model_name))
        return rerank_model

    def _join_nodes(self, nodes: List, **kwargs) -> List:
        """
        Combine and sort nodes based on their scores across multiple paths.

        Args:
            scores (List[List[float]]): Scores from different paths.
            nodes (List[NodeWithScore]): List of nodes.
            weights (Optional[List[float]]): Weights for each path. Defaults to equal weighting.

        Returns:
            List[NodeWithScore]: Top-k nodes after combining scores.
        """

        top_k = kwargs.get("top_k", 0)
        weights = kwargs.get("weights", [1.0] * len(nodes))

        node_score_dict = {}

        for path_idx, path_nodes in enumerate(nodes):
            weight = weights[path_idx]
            for node in path_nodes:
                if node.node_id not in node_score_dict:
                    node_score_dict[node.node_id] = (node, 0.0)  # Initialize node and score
                node_score_dict[node.node_id] = (
                    node,
                    node_score_dict[node.node_id][1] + weight * node.get_score(),
                )
        
        combined_nodes = []
        for node_id, (node, score) in node_score_dict.items():
            node.set_score(score)
            combined_nodes.append(node)
        sorted_nodes = sorted(combined_nodes, key=lambda x: x.get_score(), reverse=True)

        if top_k > 0:
            sorted_nodes = sorted_nodes[:top_k]
        return [node.node for node in sorted_nodes]

    def _run_component(self, nodes, query, **kwargs):
        """
        Run the reranking process for given nodes and query.

        Args:
            nodes (List[NodeWithScore]): List of nodes to rerank.
            query (str): The user query.

        Returns:
            List[NodeWithScore]: Re-ranked nodes.
        """
        top_k = 3
        if isinstance(query, dict):
            query = query.get("query", "")
        # rerank_strategy: RerankStrategy = kwargs.get("rerank_strategy")
        # rerank_models = rerank_strategy.rerank_models
        # top_k = rerank_strategy.rerank_top_k
        text_templates = ["{file_name}\n{text}", "{text}"]
        # templates_num = len(text_templates)
        # weights = [1 / templates_num] * templates_num if templates_num > 1 else [1]

        # create rerank model
        model = self._get_model
        
        def get_nodes_with_score(nodes, query):
            return [lambda nodes, query: model(
                                nodes=nodes,
                                query=query,
                                template=template
                            ) for template in text_templates]
        
        def join_nodes(nodes):
            nodes = sorted(key=lambda x:x.node_score, reverse=True)[:top_k]
            return [node.node for node in nodes]

        rerank_parallel = pipeline(get_nodes_with_score, self._join_nodes)

        result = rerank_parallel(nodes=nodes, query=query)
        return result


def get_image_path():
    return "/home/mnt/jisiyuan/projects/LazyRAG-Mineru/data/images"