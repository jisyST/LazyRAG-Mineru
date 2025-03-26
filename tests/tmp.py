# import os

# def display_directory_tree(path: str, indent: str = ""):
#     """
#     递归打印指定路径的目录树结构。
    
#     :param path: 目录路径
#     :param indent: 缩进字符串，用于格式化输出
#     """
#     if not os.path.exists(path):
#         print("路径不存在:", path)
#         return
    
#     if not os.path.isdir(path):
#         print("指定路径不是一个目录:", path)
#         return
    
#     # 获取当前目录下的所有文件和文件夹
#     entries = sorted(os.listdir(path))
    
#     for index, entry in enumerate(entries):
#         if entry == "data":
#             continue
#         full_path = os.path.join(path, entry)
#         is_last = index == len(entries) - 1
#         prefix = "└── " if is_last else "├── "
        
#         print(indent + prefix + entry)
        
#         if os.path.isdir(full_path):
#             new_indent = indent + ("    " if is_last else "│   ")
#             display_directory_tree(full_path, new_indent)

# # 示例用法
# directory_path = "/home/mnt/jisiyuan/projects/LazyRAG-Mineru"  # 替换为你想要显示的目录路径
# display_directory_tree(directory_path)




from lazyllm.tools.rag.readers import PDFReader
from pathlib import Path
import json


path = Path("/home/mnt/jisiyuan/projects/LazyRAG-Mineru/data/test15/国新证券-流动性周度观察.pdf")
nodes = PDFReader()._load_data(file=path)
jsons = []
for node in nodes:
    jsons.append({'text': node.text} | node.metadata)

with open("/home/mnt/jisiyuan/projects/LazyRAG-Mineru/data/parse_res/中国银河证券-扫描版.json", "w", encoding="utf-8") as f:
    json.dump(jsons, f, ensure_ascii=False, indent=4)
    
    
    
# from lazyllm.tools import Document, Retriever
# from lazyllm.tools.rag import DocNode, NodeTransform
# import time
# from lazyllm.tools.rag.transform import SentenceSplitter

# import lazyllm
# from lazyrag.config import BaseConfig, Constants
# from lazyrag.components.embedding import LazyEmbeddingFactory
# embed = {}
# for key, _ in Constants.embedding.items():
#     embed[key] = LazyEmbeddingFactory.create_from(LazyEmbeddingFactory(key))
# print(embed.keys())
# # embed_model = lazyllm.TrainableModule("bge-large-zh-v1.5").start()
# # =============================
# # 初始化知识库
# # =============================
# # 定义DFA算法
# class DFAFilter:
#     def __init__(self, sensitive_words=['外挂']):
#         self.root = {}
#         self.end_flag = "is_end"
#         for word in sensitive_words:
#             self.add_word(word)

#     def add_word(self, word):
#         node = self.root
#         for char in word:
#             if char not in node:
#                 node[char] = {}
#             node = node[char]
#         node[self.end_flag] = True

#     def filter(self, text, replace_char="*"):
#         result = []
#         start = 0
#         length = len(text)

#         while start < length:
#             node = self.root
#             i = start
#             while i < length and text[i] in node:
#                 node = node[text[i]]
#                 if self.end_flag in node:
#                     # 匹配到敏感词，替换为指定字符
#                     result.append(replace_char * (i - start + 1))
#                     start = i + 1
#                     break
#                 i += 1
#             else:
#                 # 未匹配到敏感词，保留原字符
#                 result.append(text[start])
#                 start += 1

#         return ''.join(result)
   
   
# # 注册为tranform
# class DFATranform(NodeTransform):
#     def __init__(self, sensitive_words):
#         super(__class__, self).__init__(num_workers=1)
#         self.dfafilter = DFAFilter(sensitive_words)

#     def transform(self, node: DocNode, **kwargs):
#         return self.dfafilter.filter(node.get_text())

#     def split_text(self, text: str):
#         if text == '':
#             return ['']
#         paragraphs = text.split(self.splitter)
#         return [para for para in paragraphs]
    

# # 定义知识库路径
# law_data_path = "/home/mnt/jisiyuan/projects/LazyLLM/tests/dataset1/法务"
# product_data_path = "/home/mnt/jisiyuan/projects/LazyLLM/tests/dataset1/产品"
# support_data_path = "/home/mnt/jisiyuan/projects/LazyLLM/tests/dataset1/支持"


# Document.create_node_group('sentences', transform=SentenceSplitter, chunk_size=512, chunk_overlap=100)
# Document.create_node_group(name="dfa_filter", parent="sentences", transform=DFATranform(['合同']))
# # 初始化知识库对象
# law_knowledge_base = Document(law_data_path, name='法务知识库', embed=embed)
# product_knowledge_base = Document(product_data_path, name='产品知识库', embed=embed)
# support_knowledge_base = Document(support_data_path, name='用户支持知识库', embed=embed)
# # 组合法务 + 产品知识库，处理与产品相关的法律问题
# retriever_product = Retriever(
#     [law_knowledge_base, product_knowledge_base],
#     group_name="dfa_filter",         #  dfa_filter 添加过滤  sentences： 不添加过滤
#     embed_keys=['bge_m3_dense'],
#     similarity="cosine",       
#     topk=1                
# )

# 组合法务 + 客服知识库，处理客户合同投诉
# retriever_support = Retriever(
#     [product_knowledge_base, support_knowledge_base],
#     group_name="dfa_filter",       
#     embed_keys=['bge_m3_dense'],
#     similarity="cosine",       
#     topk=1                
# )

# product_question = "A产品功能参数和产品合规性声明"
# product_response = retriever_product(product_question)
# print()
# print(f"========== query: {product_question } ===========")
# print()
# for node in product_response:
#     print(node.text)
#     print("="*100)

# support_question = "B产品的主要成分的投诉的处理方式"
# support_response = retriever_support(support_question)
# print()
# print(f"========== query: {support_question } ===========")
# print()
# for node in support_response:
#     print(node.text)
#     print("="*100)

