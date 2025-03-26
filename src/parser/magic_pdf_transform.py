import functools
from typing import Any, List, Sequence
from lazyllm import LOG
from lazyllm.tools.rag import NodeTransform, DocNode
from typing import Union


class MagicPDFTransform(NodeTransform):
    def __init__(self, **kwargs):
        super().__init__()

    def transform(self, document: DocNode, **kwargs) -> List[Union[str, DocNode]]:
        return 

    def batch_forward(self, documents, node_group, **kwargs):
        nodes = ConsolidationTextNodeParser.parse_nodes(documents)
        for node in nodes:
            node.excluded_embed_metadata_keys = ['bbox', 'lines']
            node.metadata = {'file_name': node.metadata['file_name']}
            node._group = node_group
        return nodes


class ConsolidationTextNodeParser:
    '''
    遍历nodes, 将所有非title类型的节点合并
    metadata中
        有 text_level 字段的 为title
        有 title 字段的 为正文
    '''

    @classmethod
    def class_name(cls) -> str:
        return "ConsolidationTextNodeParser"
    
    @staticmethod
    def parse_nodes(
        nodes: List[DocNode],
        **kwargs: Any,
    ) -> List[DocNode]:
        def _merge(_result, _node):
            '''合并算法部分'''
            if not _result:
                _result.append(_node)
            elif _node.metadata.get("text_level", 0):
                _result.append(_node)
            else:
                # if node.metadata['bbox'] == []: import pdb;pdb.set_trace()
                _result[-1]._content = '\n\n'.join([_result[-1]._content, _node._content])
                if _result[-1].metadata.get('bbox', None):
                    _result[-1].metadata['bbox'] = ConsolidationTextNodeParser._merge_bbox(
                        top_bbox=_result[-1].metadata['bbox'], bottom_bbox=_node.metadata['bbox']
                    )
                if _result[-1].metadata.get('lines', None):
                    _result[-1].metadata['lines'].extend(_node.metadata['lines'])

            return _result

        def _merge_nodes(result, node):
            # 判断是否合并的逻辑部分
            if not result:
                result.append([node])
            elif node.metadata.get("text_level", 0):
                # 长度判断规则后续按需调整优化
                if len('\n\n'.join([i._content for i in result[-1]])) < 4096:
                    result[-1] = functools.reduce(_merge, result[-1], [])
                else:
                    result = result[:-1] + [result[-1]]
                result.append([node])
            elif len('\n\n'.join([i._content for i in result[-1]] + [node._content])) >= 4096:
                result[-1] = functools.reduce(_merge, result[-1], [])
                result.append([node])
            else:
                result[-1].extend([node])

            return result

        _merged_nodes = functools.reduce(_merge_nodes, nodes, [])
        # 处理最后一组结果
        _merged_nodes[-1] = functools.reduce(_merge, _merged_nodes[-1], [])
        return [node for nodes in _merged_nodes for node in nodes]
    
    @staticmethod
    def _merge_bbox(top_bbox: List, bottom_bbox: List) -> List:
        """
        合并两个坐标框（bbox）。
        约定 bbox 格式为: [left_top_x, left_top_y, right_bottom_x, right_bottom_y]

        :param top_bbox: 上方 bbox
        :param bottom_bbox: 下方 bbox
        :return: 合并后的 bbox
        """
        # assert len(top_bbox) == 4 and len(bottom_bbox) == 4, f"每个 bbox 必须包含 4 个值\n top_bbox: {top_bbox}\n bottom_bbox: {bottom_bbox}"
        assert len(top_bbox) == 4, f"每个 bbox 必须包含 4 个值\n top_bbox: {top_bbox}"
        if len(bottom_bbox) != 4:
            bottom_bbox = top_bbox

        assert top_bbox[0] <= top_bbox[2] and top_bbox[1] <= top_bbox[3], f"top_bbox 格式不正确：{top_bbox}"
        assert bottom_bbox[0] <= bottom_bbox[2] and bottom_bbox[1] <= bottom_bbox[3], "bottom_bbox 格式不正确"

        # 确保 top_bbox 的 y 坐标较小
        if top_bbox[1] > bottom_bbox[1]:
            # LOG.warning(f"BBox 顺序需要检查：交换 top_bbox: {top_bbox} 和 bottom_bbox: {top_bbox}")
            top_bbox, bottom_bbox = bottom_bbox, top_bbox

        # 合并两个 bbox
        new_bbox = [
            max(top_bbox[0], bottom_bbox[0]),  # left_top_x
            top_bbox[1],  # left_top_y
            max(top_bbox[2], bottom_bbox[2]),  # right_bottom_x
            bottom_bbox[3],  # right_bottom_y
        ]
        return new_bbox




