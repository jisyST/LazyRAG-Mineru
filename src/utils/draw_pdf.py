import fitz  # PyMuPDF

def draw_bboxes_on_pdf(input_pdf_path, bbox_list, output_pdf_path):
    # 打开PDF文档
    pdf_document = fitz.open(input_pdf_path)

    # 遍历每一页
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)  # 加载页面

        # 遍历bbox列表
        for bbox in bbox_list:
            if bbox['page'] == page_num:  # 检查bbox是否属于当前页
                rect = fitz.Rect(bbox['x0'], bbox['y0'], bbox['x1'], bbox['y1'])  # 创建矩形框
                page.draw_rect(rect, color=(1, 0, 0), width=2)  # 绘制红色矩形框

    # 保存带有框框的新PDF文档
    pdf_document.save(output_pdf_path)
    pdf_document.close()

# 示例bbox列表
# 每个bbox包含：所属页数（从0开始）、矩形框的四个坐标（x0, y0, x1, y1）
bbox_list = [
    # {'page': 0, 'x0': 257, 'y0': 542, 'x1': 552, 'y1': 630},  # 第1页的框
    {'page': 0, 'x0': 241, 'y0': 652, 'x1': 553, 'y1': 793},    # 第2页的框
]

# 调用函数处理PDF
input_pdf_path = "/home/mnt/jisiyuan/projects/LazyRAG-Mineru/data/test11/珀莱雅.pdf"
output_pdf_path = "/home/mnt/jisiyuan/projects/LazyRAG-Mineru/data/test11/珀莱雅pro1.pdf"
draw_bboxes_on_pdf(input_pdf_path, bbox_list, output_pdf_path)


# "bbox": [
#             241,
#             652,
#             553,
#             793
#         ],

#     {
#         "text": "投资建议:美妆行业增速放缓,同时竞争激烈,公司继续坚持大单品策略,以优质产品和运营抢占市场份额,获得稳健地高质量增长,且在截止目前的双十一销售中稳定发挥。我们维持对公司 2024-2025 年的业绩预测,并略调整 2026 年预测,2024-2026 年对应归母净利润分别为 15.52、19.03、23.08 亿元(2026 年原预测为 23.09 亿元),对应当前市值(2024-10-28市值)分别约 24.7、20.1 和 16.6 倍 PE。公司作为国货龙头,近年来形成了系统化、规范化、标准化的运营体系,日益参与到与国际美妆集团的竞争中,未来有望进一步获取更大的大众美妆市场份额。目前公司估值水平处在历史中低位区间,结合当前估值水平及市场环境情况,维持“推荐”评级。",
#         "file_name": "珀莱雅.pdf",
#         "type": "text",
#         "bbox": [
#             257,
#             542,
#             552,
#             630
#         ],