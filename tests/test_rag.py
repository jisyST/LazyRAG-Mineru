import lazyllm
lazyllm.component_register.new_group('demo')

@lazyllm.component_register('demo')
def test1(input):
    return f'input is {input}'

@lazyllm.component_register('demo')
def test2(input):
    return f'input is 2 {input}'

# @lazyllm.component_register.cmd('demo')
# def test2(input):
#     return f'echo input is {input}'

print(lazyllm.demo.test1()(1))

for fc in ['test1', 'test2']:
    if hasattr(lazyllm.demo, fc):  # 检查是否存在对应的方法
        func = getattr(lazyllm.demo, fc)()  # 获取并实例化方法
        result = func(1)  # 调用方法并传入参数
        print(f"{fc} result: {result}")
    else:
        print(f"{fc} not found")