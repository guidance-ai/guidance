import re


def camel_to_snake(input_string):
    # 使用正则表达式来匹配大写字母，并在其前面添加下划线，然后将字符串转换为小写
    snake_case_str = re.sub(r"([A-Z])", r"_\1", input_string).lower()
    # 如果结果以下划线开头，则去掉开头的下划线
    if snake_case_str.startswith("_"):
        snake_case_str = snake_case_str[1:]
    return snake_case_str


def snake(value):
    """Transform the value from camel case to snake case."""
    return camel_to_snake(value)
