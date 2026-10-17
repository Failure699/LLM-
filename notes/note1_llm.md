# 大语言模型（LLM）基础

大语言模型（Large Language Model）是一个接收文字、输出文字的程序，本质是"文字接龙"机器——给它一段文字，预测下一个词，一路续写下去。

## 调 API 的三个核心参数

- `model`：用哪个模型，比如 deepseek-chat
- `messages`：对话历史，四种角色 system / user / assistant / tool
- `temperature`：随机性旋钮，0 偏稳定，1.5 天马行空

## 关键点

- 模型无状态，每次请求要重发整个 messages
- system prompt 放列表最前面，用来定人设、立规矩
- temperature=0 只是"尽量稳定"，GPU 浮点运算的不确定性让它不保证逐字一致
