# RAG (Retrieval-Augmented Generation，检索增强生成)

- 实际实现方式是，开发一个工具让模型调用去特定文本库中检索，再将检索结果和用户提示词一起喂给模型，模型输出结论

```bash
+-----------+               +----------+
    | Your Data |               |   User   |
    +-----------+               +----+-----+
          |                          ^
    +-----+------------+             |      (1) query
    | [DB] Structured  +--+          |  +------------------+
    | [Doc] Unstruct.  +--| (index)  |  |                  |
    | [API] Program.   +--+-----> [ Index ]                |
    +-----+------------+             |                     |
                                     | (2) prompt + query  |
                                     |     + relevant data |
                                     v                     |
                                +----------+               |
                                |   LLM    +---------------+
                                +----------+  (3) response
```