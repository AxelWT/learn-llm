- 查漏补缺

#### note-1-字典解包语法

```bash
** 是Python的字典解包语法，将字典的键值对展开为函数的关键字参数。

batch的内容

batch = {
    'input_ids': tensor([[101, 2003, ...]]),
    'attention_mask': tensor([[1, 1, ...]]),
    'labels': tensor([1, 0, ...])
}

两种传参方式对比
# 方式1：手动传参                                                                                                                                                             
outputs = model(
    input_ids=batch['input_ids'],
    attention_mask=batch['attention_mask'],
    labels=batch['labels']
)

# 方式2：字典解包（更简洁）                                                                                                                                                   
outputs = model(**batch)

工作原理
** batch  →  自动展开为  →  input_ids = ..., attention_mask = ..., labels = ...

类似用法
params = {'lr': 0.001, 'weight_decay': 0.01}
optimizer = AdamW(model.parameters(), **params)
# 等价于                                                                                                                                                                      
optimizer = AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

总结

┌────────┬──────────────────────┬─────────────────────────────────────────┐
│  符号   │          作用         │                  示例                   │
├────────┼──────────────────────┼─────────────────────────────────────────┤
│ **dict │ 字典解包为关键字参数    │func(**{'a': 1, 'b': 2}) → func(a=1, b=2)│
├────────┼──────────────────────┼─────────────────────────────────────────┤
│ *list  │ 列表解包为位置参数      │ func(*[1, 2]) → func(1, 2)              │
└────────┴──────────────────────┴─────────────────────────────────────────┘

好处是代码简洁，且batch的键名与模型参数名自动匹配。
```