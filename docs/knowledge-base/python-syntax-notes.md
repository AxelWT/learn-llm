- 查漏补缺

## 目录

- [note-1-字典解包语法](#note-1-字典解包语法)
- [note-2-range()函数剖析](#note-2-range函数剖析)
- [note-3-判断字符中是否有 HTML 字符](#note-3-判断字符中是否有-html-字符)

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

#### note-2-range()函数剖析

```bash
range是一个生成器吗 yield 实现？                                                                                                                                                  
                                                                                                                                                                                     
⏺ 不是。range 不是生成器，也没有用 yield 实现。
                                                                                                                                                                                     
  ┌────────┬───────────────────────────────┬─────────────────┐                                                                                                                       
  │  类型   │           实现方式             │   可重复迭代      │
  ├────────┼───────────────────────────────┼─────────────────┤                                                                                                                       
  │ range  │ 类，实现 __iter__ 和 __next__   │ ✅ 可以多次迭代   │
  ├────────┼───────────────────────────────┼─────────────────┤                                                                                                                       
  │ 生成器  │ yield 或生成器表达式             │ ❌ 只能迭代一次  │                                                                                                                        
  └────────┴───────────────────────────────┴─────────────────┘                                                                                                                       
                                                                                                                                                                                     
  示例对比                                                                                                                                                                           
                                                           
  range（可以重复迭代）：                                                                                                                                                            
  r = range(5)                                             
  list(r)  # [0, 1, 2, 3, 4]                                                                                                                                                         
  list(r)  # [0, 1, 2, 3, 4]  # 再次迭代，仍然有效                                                                                                                                   
                                                                                                                                                                                     
  生成器（只能迭代一次）：                                                                                                                                                           
  g = (x for x in range(5))  # 生成器表达式                                                                                                                                          
  list(g)  # [0, 1, 2, 3, 4]                                                                                                                                                         
  list(g)  # []  # 第二次迭代，已耗尽                                                                                                                                                
                                                                                                                                                                                     
  range 的实现原理                                                                                                                                                                   
                                                                                                                                                                                     
  # range 是一个类（伪代码示意）                                                                                                                                                     
  class range:                                                                                                                                                                       
      def __init__(self, start, stop, step):                                                                                                                                         
          self.start = start                                                                                                                                                         
          self.stop = stop                                                                                                                                                           
          self.step = step                                                                                                                                                           
                                                                                                                                                                                     
      def __iter__(self):                                                                                                                                                            
          # 返回一个新的迭代器对象                                                                                                                                                   
          return RangeIterator(self.start, self.stop, self.step)                                                                                                                     
                                                                                                                                                                                     
  class RangeIterator:                                                                                                                                                               
      def __init__(self, start, stop, step):                                                                                                                                         
          self.current = start                                                                                                                                                       
          self.stop = stop                                                                                                                                                           
          self.step = step                                                                                                                                                           
                                                                                                                                                                                     
      def __next__(self):                                                                                                                                                            
          if self.current >= self.stop:                                                                                                                                              
              raise StopIteration                                                                                                                                                    
          value = self.current                                                                                                                                                       
          self.current += self.step                                                                                                                                                  
          return value                                                                                                                                                               
                                                                                                                                                                                     
  关键点：                                                                                                                                                                           
  - range 对象本身不是迭代器，是可迭代对象                                                                                                                                           
  - 每次调用 __iter__() 会创建新的迭代器                                                                                                                                             
  - 所以可以重复迭代                                       
                                                                                                                                                                                     
  验证                                                                                                                                                                               
                                                                                                                                                                                     
  r = range(5)                                                                                                                                                                       
  print(type(r))        # <class 'range'>                                                                                                                                            
                                                                                                                                                                                     
  # range 对象本身不是迭代器                                                                                                                                                         
  print(iter(r) is r)   # False（每次 iter() 创建新迭代器）                                                                                                                          
                                                                                                                                                                                     
  # 生成器本身就是迭代器                                                                                                                                                             
  g = (x for x in range(5))                                                                                                                                                          
  print(iter(g) is g)   # True（生成器就是自己的迭代器）                                                                                                                             
                                                                                                                                                                                     
  总结：range 是惰性计算的类，不占内存，可重复迭代；生成器是 yield 实现，只能用一次。
```

#### note-3-判断字符中是否有 HTML 字符
```bash
判断字符串中是否有 HTML 实体编码，可以用这个方法：

  import html

  def has_html_entities(text):
      """检查字符串是否包含 HTML 实体编码"""
      # HTML 实体的常见模式：
      # - 数字实体：&#039; &#x27; 等
      # - 命名实体：&amp; &lt; &gt; &quot; 等

      unescaped = html.unescape(text)
      return unescaped != text  # 如果解码后不同，说明有实体

  # 使用示例
  text = "I&#039;m a transformer called BERT"
  print(has_html_entities(text))  # True

  normal_text = "I'm a transformer called BERT"
  print(has_html_entities(normal_text))  # False

  # 解码结果
  print(html.unescape(text))  # "I'm a transformer called BERT"

  常见 HTML 实体：

  ┌────────────┬────────────┬─────────────────────┐
  │   实体编码   │   解码字符   │        说明         │
  ├────────────┼────────────┼─────────────────────┤
  │   &amp;    │     &      │    ampersand        │
  ├────────────┼────────────┼─────────────────────┤
  │   &lt;     │     <      │    less than        │
  ├────────────┼────────────┼─────────────────────┤
  │   &gt;     │     >      │    greater than     │
  ├────────────┼────────────┼─────────────────────┤
  │   &quot;   │     "      │    quotation mark   │
  ├────────────┼────────────┼─────────────────────┤
  │   &apos;   │     '      │    apostrophe       │
  ├────────────┼────────────┼─────────────────────┤
  │   &#039;   │     '      │    数字实体(单引号)  │
  ├────────────┼────────────┼─────────────────────┤
  │   &#x27;   │     '      │    十六进制实体      │
  ├────────────┼────────────┼─────────────────────┤
  │   &nbsp;   │    空格    │    non-breaking space│
  ├────────────┼────────────┼─────────────────────┤
  │   &copy;   │     ©      │    copyright        │
  ├────────────┼────────────┼─────────────────────┤
  │   &reg;    │     ®      │    registered       │
  └────────────┴────────────┴─────────────────────┘
   
  
```