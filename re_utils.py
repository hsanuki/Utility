# coding: utf-8
import re
"""
例
"""
pattern = r"(築[0-9]+年)"
data["age"] = data["age"].apply(lambda x: re.search(pattern, x))

"""
正規表現
"""
# カッコ内に示されたものいずれか
[a-z], [abc]
# 否定
"<[^>]>"
# ０回以上の繰り返し
*
# なんでもいい
.
# １回以上の繰り返し
+
# matchした部分の取り出し
([a-z])
→　re.search(patter, x).group(1)
# 直前の文字が全くないか，１つだけあるという意味
→ windows?
# |で区切られたいずれかの文字列が存在した時に使う
|
→　IBM | マイクロソフト
"""
検索方法
"""
match・・文字列の先頭でmatchするか
search・・どこでもいいのでmatchするか
findall・・matchする部分を全て抽出する

"""
使用例
"""
pattern = r"([0-9]+.*[0-9]+)g"
# 括弧内に何か文字があるかチェックする
pattern = r"《.+?》
