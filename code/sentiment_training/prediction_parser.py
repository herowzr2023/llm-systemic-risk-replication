import re

VERSION = 'unified-output-neutral-fallback-v1.2'
GROUPS = {
    1: ['正面', '积极', '看涨', '看好', '乐观', '利好', 'positive', 'bullish'],
    -1: ['负面', '消极', '看跌', '悲观', '利空', 'negative', 'bearish'],
    0: ['中性', '中立', 'neutral', '不确定', '无法判断', '难以判断', '信息不足'],
}
MAP = {word: label for label, words in GROUPS.items() for word in words}
TOKEN = '|'.join(sorted(MAP, key=len, reverse=True))
RX = re.compile(r'(?<![A-Za-z])(?:' + TOKEN + r')(?![A-Za-z])', re.I)
PREFIX = re.compile(
    r'^\s*[\s`*#：:，,。.!！\[\]【】（）()“”]*'
    r'(?:(?:情绪|情感|态度|sentiment|分类|标签|判断|答案|结果)'
    r'(?:倾向|分类|标签)?\s*(?:为|是|[:：=])\s*)?', re.I
)
NEG = re.compile(
    r'(?:不|非|没|没有|并非|不是|无|未|not\s+)'
    r'(?:明显|显著|属于|具有|偏向|很|是|的|\s)*$', re.I
)
FRAME = re.compile(
    r'情绪|情感|态度|倾向|分类|标签|判断|答案|结果|表达|表现|传达|'
    r'sentiment|classified|classification', re.I
)
CONDITIONAL = re.compile(r'如果|若|可能|也许|例如|比如|或者|还是|if\b|maybe\b', re.I)
SPLIT = re.compile(r'[。！？!?\n;；,，()（）【】\[\]/／、]')
STANDALONE = re.compile(r'[\s`*#：:,，“”]*(?:' + TOKEN + r')[\s`*#：:,，“”]*', re.I)
EXPLICIT_ASSERTION = re.compile(
    r'(?:该帖|本帖|这条(?:帖子|内容)?|文本|内容|整体|它)\s*'
    r'(?:的)?\s*(?:实际|总体|整体)?\s*(?:为|是|属于|偏向)\s*'
    r'(?:' + TOKEN + r')', re.I
)


def parse(raw):
    text = re.split(
        r'\n\s*(?:用户|user|标题|title)\s*[:：]',
        str(raw).strip(), maxsplit=1, flags=re.I
    )[0]
    evidence = []
    m = RX.match(text, PREFIX.match(text).end())
    if m and not re.match(
        r'^[\u4e00-\u9fff]|^\s*(?:或|和|与|及|/|、|or\b|and\b)',
        text[m.end():], re.I
    ):
        evidence.append((MAP[m.group().lower()], m.group()))

    for clause in SPLIT.split(text):
        hits = list(RX.finditer(clause))
        if not hits or CONDITIONAL.search(clause):
            continue
        if not (FRAME.search(clause) or STANDALONE.fullmatch(clause)):
            # Only an explicit assertion about the post itself is evidence
            # without a classification frame; market mentions are ignored.
            for assertion in EXPLICIT_ASSERTION.finditer(clause):
                h = RX.search(assertion.group())
                if h and not NEG.search(assertion.group()[:h.start()]):
                    evidence.append((MAP[h.group().lower()], h.group()))
            continue
        for h in hits:
            if not NEG.search(clause[:h.start()]):
                evidence.append((MAP[h.group().lower()], h.group()))

    labels = {label for label, _ in evidence}
    if len(labels) == 1:
        return labels.pop(), 'unambiguous_classification', '|'.join(word for _, word in evidence)
    return 0, ('conflicting_neutral' if len(labels) > 1 else 'fallback_neutral'), '|'.join(
        word for _, word in evidence
    )
