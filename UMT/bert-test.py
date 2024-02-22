import re
import torch
import math
import numpy as np
from random import *
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
from torch.utils.data import Dataset, DataLoader

# device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# 自定义的对话文本
text = (
    'Hello, how are you? I am Romeo.\n'  # R
    'Hello, Romeo My name is Juliet. Nice to meet you.\n'  # J
    'Nice meet you too. How are you today?\n'  # R
    'Great. My baseball team won the competition.\n'  # J
    'Oh Congratulations, Juliet\n'  # R
    'Thank you Romeol\n'  # J
    'Where are you going today?\n'  # R
    'I am going shopping. What about you?\n'  # J
    'I am going to visit my grandmother. she is not very well'  # R
)
# 正则表达式去除标点，字母小写化，然后按照换行符分割
sentences = re.sub('[.,?!]', '', text.lower()).split('\n')
# 将预处理后的token放在word_list中
word_list = list(set(' '.join(sentences).split()))
# 将token映射为index
word2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
for i, w in enumerate(word_list):
    word2idx[w] = i + 4
# index -> token
idx2word = {i: w for i, w in enumerate(word2idx)}
vocab_size = len(word2idx)

# 做token_list
token_list = []
for sen in sentences:
    token_list.append([word2idx[i] for i in sen.split()])
# [[29, 6, 32, 37, 33, 31, 13], [29, 13, 16, 22, 21...

# bert parameters
maxlen = 30  # max length
batch_size = 6
max_pred = 5  # max tokrns of prediction 最多5个token被mask
n_layers = 6  # encoder 层数
n_heads = 12  # multi-heads个数
d_model = 768  # embedding维度
d_ff = 768 * 4  # 全连接神经网络的维度
d_k = d_v = 64  # dimension of K(=Q), V
n_segments = 2  # 每一行由多少句话构成


# 预处理
# IsNext和NotNext的个数得一样
# sample IsNext and NotNext to be same in small batch size
def make_data():
    batch = []
    positive = negative = 0  # positive若样本中两条样本相邻 加 1 ； 不相邻则negative 加 1 但最终需要保证positive 与negative比例相等
    while positive != batch_size / 2 or negative != batch_size / 2:
        # tokens_a_index：第一个文本的索引 tokens_b_index:第二个文本的索引
        # randrange(0-8) 随机抽取两个索引，判断tokens_a_index + 1 是否等于 tokens_b_index 得到两个文本是否相邻
        tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(
            len(sentences))  # sample random index in sentences
        # 通过文本的索引tokens_a_index，获取文本中的每个token的索引放到tokens_a中；tokens_b_index同理
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]
        # 在token前后拼接[CLS] 与 [SEP]
        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']]
        # segment_ids表示模型中的segment embedding，前一句全为0，个数是【 1 + len(tokens_a) + 1 】个 ，前后两个1表示特殊标识的1；后一句全0
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)

        # MASK LM
        # 先取整个句子长度的15%做mask,但需要注意的是有时整个句子长度太短，比如6个token，6*0.15小于1，此时需要和1进行比较取最大值
        # 但有时句子过长，我们设置的界限是mask不超过5个，因此要和max_pred取最小值
        n_pred = min(max_pred, max(1, int(len(input_ids) * 0.15)))  # 15 % of tokens in one sentence
        # cand_maked_pos：候选mask标记，由于特殊标记[CLS]和[SEP]做mask是无意义的，因此需要排除
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          # 只要不是[CLS]和[SEP] 就可以将索引存入cand_maked_pos
                          if token != word2idx['[CLS]'] and token != word2idx['[SEP]']]  # candidate masked position
        shuffle(cand_maked_pos)  # 由于是随机mask,将cand_maked_pos列表中的元素随机打乱
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:  # 取乱序的索引cand_maked_pos前n_pred 做mask
            masked_pos.append(pos)  # masked_pos：mask标记对应的索引
            masked_tokens.append(input_ids[pos])  # masked_tokens：mask标记对应的原来的token值
            # Bert模型中mask标记有80%被替换为真正的mask，10% 被随机替换为词表中的任意词，10%不变
            if random() < 0.8:  # 80% 的 概率 被替换为真正的mask
                input_ids[pos] = word2idx['[MASK]']  # make mask
            elif random() > 0.9:  # 10% 的概率被随机替换为词表中的任意词
                index = randint(0, vocab_size - 1)  # random index in vocabulary 从词表中随机选择一个词的索引 可以是本身
                while index < 4:  # can't involve 'CLS', 'SEP', 'PAD' 但不能是特殊标记，也就是说索引要大于4
                    index = randint(0, vocab_size - 1)  # 索引小于4需要重新获取一个随机数
                input_ids[pos] = index  # replace 用随机的词替换该位置的token

        # Zero Paddings
        # 对长度不足maxlen30的文本 补 PAD
        n_pad = maxlen - len(input_ids)  # 30 - 文本长度 = 需要补 0 的个数
        input_ids.extend([0] * n_pad)  # 不足30的位置token补0
        segment_ids.extend([0] * n_pad)  # segment embedding 补 0

        # Zero Padding (100% - 15%) tokens
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)  # 保证masked_tokens和masked_pos长度始终为max_pred(5)
            masked_pos.extend([0] * n_pad)

        # 判断两个文本是否相邻  tokens_a_index + 1 是否等于 tokens_b_index  ；前提是positive 和negative比例相等
        if tokens_a_index + 1 == tokens_b_index and positive < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])  # IsNext
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])  # NotNext
            negative += 1
    print(batch)
    return batch


# Proprecessing Finished 数据预处理结束

batch = make_data()
input_ids, segment_ids, masked_tokens, masked_pos, isNext = zip(*batch)
input_ids, segment_ids, masked_tokens, masked_pos, isNext = \
    torch.LongTensor(input_ids), torch.LongTensor(segment_ids), torch.LongTensor(masked_tokens), \
        torch.LongTensor(masked_pos), torch.LongTensor(isNext)


class MyDataSet(Data.Dataset):
    def __init__(self, input_ids, segment_ids, masked_tokens, masked_pos, isNext):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos
        self.isNext = isNext

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.segment_ids[idx], self.masked_tokens[idx], self.masked_pos[idx], self.isNext[
            idx]


loader = Data.DataLoader(MyDataSet(input_ids, segment_ids, masked_tokens, masked_pos, isNext), batch_size, True)


# 构建BERT模型
def get_attn_pad_mask(seq_q, seq_k):
    batch_size, seq_len = seq_q.size()
    # eq(zero) is PAD token
    pad_attn_mask = seq_q.data.eq(0).unsqueeze(1)  # [batch_size,1,seq_len]
    # eq(0)表示和0相等的返回True，不相等返回False。unsqueeze(1)在第一维上
    return pad_attn_mask.expand(batch_size, seq_len, seq_len)  # [batch_size,seq_len,seq_len]

# 激活函数gelu()
def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class Embedding(nn.Module):
    def __init__(self):
        super(Embedding, self).__init__()
        self.tok_embed = torch.nn.Embedding(vocab_size, d_model)
        self.pos_embed = torch.nn.Embedding(maxlen, d_model)
        self.seg_embed = torch.nn.Embedding(n_segments, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, seg):
        seq_len = x.size(1)
        pos = torch.arange(seq_len, dtype=torch.long)
        pos = pos.unsqueeze(0).expand_as(x)  # [seq_len] -> [batch_size,seq_len]
        # embedding相加
        embedding = self.tok_embed(x) + self.pos_embed(pos) + self.seg_embed(seg)
        return self.norm(embedding)


class ScaleDotProductAttention(nn.Module):
    def __init__(self):
        super(ScaleDotProductAttention, self).__init__()

    # Q: [batch_size, seq_len, d_model], K: [batch_size, seq_len, d_model], V: [batch_size, seq_len, d_model]
    def forward(self, Q, K, V, attn_mask):
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(d_k)  # matmul做内积，transpose做转置
        scores.masked_fill_(attn_mask, -1e9)  # fills elements of self tensor with value where mask is one.
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context


class MultiHeadAttention(nn.Module):
    def __init__(self):
        super(MultiHeadAttention, self).__init__()
        self.W_Q = nn.Linear(d_model, d_k * n_heads)
        self.W_K = nn.Linear(d_model, d_k * n_heads)
        self.W_V = nn.Linear(d_model, d_v * n_heads)
        self.linear = nn.Linear(n_heads * d_v, d_model)
        self.LN = nn.LayerNorm(d_model)

    def forward(self, Q, K, V, attn_mask):
        residual, batch_size = Q, Q.size(0)
        # (B,S,D) --proj-> (B,S,D) --split-> (B,S,H,W) --trans-> (B,H,S,W)
        q_s = self.W_Q(Q).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # [batch_size,n_heads,seq_len,d_k]
        k_s = self.W_K(K).view(batch_size, -1, n_heads, d_k).transpose(1, 2)  # [batch_size,n_heads,seq_len,d_k]
        v_s = self.W_V(V).view(batch_size, -1, n_heads, d_v).transpose(1, 2)  # [batch_size,n_heads,seq_len,d_v]

        # 将attn_mask三维拓展为四维才能和 Q K V矩阵相乘
        attn_mask = attn_mask.unsqueeze(1).repeat(1, n_heads, 1, 1)  # [batch_size,n_heads,seq_len,seq_len]

        # context:[batch_size,n_heads,seq_len,d_v],attn:[batch_size,n_heads,seq_len,seq_len]
        context = ScaleDotProductAttention()(q_s, k_s, v_s, attn_mask)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, n_heads * d_v) # context: [batch_size, seq_len, n_heads * d_v]
        output = self.linear(context)
        return self.LN(output + residual)
        # output: [batch_size,seq_len,d_model]


class PoswiseFeedForwardNet(nn.Module):
    def __init__(self):
        super(PoswiseFeedForwardNet, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        # (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_ff) -> (batch_size, seq_len, d_model)
        return self.fc2(gelu(self.fc1(x)))


class EncoderLayer(nn.Module):
    def __init__(self):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MultiHeadAttention()
        self.pos_ffn = PoswiseFeedForwardNet()

    def forward(self, enc_inputs, enc_self_attn_mask):
        # self-attention 通过MultiHeadAttention实现 传入三个enc_inputs作用是分别于W(Q,K,V)相乘生成 Q,K,V矩阵
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs, enc_self_attn_mask)  # same Q,K,V
        # pos_ffn 特征提取 通过PoswiseFeedForwardNet实现
        enc_outputs = self.pos_ffn(enc_outputs)
        return enc_outputs


class BERT(nn.Module):
    def __init__(self):
        super(BERT, self).__init__()
        self.embedding = Embedding()  # 构建词向量矩阵
        self.layers = nn.ModuleList([EncoderLayer() for _ in range(n_layers)])
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Dropout(0.5),
            nn.Tanh(),
        )
        self.classifier = nn.Linear(d_model, 2)  # 二分类任务
        self.linear = nn.Linear(d_model, d_model)  # 添加全连接层
        self.activ2 = gelu  # 这是上边定义的gelu，self.activ2就是gelu函数

        embed_weight = self.embedding.tok_embed.weight
        self.fc2 = nn.Linear(d_model, vocab_size, bias=False)
        self.fc2.weight = embed_weight

    def forward(self, input_ids, segment_ids, masked_pos):
        output = self.embedding(input_ids, segment_ids)  # [batch_size,seq_len,d_model]
        enc_self_attn_mask = get_attn_pad_mask(input_ids, input_ids)  # [batch_size,maxlen,maxlen]
        for layer in self.layers:
            output = layer(output, enc_self_attn_mask)  # output: [batch_size, max_len, d_model]
        # it will be decided by first token(CLS)
        h_pooled = self.fc(output[:, 0])  # [batch_size, d_model]
        logits_clsf = self.classifier(h_pooled)  # 得到判断两句话是不是上下句关系的结果

        # 得到被mask位置的词，准备与正确词进行比较
        masked_pos = masked_pos[:, :, None].expand(-1, -1, d_model)  # [batch_size,max_pred,d_model]
        # 将token 与masked_pos进行对齐
        h_masked = torch.gather(output, 1, masked_pos)  # masking position [batch_size,max_pred,d_model]
        # 让h_masked(带mask的token)过全连接层 ，再过激活函数2
        h_masked = self.activ2(self.linear(h_masked))  # [batch_size,max_pred,d_model]
        # 再解码到词表大小 d_model-> vocab_size
        logits_lm = self.fc2(h_masked)  # [batch_size,max_pred,vocab_size]
        return logits_lm, logits_clsf  # 前者预测被mask的词，后者预测两句话是否为上下句关系


model = BERT()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(50):
    for input_ids, segment_ids, masked_tokens, masked_pos, isNext in loader:
        logits_lm, logits_clsf = model(input_ids, segment_ids, masked_pos)
        loss_lm = criterion(logits_lm.view(-1, vocab_size), masked_tokens.view(-1))
        loss_lm = (loss_lm.float()).mean()  # MLM loss
        loss_clsf = criterion(logits_clsf, isNext)  # NSP loss
        loss = loss_clsf + loss_lm
        if (epoch + 1) % 10 == 0:
            print('Epoch: %04d' % (epoch + 1), 'loss=', '{:.6f}'.format(loss))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# 测试
input_ids, segment_ids, masked_tokens, masked_pos, isNext = batch[1]
print(text)
print([idx2word[w] for w in input_ids if idx2word[w] != '[PAD]'])

logits_lm, logits_clsf = model(torch.LongTensor([input_ids]), torch.LongTensor([segment_ids]), torch.LongTensor([masked_pos]))
logits_lm = logits_lm.data.max(2)[1][0].data.numpy()

"""
max(2):再第2维上取最大值，取完后维度：[batch,max_pred],[batch,max_pred]。此时有两个维度值。第一个是具体的值，第二个是位置
[1]:取到最大值对应的位置。维度是：[batch,max_pred]
[0]:因为batch为1（只取了第一组值），所以此时维度是：[max_pred]
注：max会使tensor降维
"""
print('masked token list:', [pos for pos in masked_tokens if pos != 0])
print('predict masked tokens list:', [pos for pos in logits_lm if pos != 0])

logits_clsf = logits_clsf.data.max(1)[1].data.numpy()[0]
print('isNext:', True if isNext else False)
print('predict isNext:', True if logits_clsf else False)