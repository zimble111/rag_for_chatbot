import pandas as pd
import re
import torch
import faiss
import numpy as np
from transformers import (
    DPRQuestionEncoder, DPRQuestionEncoderTokenizer,
    DPRContextEncoder, DPRContextEncoderTokenizer,
    AutoTokenizer, AutoModelForCausalLM
)
from collections import Counter
import gradio as gr


config = {
    "retriever": {
        "model_q": "facebook/dpr-question_encoder-multiset-base",
        "model_c": "facebook/dpr-ctx_encoder-multiset-base",
        "max_len_q": 64,
        "max_len_c": 256,
        "top_k": 10
    },
    "generator": {
        "model_name": "mistralai/Mistral-7B-Instruct-v0.3",
        "max_new_tokens": 256,
        "temperature": 0.7,
        "top_p": 0.9
    }
}

device = "cuda" if torch.cuda.is_available() else "cpu"


def f1_score(pred: str, ref: str) -> float:
    pred_tokens = pred.lower().split()
    ref_tokens  = ref.lower().split()
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(ref_tokens)
    return 2 * precision * recall / (precision + recall)

def rouge_l(pred: str, ref: str) -> float:
    p = pred.lower().split()
    r = ref.lower().split()
    n, m = len(p), len(r)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n):
        for j in range(m):
            if p[i] == r[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    lcs_len = dp[n][m]
    if lcs_len == 0:
        return 0.0
    prec = lcs_len / n
    rec  = lcs_len / m
    return 2 * prec * rec / (prec + rec)


def chat_respond(message, history):
    raw = rag.answer(message)
    answer = re.sub(r'^[\s\S]*?Ответ:\s*', '', raw).strip()
    return answer


class DPRRetriever:
    def __init__(self, passages: list[str], cfg: dict):
        # Load DPR encoders
        self.q_tok = DPRQuestionEncoderTokenizer.from_pretrained(cfg["model_q"])
        self.q_enc = DPRQuestionEncoder.from_pretrained(cfg["model_q"]).to(device)
        self.c_tok = DPRContextEncoderTokenizer.from_pretrained(cfg["model_c"])
        self.c_enc = DPRContextEncoder.from_pretrained(cfg["model_c"]).to(device)
        self.max_len_q = cfg["max_len_q"]
        self.max_len_c = cfg["max_len_c"]
        self.top_k     = cfg["top_k"]
        self.passages  = passages

        # Build embeddings and FAISS index
        embs = []
        for text in passages:
            inp = self.c_tok(text, return_tensors="pt",
                             truncation=True, max_length=self.max_len_c).to(device)
            with torch.no_grad():
                emb = self.c_enc(**inp).pooler_output
            embs.append(emb.cpu().numpy())
        self.embeddings = np.vstack(embs).astype('float32')
        faiss.normalize_L2(self.embeddings)
        self.index = faiss.IndexFlatIP(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def retrieve(self, question: str) -> list[str]:
        inp = self.q_tok(question, return_tensors="pt",
                         truncation=True, max_length=self.max_len_q).to(device)
        with torch.no_grad():
            q_emb = self.q_enc(**inp).pooler_output.cpu().numpy()
        faiss.normalize_L2(q_emb)
        _, idxs = self.index.search(q_emb, self.top_k)
        return [self.passages[i] for i in idxs[0]]


class MistralGenerator:
    def __init__(self, cfg: dict):
        self.tokenizer = AutoTokenizer.from_pretrained(
            cfg["model_name"], trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            cfg["model_name"],
            trust_remote_code=True,
            load_in_8bit=True,
            device_map="auto"
        )
        self.max_new_tokens = cfg["max_new_tokens"]
        self.temperature    = cfg["temperature"]
        self.top_p          = cfg["top_p"]

    def generate(self, question: str, contexts: list[str]) -> str:
        ctx_block = "\n\n".join(contexts) if contexts else "(контекст отсутствует)"
        prompt = (
            "<s>[INST] "
            "Вы — бот помощник менеджера ПВЗ. "
            "Отвечайте строго по контексту; если контекст пуст — дайте краткую уместную реакцию. "
            "Отвечайте на русском.\n\n"
            f"Вопрос: {question}\n\n"
            "Контекст:\n" + ctx_block + "\n\n"
            "Ответ: [/INST]"
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            **inputs,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            eos_token_id=self.tokenizer.eos_token_id,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(out[0], skip_special_tokens=True)


class RAGPipeline:
    def __init__(self, retriever: DPRRetriever, generator: MistralGenerator):
        self.retriever = retriever
        self.generator = generator

    def answer(self, question: str) -> str:
        contexts = self.retriever.retrieve(question)
        if not contexts:
            return "Информация в базе отсутствует."
        return self.generator.generate(question, contexts)


if __name__ == "__main__":
    # initialize pipeline
    df = pd.read_excel("knowledge_base.xlsx")
    raw_texts = df['chunk'].dropna().astype(str).tolist()
    # chunk texts ~120 words
    passages = []
    for raw in raw_texts:
        parts, buf = re.split(r"[.!?]\s+", raw), []
        for sent in parts:
            buf.append(sent)
            if len(" ".join(buf).split()) >= 120:
                passages.append(" ".join(buf)); buf = []
        if buf: passages.append(" ".join(buf))

    retriever = DPRRetriever(passages, config["retriever"])
    generator = MistralGenerator(config["generator"])
    rag = RAGPipeline(retriever, generator)
    chat_demo = gr.ChatInterface(
        fn=chat_respond,
        title="PVZ Chat-бот",
        description="Задавайте вопросы по работе ПВЗ. Бот отвечает кратко и по делу.",
        theme="soft"
    )

    chat_demo.launch(share=True)
