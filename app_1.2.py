import streamlit as st
import requests
import xml.etree.ElementTree as ET
import pandas as pd
import re
import io
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
from collections import defaultdict, Counter
import plotly.express as px

st.set_page_config(page_title="PubMed AI 工具", layout="wide")
st.title("🔬 PubMed AI 文献检索工具")

# ===== 说明 =====
st.info("关键词用空格分隔，例如：pkm2 ldha lactate")
st.info("年份=近n年，例如5表示近5年")
st.warning("""
⚠️ 注意事项  
- 本工具用于科研辅助  
- AI筛选存在误差，请人工复核  
- PubMed接口可能限速  
- 首次运行模型加载较慢  
""")
st.markdown("""
### 🧠 Confidence 说明
- 值域 0~1  
- < 0.55 已筛除  
- 0.55~0.7：可信度一般，需要人工复核  
- 0.7~0.85：可信度较高  
- >0.85：可信度高，机制句很可能可靠
""")

# ===== 输入 =====
keywords_input = st.text_input("关键词")
years = st.number_input("近几年", 1, 30, 5)
max_results = st.number_input("文献数", 1, 200, 20)

# ===== 模型加载 =====
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()
SIM_THRESHOLD = 0.55

# ===== 分类 =====
CATEGORIES = [
    "Metabolism","Inflammation","Immune",
    "Signal Transduction","Cell Death",
    "Oxidative Stress","Other"
]

CATEGORY_COLORS = {
    "Metabolism":"#FF6F61",
    "Inflammation":"#6B5B95",
    "Immune":"#88B04B",
    "Signal Transduction":"#F7CAC9",
    "Cell Death":"#92A8D1",
    "Oxidative Stress":"#955251",
    "Other":"#B565A7"
}

def classify(sentence):
    labels = [
        "metabolism","inflammation","immune response",
        "signaling pathway","apoptosis",
        "oxidative stress","other"
    ]
    e1 = model.encode(sentence, convert_to_tensor=True)
    e2 = model.encode(labels, convert_to_tensor=True)
    return CATEGORIES[util.cos_sim(e1, e2)[0].argmax().item()]

# ===== PubMed =====
def search(query, years, n):
    y = datetime.now().year
    q = f"({query}) AND ({y-years}:{y}[dp])"
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    r = requests.get(url, params={"db":"pubmed","term":q,"retmax":n,"retmode":"json"})
    return r.json()["esearchresult"]["idlist"]

def get_full_abstract(article):
    texts = []
    for t in article.findall(".//AbstractText"):
        label = t.attrib.get("Label", "")
        text = "".join(t.itertext())
        if label:
            texts.append(f"{label}: {text}")
        else:
            texts.append(text)
    return "\n".join(texts).strip()

def fetch(pmids):
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    r = requests.get(url, params={"db":"pubmed","id":",".join(pmids),"retmode":"xml"})
    root = ET.fromstring(r.text)
    papers=[]
    for a in root.findall(".//PubmedArticle"):
        title=a.findtext(".//ArticleTitle")
        pmid=a.findtext(".//PMID")
        full_abstract = get_full_abstract(a)
        authors=", ".join([au.findtext("LastName") for au in a.findall(".//Author") if au.findtext("LastName")][:5])
        journal=a.findtext(".//Journal/Title")
        year=a.findtext(".//PubDate/Year")
        papers.append({
            "pmid":pmid,"title":title,"full_abstract":full_abstract,
            "authors":authors,"journal":journal,"year":year
        })
    return papers

# ===== NLP =====
def split(text):
    return [i.strip() for i in re.split(r'[.!?]', text) if len(i.strip())>20]

def conf(s, kw):
    e1=model.encode(s, convert_to_tensor=True)
    e2=model.encode(" ".join(kw), convert_to_tensor=True)
    return util.cos_sim(e1,e2).item()

def conf_label(score):
    if score < 0.55:
        return "已过滤", "gray"
    elif score < 0.7:
        return "一般", "orange"
    elif score < 0.85:
        return "较高", "blue"
    else:
        return "高", "green"

def conf_bg_color(score):
    if score < 0.55:
        score = 0.55
    score = min(score,1.0)
    start_rgb = (237,231,246)
    end_rgb = (103,58,183)
    r = int(start_rgb[0] + (end_rgb[0]-start_rgb[0])*(score-0.55)/0.45)
    g = int(start_rgb[1] + (end_rgb[1]-start_rgb[1])*(score-0.55)/0.45)
    b = int(start_rgb[2] + (end_rgb[2]-start_rgb[2])*(score-0.55)/0.45)
    return f'rgb({r},{g},{b})'

# ===== 主逻辑 =====
if st.button("开始检索"):
    kw = re.split(r'[,\s]+', keywords_input.strip())
    query = " AND ".join(kw)
    pmids = search(query, years, max_results)
    papers = fetch(pmids)
    paper_dict = defaultdict(lambda:{"Sentences":[],"Mechanisms":set()})
    prog = st.progress(0)
    for i,p in enumerate(papers):
        for s in split(p["full_abstract"]):
            if sum(k in s.lower() for k in kw) < 2:
                continue
            c = conf(s, kw)
            if c < SIM_THRESHOLD:
                continue
            m = classify(s)
            paper_dict[p["pmid"]]["Sentences"].append(s)
            paper_dict[p["pmid"]]["Mechanisms"].add(m)
        prog.progress((i+1)/len(papers))
    if not paper_dict:
        st.warning("没有结果")
    else:
        rows=[]
        mech_counter=Counter()
        year_counter=Counter()
        for p in papers:
            pid=p["pmid"]
            if pid not in paper_dict:
                continue
            sentences=" | ".join(paper_dict[pid]["Sentences"])
            mechs=", ".join(paper_dict[pid]["Mechanisms"])
            for m in paper_dict[pid]["Mechanisms"]:
                mech_counter[m]+=1
            if p["year"]:
                year_counter[p["year"]]+=1
            max_conf = max([conf(s, kw) for s in paper_dict[pid]["Sentences"]]) if paper_dict[pid]["Sentences"] else 0
            rows.append({
                "PMID":pid,
                "Title":p["title"],
                "Authors":p["authors"],
                "Journal":p["journal"],
                "Year":p["year"],
                "Mechanism":mechs,
                "Sentences":sentences,
                "Max_Conf": max_conf,
                "Full_Abstract": p["full_abstract"]
            })
        df=pd.DataFrame(rows)
        df = df.sort_values(by='Max_Conf', ascending=False)
        st.success(f"{len(df)}篇文献")

        # ===== 文献展示（每行最多3篇，剩余固定1/3宽） =====
        st.subheader("📄 文献展示（按Confidence排序）")
        max_cols = 3
        num_blocks = len(df)
        blocks = df.to_dict('records')
        from math import ceil
        for i in range(0, num_blocks, max_cols):
            row_blocks = blocks[i:i+max_cols]
            n_cols = len(row_blocks)
            widths = [1/3]*n_cols
            cols = st.columns(widths)
            for j, row in enumerate(row_blocks):
                with cols[j]:
                    conf_text, conf_color = conf_label(row['Max_Conf'])
                    bg_color = conf_bg_color(row['Max_Conf'])

                    # 高亮 abstract 中同时包含所有关键词的句子
                    sentences = re.split(r'(?<=[.!?]) +', row['Full_Abstract'])
                    highlighted_sentences = []
                    for s in sentences:
                        if all(k.lower() in s.lower() for k in kw):  # ✅ 全关键词同时出现
                            highlighted_sentences.append(f"<span style='background-color:#fff9c4'>{s}</span>")
                        else:
                            highlighted_sentences.append(s)
                    highlighted_abstract = " ".join(highlighted_sentences)

                    st.markdown(f"<div style='background-color:{bg_color};padding:10px;border-radius:8px;'>", unsafe_allow_html=True)
                    st.markdown(f"**{row['Title']}**")
                    st.markdown(f"<span style='color:orange'>{row['Year']}</span> | {row['Journal']} | {row['Authors']}", unsafe_allow_html=True)
                    st.markdown(f"Confidence: <span style='color:{conf_color}'>{conf_text} ({row['Max_Conf']:.2f})</span>", unsafe_allow_html=True)
                    mech_tags = ""
                    for m in row['Mechanism'].split(","):
                        m = m.strip()
                        color = CATEGORY_COLORS.get(m,"gray")
                        mech_tags += f"<span style='background-color:{color};color:white;padding:2px 6px;border-radius:4px;margin-right:3px'>{m}</span>"
                    st.markdown(mech_tags, unsafe_allow_html=True)
                    st.markdown(f"<div style='max-height:200px; overflow-y:scroll; padding:5px; border:1px solid #ddd; border-radius:5px'>{highlighted_abstract}</div>", unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

        # ===== 可视化分析 =====
        st.subheader("📊 可视化分析")
        col1, col2 = st.columns(2)

        mech_df = pd.DataFrame({
            "Mechanism": list(mech_counter.keys()),
            "Count": list(mech_counter.values()),
            "PMIDs": [", ".join(df[df['Mechanism'].str.contains(m)]['PMID']) for m in mech_counter.keys()]
        })
        fig1 = px.bar(mech_df, x="Mechanism", y="Count", color="Mechanism", hover_data=["PMIDs"])
        col1.plotly_chart(fig1, use_container_width=True)

        year_df = pd.DataFrame({
            "Year": list(year_counter.keys()),
            "Count": list(year_counter.values()),
            "PMIDs": [", ".join(df[df['Year']==y]['PMID']) for y in year_counter.keys()]
        })
        fig2 = px.bar(year_df, x="Year", y="Count", color="Year", hover_data=["PMIDs"])
        col2.plotly_chart(fig2, use_container_width=True)

        # ===== 下载 =====
        buf=io.BytesIO()
        df.to_excel(buf,index=False)
        buf.seek(0)
        st.download_button("下载Excel",buf,"results.xlsx")