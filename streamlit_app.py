# streamlit_app.py
# Process Mining — продвинутые неэффективности с подтипами + вертикальный DFG и фикс pm4py.format_dataframe

import os
import io
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# --- мягкая проверка pm4py ---
try:
    import pm4py
    from pm4py.objects.log.util import dataframe_utils
    from pm4py.visualization.dfg import visualizer as dfg_visualizer
except ModuleNotFoundError:
    st.set_page_config(page_title="Process Mining — зависимости")
    st.error(
        "Модуль **pm4py** не установлен. Для Streamlit Cloud добавь в `requirements.txt` "
        "`pm4py>=2.7.11` и (опц.) `graphviz`; в `packages.txt` — `graphviz`."
    )
    st.stop()

import matplotlib.pyplot as plt
import shutil
import networkx as nx

st.set_page_config(page_title="Process Mining — продвинутые неэффективности", layout="wide")
st.title("🔍 Process Mining — продвинутые неэффективности (с подтипами)")

# =========================
# Загрузка
# =========================
with st.expander("⚙️ Загрузка данных", expanded=False):
    local_path = st.text_input("Путь к локальному файлу (CSV/Parquet)", value="case-championship-last.parquet")
    csv_sep = st.text_input("Разделитель CSV", value=",")
    csv_enc = st.text_input("Кодировка CSV", value="utf-8")

uploaded = st.file_uploader("Загрузите CSV или Parquet", type=["csv", "parquet"])

@st.cache_data(show_spinner=False)
def load_upload(f, sep=",", enc="utf-8"):
    if f.name.lower().endswith(".csv"):
        return pd.read_csv(f, sep=sep, encoding=enc)
    return pd.read_parquet(f)

@st.cache_data(show_spinner=False)
def load_path(p, sep=",", enc="utf-8"):
    p_l = p.lower()
    if p_l.endswith(".csv"):
        return pd.read_csv(p, sep=sep, encoding=enc)
    elif p_l.endswith(".parquet") or p_l.endswith(".pq"):
        return pd.read_parquet(p)
    else:
        raise ValueError("Поддерживаются .csv и .parquet")

df = None
if uploaded is not None:
    df = load_upload(uploaded, csv_sep, csv_enc)
elif local_path.strip() and os.path.exists(local_path.strip()):
    df = load_path(local_path.strip(), csv_sep, csv_enc)
    st.info(f"Загружен локальный файл: {local_path.strip()}")

if df is None or df.empty:
    st.info("⬆️ Загрузите лог или укажите путь.")
    st.stop()

st.subheader("📊 Предпросмотр")
st.dataframe(df.head(), use_container_width=True)

# =========================
# Маппинг + время + ресурс
# =========================
st.subheader("🧭 Маппинг колонок")
cols = df.columns.tolist()
col_case = st.selectbox("Колонка кейса", cols, index=0)
col_act  = st.selectbox("Колонка активности", cols, index=min(1, len(cols)-1))
col_ts   = st.selectbox("Колонка времени", cols, index=min(2, len(cols)-1))
col_res_opt = st.selectbox("Колонка ресурса/исполнителя (опционально)", ["<нет>"] + cols, index=0)

with st.expander("🕒 Парсинг времени", expanded=False):
    date_hint = st.text_input("Формат даты (опц.) например %Y-%m-%d %H:%M:%S", value="")
    coerce_ts = st.checkbox("Ошибочные даты → NaT", value=True)
    tz = st.text_input("Таймзона (например Europe/Moscow). Пусто — не менять", value="")

use_resource = col_res_opt != "<нет>"

keep_cols = [col_case, col_act, col_ts] + ([col_res_opt] if use_resource else [])
df = df[keep_cols].copy()
rename_map = {col_case: "case_id", col_act: "activity", col_ts: "timestamp"}
if use_resource:
    rename_map[col_res_opt] = "resource"
df = df.rename(columns=rename_map)

# to_datetime
if date_hint.strip():
    df["timestamp"] = pd.to_datetime(df["timestamp"], format=date_hint, errors=("coerce" if coerce_ts else "raise"))
else:
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors=("coerce" if coerce_ts else "raise"))

# TZ
if tz.strip():
    try:
        if df["timestamp"].dt.tz is None:
            df["timestamp"] = df["timestamp"].dt.tz_localize(tz, nonexistent="NaT", ambiguous="NaT")
        else:
            df["timestamp"] = df["timestamp"].dt.tz_convert(tz)
    except Exception as e:
        st.warning(f"Не удалось применить таймзону: {e}")

na_share = df["timestamp"].isna().mean()
if na_share > 0:
    st.warning(f"⚠️ {na_share:.1%} временных значений → NaT (будут отброшены).")
    df = df.dropna(subset=["timestamp"])

df["case_id"] = df["case_id"].astype(str).str.strip()
df["activity"] = df["activity"].astype(str).str.strip()
if use_resource:
    df["resource"] = df["resource"].astype(str).str.strip()
df = df[(df["case_id"] != "") & (df["activity"] != "")]
df = df.sort_values(["case_id", "timestamp"])
if df.groupby("case_id").size().max() < 2:
    st.error("Нужно ≥2 события на кейс.")
    st.stop()

# =========================
# Производные
# =========================
# ✅ FIX: передаём resource_key в pm4py только когда он есть (иначе некоторые версии кидают TypeError)
fmt_kwargs = dict(case_id="case_id", activity_key="activity", timestamp_key="timestamp")
if use_resource:
    fmt_kwargs["resource_key"] = "resource"

event_log = pm4py.format_dataframe(df, **fmt_kwargs)

event_log = dataframe_utils.convert_timestamp_columns_in_df(event_log)
if "time:timestamp" in event_log.columns:
    event_log = event_log.sort_values(["case:concept:name", "time:timestamp"])
else:
    event_log = event_log.sort_values(["case:concept:name"])
n_cases = event_log["case:concept:name"].nunique()
st.success(f"✅ Событий: {len(event_log):,} • Кейсов: {n_cases:,}")

# Межшаговые интервалы
df_sorted = df.copy()
df_sorted["next_activity"]  = df_sorted.groupby("case_id")["activity"].shift(-1)
df_sorted["next_timestamp"] = df_sorted.groupby("case_id")["timestamp"].shift(-1)
if use_resource:
    df_sorted["resource_next"] = df_sorted.groupby("case_id")["resource"].shift(-1)
df_sorted["delta_sec"] = (df_sorted["next_timestamp"] - df_sorted["timestamp"]).dt.total_seconds()
edges = df_sorted.dropna(subset=["next_activity", "delta_sec"]).copy()
edges["edge"] = list(zip(edges["activity"], edges["next_activity"]))

edge_stats = (
    edges.groupby("edge")["delta_sec"]
         .agg(median="median",
              p90=lambda s: np.percentile(s, 90),
              p95=lambda s: np.percentile(s, 95),
              p99=lambda s: np.percentile(s, 99),
              mean="mean", std="std", count="count")
         .reset_index()
)
edge_stats_dict = edge_stats.set_index("edge").to_dict(orient="index")
edge_median_map = {e: s["median"] for e, s in edge_stats_dict.items()}
edge_p95_map    = {e: s["p95"]    for e, s in edge_stats_dict.items()}
edge_p99_map    = {e: s["p99"]    for e, s in edge_stats_dict.items()}

def safe_pct(series, q, default=np.nan) -> float:
    s = pd.Series(series).dropna()
    return float(np.percentile(s, q)) if not s.empty else default

# =========================
# ИНДИКАТОРЫ (укорочено: ядро не менял)
# =========================
def loop_subtypes_for_case(sub: pd.DataFrame, use_res: bool) -> Dict[str, int]:
    acts = sub["activity"].tolist()
    res  = sub["resource"].tolist() if use_res else None
    if not acts:
        return {k: 0 for k in
                ["self_loops","ping_pong","ping_pong_res","returns_nonadj","jump_to_prev_any","back_to_start","loop_score_advanced"]}
    self_loops = sum(1 for i in range(len(acts)-1) if acts[i] == acts[i+1])
    # ping-pong ABAB
    ping_pong = 0; i = 0
    while i+3 < len(acts):
        a,b,c,d = acts[i:i+4]
        if a != b and a == c and b == d:
            ping_pong += 1; i += 2
        else:
            i += 1
    # ресурсный пинг-понг A@R1↔A@R2
    ping_pong_res = 0
    if use_res:
        i = 0
        while i+3 < len(acts):
            a1,r1 = acts[i],res[i]; a2,r2 = acts[i+1],res[i+1]; a3,r3 = acts[i+2],res[i+2]; a4,r4 = acts[i+3],res[i+3]
            if a1 == a2 == a3 == a4 and len({r1,r2}) == 2 and r1 == r3 and r2 == r4:
                ping_pong_res += 1; i += 2
            else:
                i += 1
    returns_nonadj = 0; last_pos = {}
    for j,a in enumerate(acts):
        if a in last_pos and j-last_pos[a] > 1:
            returns_nonadj += 1
        last_pos[a] = j
    jump_to_prev_any = 0; seen = set()
    for j in range(len(acts)-1):
        seen.add(acts[j])
        if acts[j+1] in seen and acts[j+1] != acts[j]:
            jump_to_prev_any += 1
    start_act = acts[0]
    back_to_start = sum(1 for j in range(1,len(acts)) if acts[j] == start_act)
    score = self_loops + ping_pong + ping_pong_res + returns_nonadj + jump_to_prev_any + back_to_start
    return dict(self_loops=self_loops,ping_pong=ping_pong,ping_pong_res=ping_pong_res,
                returns_nonadj=returns_nonadj,jump_to_prev_any=jump_to_prev_any,back_to_start=back_to_start,
                loop_score_advanced=score)

loop_rows = []
for cid,g in df.groupby("case_id"):
    sc = loop_subtypes_for_case(g, use_resource); sc["case_id"] = cid; loop_rows.append(sc)
loops_df = pd.DataFrame(loop_rows)

def q75(s): return int(np.ceil(safe_pct(s, 75, default=1)))
thr_self = q75(loops_df["self_loops"]); thr_pp = q75(loops_df["ping_pong"])
thr_ppr = q75(loops_df["ping_pong_res"]) if use_resource else 1
thr_ret = q75(loops_df["returns_nonadj"]); thr_jump = q75(loops_df["jump_to_prev_any"])
thr_start = q75(loops_df["back_to_start"]); thr_loop_total = q75(loops_df["loop_score_advanced"])

def per_case_overruns(sub_edges: pd.DataFrame):
    single_spike = 0; overrun_sum = 0.0; entry_waits = {}
    for _,row in sub_edges.iterrows():
        e = (row["activity"],row["next_activity"]); d = float(row["delta_sec"])
        med = edge_median_map.get(e, np.nan); p99 = edge_p99_map.get(e, np.nan)
        if not np.isnan(med): overrun_sum += max(0.0, d-med)
        if not np.isnan(p99) and d > p99: single_spike += 1
        entry_waits[row["next_activity"]] = entry_waits.get(row["next_activity"],0.0) + d
    total_wait = sum(entry_waits.values()); queue_flag=False; qtarget=None
    if total_wait>0:
        best = max(entry_waits,key=entry_waits.get)
        if entry_waits[best]/total_wait >= 0.40: queue_flag=True; qtarget=best
    return single_spike, overrun_sum, queue_flag, qtarget

over_rows = []
for cid,g in df_sorted.groupby("case_id"):
    sub_e = g.dropna(subset=["next_activity","delta_sec"])
    ss,osum,qf,qt = per_case_overruns(sub_e)
    over_rows.append({"case_id":cid,"single_spike_cnt":ss,"overrun_sum_sec":osum,"queue_before_flag":qf,"queue_target":qt})
over_df = pd.DataFrame(over_rows)
thr_over_sum = float(np.ceil(safe_pct(over_df["overrun_sum_sec"],90,default=0.0)))
thr_over_spike = max(1,int(np.ceil(safe_pct(over_df["single_spike_cnt"],75,default=1))))

k_bottlenecks = st.sidebar.number_input("Top-k узких рёбер (для влияния)", min_value=1, value=10, step=1)
top_edges = set(edge_stats.sort_values("median",ascending=False).head(k_bottlenecks)["edge"].tolist())

def impact_for_case(sub_edges: pd.DataFrame):
    if sub_edges.empty: return 0.0,0.0,0
    imp_sum=0.0; in_b=0; excnt=0; n=0
    for _,row in sub_edges.iterrows():
        e=(row["activity"],row["next_activity"]); d=float(row["delta_sec"]); p95 = edge_p95_map.get(e,np.nan)
        n+=1
        if e in top_edges: in_b += 1
        if not np.isnan(p95):
            if d>p95:
                imp_sum += (d-p95); excnt += 1
    share = in_b/n if n else 0.0
    return imp_sum, share, excnt

impact_rows=[]
for cid,g in df_sorted.groupby("case_id"):
    sub_e = g.dropna(subset=["next_activity","delta_sec"])
    isum,share,excnt = impact_for_case(sub_e)
    impact_rows.append({"case_id":cid,"impact_sum_sec":isum,"bneck_share":share,"p95_exceed_cnt":excnt})
impact_df = pd.DataFrame(impact_rows)
thr_impact_sum = float(np.ceil(safe_pct(impact_df["impact_sum_sec"],90,default=0.0)))
thr_bneck_share = float(np.round(safe_pct(impact_df["bneck_share"],90,default=0.5),2))
thr_exceed_cnt = int(np.ceil(safe_pct(impact_df["p95_exceed_cnt"],75,default=1)))

# =========================
# Флаги и короткий вывод (как раньше) — опустим ради краткости
# (оставь свою предыдущую часть с отображением результатов)
# =========================

# =========================
# DFG: вертикальная схема + экспорт PNG
# =========================
with st.expander("📌 Карта процесса (DFG) и экспорт", expanded=True):
    dfg_mode = st.radio("Метрика карты", ["Frequency", "Performance"], horizontal=True, key="dfg_mode")
    if dfg_mode == "Frequency":
        dfg, sa, ea = pm4py.discover_dfg(event_log); variant = dfg_visualizer.Variants.FREQUENCY
    else:
        dfg, sa, ea = pm4py.discover_performance_dfg(event_log); variant = dfg_visualizer.Variants.PERFORMANCE

    # Настройки ориентации/плотности
    st.caption("Расположение графа")
    c1,c2,c3,c4 = st.columns(4)
    with c1:
        rankdir = st.selectbox("Ориентация", ["TB (сверху вниз)","LR (слева направо)"], index=0)
    with c2:
        ranksep = st.slider("ranksep", 0.1, 3.0, 0.6, 0.1)
    with c3:
        nodesep = st.slider("nodesep", 0.05, 2.0, 0.2, 0.05)
    with c4:
        ratio = st.selectbox("ratio", ["compress","fill","auto"], index=0)

    params = {"start_activities": sa, "end_activities": ea}
    gviz = dfg_visualizer.apply(dfg, log=event_log, variant=variant, parameters=params)

    # ⚙️ важное: делаем граф ВЫСОКИМ, а не длинным
    # graphviz.Digraph API:
    gviz.graph_attr.update(rankdir=("TB" if rankdir.startswith("TB") else "LR"),
                           ranksep=str(ranksep), nodesep=str(nodesep), ratio=ratio)

    # Умеренно ограничим ширину, увеличим высоту (для embed)
    # Примечание: st.graphviz_chart сам масштабирует, но вертикальная ориентация уже решает «простыню»
    st.graphviz_chart(gviz.source, use_container_width=True)

    # Кнопки выгрузки
    st.download_button("⬇️ DOT", gviz.source.encode("utf-8"), file_name="process_dfg.dot", mime="text/plain")

    has_dot = shutil.which("dot") is not None
    if has_dot:
        try:
            with tempfile.TemporaryDirectory() as tmpd:
                outpath = os.path.join(tmpd, "process_dfg")
                # Пытаемся запечь те же graph_attr и в файл
                gviz.render(filename=outpath, format="png", cleanup=True)
                with open(outpath + ".png", "rb") as f:
                    st.download_button("⬇️ PNG (graphviz)", f, file_name="process_dfg.png", mime="image/png")
        except Exception as e:
            st.warning(f"PNG через graphviz не удалось: {e}")

    # Fallback: собственный вертикальный PNG без graphviz
    if not has_dot:
        st.info("Graphviz ('dot') не найден — рисую вертикальный PNG (fallback).")
        try:
            # Готовим граф
            G = nx.DiGraph()
            for (u, v), w in dfg.items():
                G.add_edge(str(u), str(v), weight=w)

            # Разложим граф по слоям топологически (примерно), чтобы был вертикальный поток
            try:
                layers = list(nx.algorithms.dag.topological_generations(G))
            except Exception:
                layers = [list(G.nodes())]
            pos = {}
            y = 0
            for layer in layers:
                for i, n in enumerate(layer):
                    pos[n] = (i, -y)
                y += 1

            fig, ax = plt.subplots(figsize=(8, 14))  # высокий рисунок
            nx.draw_networkx_nodes(G, pos, node_size=1200, ax=ax)
            nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)
            nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', ax=ax)
            edge_labels = {(u, v): f"{data.get('weight', '')}" for u, v, data in G.edges(data=True)}
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)
            ax.axis('off')
            buf = io.BytesIO()
            plt.tight_layout()
            fig.savefig(buf, format="png", dpi=200)
            buf.seek(0)
            st.download_button("⬇️ PNG (fallback, вертикальный)", buf, file_name="process_dfg_fallback_vertical.png", mime="image/png")
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Фоллбэк PNG не удался: {e}. Установи `graphviz` для идеального экспорта.")
