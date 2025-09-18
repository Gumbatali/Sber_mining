# streamlit_app.py
# Process Mining — авто-поиск неэффективностей + DFG PNG с фоллбэком, если нет system graphviz

import os
import io
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

import pm4py
from pm4py.objects.log.util import dataframe_utils
from pm4py.statistics.traces.generic.log import case_statistics
from pm4py.visualization.dfg import visualizer as dfg_visualizer

import matplotlib.pyplot as plt
import shutil  # для проверки наличия 'dot'
import networkx as nx  # фоллбэк-рендер DFG без graphviz

st.set_page_config(page_title="Process Mining — QA неэффективностей", layout="wide")
st.title("🔍 Process Mining — авто-поиск неэффективностей")
st.caption("Все кейсы анализируются в фоне. На экран выводятся только факты наличия неэффективностей и короткие примеры. "
           "DFG PNG: system graphviz → красивый рендер; без него → упрощённый фоллбэк.")

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
    st.info("⬆️ Загрузите лог или укажите путь к файлу.")
    st.stop()

st.subheader("📊 Предпросмотр")
st.dataframe(df.head(), use_container_width=True)

# =========================
# Маппинг + время
# =========================
st.subheader("🧭 Маппинг колонок")
cols = df.columns.tolist()
col_case = st.selectbox("Колонка кейса", cols, index=0)
col_act  = st.selectbox("Колонка активности", cols, index=min(1, len(cols)-1))
col_ts   = st.selectbox("Колонка времени", cols, index=min(2, len(cols)-1))

with st.expander("🕒 Парсинг времени", expanded=False):
    date_hint = st.text_input("Формат даты (опц.) например %Y-%m-%d %H:%M:%S", value="")
    coerce_ts = st.checkbox("Ошибочные даты → NaT", value=True)
    tz = st.text_input("Таймзона (например Europe/Moscow). Пусто — не менять", value="")

df = df[[col_case, col_act, col_ts]].copy()
df.columns = ["case_id", "activity", "timestamp"]

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
    st.warning(f"⚠️ {na_share:.1%} значений времени → NaT (будут отброшены).")
    df = df.dropna(subset=["timestamp"])

df["case_id"] = df["case_id"].astype(str).str.strip()
df["activity"] = df["activity"].astype(str).str.strip()
df = df[(df["case_id"] != "") & (df["activity"] != "")]
df = df.sort_values(["case_id", "timestamp"])

events_per_case = df.groupby("case_id").size()
if df.empty or events_per_case.max() < 2:
    st.error("Нужно ≥2 события на кейс.")
    st.stop()

# =========================
# Производные метрики
# =========================
# PM4Py event log
event_log = pm4py.format_dataframe(df, case_id="case_id", activity_key="activity", timestamp_key="timestamp")
event_log = dataframe_utils.convert_timestamp_columns_in_df(event_log)
if "time:timestamp" in event_log.columns:
    event_log = event_log.sort_values(["case:concept:name", "time:timestamp"])
else:
    event_log = event_log.sort_values(["case:concept:name"])

n_cases = event_log["case:concept:name"].nunique()
st.success(f"✅ Событий: {len(event_log):,} • Кейсов: {n_cases:,}")

# Длительности кейсов
case_dur_df = (
    event_log.groupby("case:concept:name")["time:timestamp"]
             .agg(start="min", end="max")
             .assign(duration_sec=lambda x: (x["end"] - x["start"]).dt.total_seconds())
             .rename_axis("case_id")
             .reset_index()
)

# Межшаговые интервалы
df_sorted = df.copy()
df_sorted["next_activity"] = df_sorted.groupby("case_id")["activity"].shift(-1)
df_sorted["next_timestamp"] = df_sorted.groupby("case_id")["timestamp"].shift(-1)
df_sorted["delta_sec"] = (df_sorted["next_timestamp"] - df_sorted["timestamp"]).dt.total_seconds()
edges = df_sorted.dropna(subset=["next_activity", "delta_sec"]).copy()
edges["edge"] = list(zip(edges["activity"], edges["next_activity"]))

# Глобальные статистики рёбер
edge_stats = (
    edges.groupby("edge")["delta_sec"]
         .agg(median="median",
              p90=lambda s: np.percentile(s, 90),
              p95=lambda s: np.percentile(s, 95),
              mean="mean",
              std="std",
              count="count")
         .reset_index()
)
edge_stats_dict = edge_stats.set_index("edge").to_dict(orient="index")
edge_median_map = {e: s["median"] for e, s in edge_stats_dict.items()}
edge_p95_map = {e: s["p95"] for e, s in edge_stats_dict.items()}

def safe_pct(series, q, default=np.nan) -> float:
    s = pd.Series(series).dropna()
    return float(np.percentile(s, q)) if not s.empty else default

# =========================
# Индикаторы (из презентации): Зацикленность • Длительность операции • Влияние
# =========================
def loop_scores_for_case(sub: pd.DataFrame) -> Dict[str, int]:
    acts = sub["activity"].tolist()
    if not acts:
        return {"rework_total": 0, "adjacent_loops": 0, "cycle_returns": 0, "ab_cycle": 0, "loop_score": 0}
    uniq, cnt = np.unique(acts, return_counts=True)
    rework_total = int(np.sum(cnt[cnt > 1] - 1))
    adjacent_loops = sum(1 for i in range(len(acts)-1) if acts[i] == acts[i+1])
    last_pos, cycle_returns = {}, 0
    for i, a in enumerate(acts):
        if a in last_pos and i - last_pos[a] > 1:
            cycle_returns += 1
        last_pos[a] = i
    ab_cycle = 0
    for i in range(len(acts)-3):
        if acts[i] != acts[i+1] and acts[i] == acts[i+2] and acts[i+1] == acts[i+3]:
            ab_cycle += 1
    loop_score = rework_total + adjacent_loops + cycle_returns + ab_cycle
    return dict(rework_total=rework_total, adjacent_loops=adjacent_loops,
                cycle_returns=cycle_returns, ab_cycle=ab_cycle, loop_score=loop_score)

loop_rows = []
for cid, g in df.groupby("case_id"):
    sc = loop_scores_for_case(g)
    sc["case_id"] = cid
    loop_rows.append(sc)
loops_df = pd.DataFrame(loop_rows)
auto_loop_thr = int(np.ceil(safe_pct(loops_df["loop_score"], 75, default=1)))

def overrun_for_case(sub_edges: pd.DataFrame) -> float:
    over = 0.0
    for _, row in sub_edges.iterrows():
        edge = (row["activity"], row["next_activity"])
        d = float(row["delta_sec"])
        med = edge_median_map.get(edge, np.nan)
        if not np.isnan(med):
            over += max(0.0, d - float(med))
    return over

over_rows = []
for cid, g in df_sorted.groupby("case_id"):
    sub_edges = g.dropna(subset=["next_activity", "delta_sec"])
    over = overrun_for_case(sub_edges)
    over_rows.append({"case_id": cid, "overrun_sum_sec": over})
over_df = pd.DataFrame(over_rows)
auto_over_thr = float(np.ceil(safe_pct(over_df["overrun_sum_sec"], 90, default=0.0)))

k_bottlenecks = st.sidebar.number_input("Top-k узких рёбер для метрики влияния", min_value=1, value=10, step=1)
top_edges = edge_stats.sort_values("median", ascending=False).head(k_bottlenecks)["edge"].tolist()
top_edge_set = set(top_edges)

def impact_for_case(sub_edges: pd.DataFrame) -> Tuple[float, float, int]:
    if sub_edges.empty:
        return 0.0, 0.0, 0
    impact_sum = 0.0
    in_bneck = 0
    n_edges = 0
    for _, row in sub_edges.iterrows():
        edge = (row["activity"], row["next_activity"])
        d = float(row["delta_sec"])
        p95 = edge_p95_map.get(edge, np.nan)
        n_edges += 1
        if edge in top_edge_set:
            in_bneck += 1
        if not np.isnan(p95):
            impact_sum += max(0.0, d - float(p95))
    frac_bneck = (in_bneck / n_edges) if n_edges else 0.0
    return impact_sum, frac_bneck, n_edges

impact_rows = []
for cid, g in df_sorted.groupby("case_id"):
    sub_edges = g.dropna(subset=["next_activity", "delta_sec"])
    imp_sum, frac_bneck, n_edges = impact_for_case(sub_edges)
    impact_rows.append({"case_id": cid, "impact_sum_sec": imp_sum, "frac_bottleneck_edges": frac_bneck, "n_edges": n_edges})
impact_df = pd.DataFrame(impact_rows)
auto_impact_thr = float(np.ceil(safe_pct(impact_df["impact_sum_sec"], 90, default=0.0)))
auto_frac_bneck_thr = float(np.round(safe_pct(impact_df["frac_bottleneck_edges"], 90, default=0.5), 2))

# =========================
# Режим: Авто / Ручной
# =========================
st.subheader("🧠 Пороговые значения")
mode = st.radio("Режим порогов", ["Авто", "Ручной"], horizontal=True, index=0)
if mode == "Авто":
    loop_thr = auto_loop_thr
    over_thr = auto_over_thr
    impact_thr = auto_impact_thr
    frac_bneck_thr = auto_frac_bneck_thr
else:
    c1, c2 = st.columns(2)
    with c1:
        loop_thr = st.number_input("Порог зацикленности (loop_score ≥)", min_value=0, value=int(auto_loop_thr), step=1)
        over_thr = st.number_input("Порог длительности операции (Σ overrun, сек ≥)", min_value=0, value=int(auto_over_thr), step=10)
    with c2:
        impact_thr = st.number_input("Порог влияния (impact_sum, сек ≥)", min_value=0, value=int(auto_impact_thr), step=10)
        frac_bneck_thr = st.number_input("Доля узких рёбер (≥)", min_value=0.0, max_value=1.0,
                                         value=float(auto_frac_bneck_thr), step=0.05)

max_show = st.slider("Сколько примеров-«доказательств» показывать", 1, 20, 5)

# =========================
# Флаги (бэкенд)
# =========================
loops_merge = loops_df.copy()
loops_merge["flag_loop"] = loops_merge["loop_score"] >= loop_thr

over_merge = over_df.copy()
over_merge["flag_over"] = over_merge["overrun_sum_sec"] >= over_thr

impact_merge = impact_df.copy()
impact_merge["flag_impact"] = (impact_merge["impact_sum_sec"] >= impact_thr) | \
                              (impact_merge["frac_bottleneck_edges"] >= frac_bneck_thr)

summary = (loops_merge[["case_id", "loop_score", "flag_loop"]]
           .merge(over_merge[["case_id", "overrun_sum_sec", "flag_over"]], on="case_id", how="outer")
           .merge(impact_merge[["case_id", "impact_sum_sec", "frac_bottleneck_edges", "flag_impact"]],
                  on="case_id", how="outer"))
summary["any_flag"] = summary[["flag_loop", "flag_over", "flag_impact"]].fillna(False).any(axis=1)

# =========================
# Вывод (коротко)
# =========================
st.header("🧪 Результаты по индикаторам")

def download_df_button(df_to_dl: pd.DataFrame, filename: str, label: str):
    csv = df_to_dl.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, file_name=filename, mime="text/csv")

# 1) Зацикленность
st.subheader("1) Зацикленность (повторы/циклы)")
bad_loops = loops_merge[loops_merge["flag_loop"]].copy()
count_bad = int(bad_loops.shape[0])
st.write(f"Наличие неэффективности: **{'Да' if count_bad>0 else 'Нет'}**  • Затронуто кейсов: **{count_bad}** из {n_cases}")
if count_bad > 0:
    show = bad_loops.sort_values("loop_score", ascending=False).head(max_show)
    st.markdown("**Показательные кейсы:**")
    for _, r in show.iterrows():
        st.markdown(f"- **{r['case_id']}** — loop_score={int(r['loop_score'])} "
                    f"(rework={int(r['rework_total'])}, A→A={int(r['adjacent_loops'])}, A…A={int(r['cycle_returns'])}, AB…AB={int(r['ab_cycle'])})")
    with st.expander("⬇️ Все кейсы с зацикленностью"):
        download_df_button(bad_loops.sort_values("loop_score", ascending=False),
                           "ineff_loops_cases.csv", "Скачать CSV")

# 2) Длительность операции
st.subheader("2) Длительность операции (перерасход времени)")
bad_over = over_merge[over_merge["flag_over"]].copy()
count_bad_over = int(bad_over.shape[0])
st.write(f"Наличие неэффективности: **{'Да' if count_bad_over>0 else 'Нет'}**  • Затронуто кейсов: **{count_bad_over}** из {n_cases}")
if count_bad_over > 0:
    show = bad_over.sort_values("overrun_sum_sec", ascending=False).head(max_show)
    st.markdown("**Показательные кейсы:**")
    for _, r in show.iterrows():
        st.markdown(f"- **{r['case_id']}** — Σ overrun = {int(r['overrun_sum_sec'])} сек")
    with st.expander("⬇️ Все кейсы с перерасходом"):
        download_df_button(bad_over.sort_values("overrun_sum_sec", ascending=False),
                           "ineff_duration_cases.csv", "Скачать CSV")

# 3) Влияние на процесс
st.subheader("3) Влияние на процесс (участие в узких местах)")
bad_impact = impact_merge[impact_merge["flag_impact"]].copy()
count_bad_impact = int(bad_impact.shape[0])
st.write(f"Наличие неэффективности: **{'Да' if count_bad_impact>0 else 'Нет'}**  • Затронуто кейсов: **{count_bad_impact}** из {n_cases}")
if count_bad_impact > 0:
    show = bad_impact.sort_values(["impact_sum_sec", "frac_bottleneck_edges"], ascending=False).head(max_show)
    st.markdown("**Показательные кейсы:**")
    for _, r in show.iterrows():
        st.markdown(f"- **{r['case_id']}** — impact_sum={int(r['impact_sum_sec'])} сек, доля узких рёбер={r['frac_bottleneck_edges']:.2f}")
    with st.expander("⬇️ Все кейсы с влиянием"):
        download_df_button(bad_impact.sort_values(["impact_sum_sec", "frac_bottleneck_edges"], ascending=False),
                           "ineff_impact_cases.csv", "Скачать CSV")

st.header("📋 Сводка фактов")
total_any = int(summary["any_flag"].sum())
st.write(
    f"- Зацикленность: **{count_bad}** кейсов.\n"
    f"- Длительность операций: **{count_bad_over}** кейсов.\n"
    f"- Влияние на процесс: **{count_bad_impact}** кейсов.\n"
    f"- Любая неэффективность: **{total_any}** из {n_cases}."
)

# =========================
# DFG + PNG: с фоллбэком
# =========================
with st.expander("📌 Карта процесса (DFG) и экспорт PNG", expanded=True):
    dfg_mode = st.radio("Метрика карты", ["Frequency", "Performance"], horizontal=True, key="dfg_mode")
    if dfg_mode == "Frequency":
        dfg, sa, ea = pm4py.discover_dfg(event_log)
        variant = dfg_visualizer.Variants.FREQUENCY
    else:
        dfg, sa, ea = pm4py.discover_performance_dfg(event_log)
        variant = dfg_visualizer.Variants.PERFORMANCE

    params = {"start_activities": sa, "end_activities": ea}
    gviz = dfg_visualizer.apply(dfg, log=event_log, variant=variant, parameters=params)

    st.graphviz_chart(gviz.source, use_container_width=True)

    has_dot = shutil.which("dot") is not None

    # 1) Кнопка DOT (всегда доступна)
    st.download_button("⬇️ DOT", gviz.source.encode("utf-8"), file_name="process_dfg.dot", mime="text/plain")

    # 2) Красивый PNG через graphviz, если есть 'dot'
    if has_dot:
        try:
            with tempfile.TemporaryDirectory() as tmpd:
                outpath = os.path.join(tmpd, "process_dfg")
                gviz.render(filename=outpath, format="png", cleanup=True)
                with open(outpath + ".png", "rb") as f:
                    st.download_button("⬇️ PNG (graphviz)", f, file_name="process_dfg.png", mime="image/png")
        except Exception as e:
            st.warning(f"PNG через graphviz не удалось: {e}")

    # 3) Фоллбэк PNG без graphviz — рисуем DFG сами через networkx/matplotlib
    if not has_dot:
        st.info("Graphviz ('dot') не найден. Использую упрощённый рендер PNG без graphviz.")
        try:
            # Собираем граф
            G = nx.DiGraph()
            # Узлы: старт/финиш + активности
            for a in set(list(dfg.keys()) + list(sa.keys()) + list(ea.keys())):
                if isinstance(a, tuple):  # защита от неожиданных ключей
                    continue
                G.add_node(a)
            # Рёбра с весами
            for (u, v), w in dfg.items():
                G.add_edge(u, v, weight=w)

            # Layout
            pos = nx.spring_layout(G, k=1.2, seed=42)
            fig, ax = plt.subplots(figsize=(12, 6))
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
            st.download_button("⬇️ PNG (fallback)", buf, file_name="process_dfg_fallback.png", mime="image/png")
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Фоллбэк PNG тоже не удался: {e}\nПопробуйте установить system graphviz.")

st.success("Готово.")
