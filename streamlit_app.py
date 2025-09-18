# streamlit_app.py
# Финальный “крутой” код: Bottle neck (по слайду) + полный блок Зацикленности (все подпункты)
# + DFG (вертикально) с корректным PNG-экспортом (graphviz / fallback networkx)

import os
import io
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

# ───────────────────────────────
# pm4py/graphviz — опционально
# ───────────────────────────────
HAS_PM4PY = True
try:
    import pm4py
    from pm4py.objects.log.util import dataframe_utils as pm_df_utils
    from pm4py.visualization.dfg import visualizer as dfg_visualizer
except Exception:
    HAS_PM4PY = False

import shutil
import networkx as nx

st.set_page_config(page_title="Process Mining — Bottlenecks & Loops", layout="wide")
st.title("🚦 Bottle neck + 🔄 Зацикленность (подтипы) + 📌 DFG")

# =========================================================
# 0) ХЕЛПЕРЫ
# =========================================================
def fmt_time(sec: float) -> str:
    if pd.isna(sec): return ""
    sec = float(sec)
    if sec < 120: return f"{int(round(sec))} с"
    m = sec / 60
    if m < 120: return f"{m:.1f} мин"
    h = m / 60
    if h < 48: return f"{h:.1f} ч"
    d = h / 24
    return f"{d:.1f} д"

def safe_percentile(s, q, default=np.nan):
    s = pd.Series(s).dropna()
    return float(np.percentile(s, q)) if len(s) else default

def download_df(df: pd.DataFrame, name: str, label: str):
    st.download_button(label, df.to_csv(index=False).encode("utf-8"),
                       file_name=name, mime="text/csv")

# =========================================================
# 1) ЗАГРУЗКА
# =========================================================
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

# =========================================================
# 2) МАППИНГ
# =========================================================
st.subheader("🧭 Маппинг колонок")
cols = df.columns.tolist()
case_col = st.selectbox("Колонка кейса", cols, index=0)
act_col  = st.selectbox("Колонка активности", cols, index=min(1, len(cols)-1))
ts_col   = st.selectbox("Колонка времени", cols, index=min(2, len(cols)-1))

with st.expander("🕒 Парсинг времени", expanded=False):
    fmt = st.text_input("Формат даты (опц.), напр. %Y-%m-%d %H:%M:%S", value="")
    coerce = st.checkbox("Ошибочные даты → NaT", value=True)

work = df[[case_col, act_col, ts_col]].copy()
work.columns = ["case_id", "activity", "timestamp"]

if fmt.strip():
    work["timestamp"] = pd.to_datetime(work["timestamp"], format=fmt, errors=("coerce" if coerce else "raise"))
else:
    work["timestamp"] = pd.to_datetime(work["timestamp"], errors=("coerce" if coerce else "raise"))

work = work.dropna(subset=["timestamp"])
work["case_id"] = work["case_id"].astype(str).str.strip()
work["activity"] = work["activity"].astype(str).str.strip()
work = work[(work["case_id"] != "") & (work["activity"] != "")]
work = work.sort_values(["case_id", "timestamp"])

if work.groupby("case_id").size().max() < 2:
    st.error("Нужно ≥2 события на кейс.")
    st.stop()

# =========================================================
# 3) BOTTLE NECK (по слайду)
# =========================================================
st.header("🔎 Bottle neck — идентификация (по презентации)")

# прокси-длительность операции = Δ к следующему событию
work["next_ts"] = work.groupby("case_id")["timestamp"].shift(-1)
work["delta_sec"] = (work["next_ts"] - work["timestamp"]).dt.total_seconds()
ev = work.dropna(subset=["delta_sec"]).copy()

ops = (
    ev.groupby("activity")["delta_sec"]
      .agg(count="count", mean_dur="mean", median_dur="median", std_dur="std")
      .reset_index()
)

pctl = st.slider("Перцентиль «долгих» операций (по mean_dur)", 50, 99, 90)
p_long = safe_percentile(ops["mean_dur"], pctl, default=np.inf)
ops["is_long"] = ops["mean_dur"] >= p_long
ops["mm_ratio"] = ops["mean_dur"] / ops["median_dur"]
ops["cond_mm"] = (ops["mm_ratio"] > 0.9) & (ops["mm_ratio"] < 1.1)
mean_std_all = ops["std_dur"].replace([np.inf, -np.inf], np.nan).dropna().mean()
ops["norm_std"] = ops["std_dur"] / (mean_std_all if mean_std_all and not np.isnan(mean_std_all) else 1.0)
ops["cond_std"] = ops["norm_std"] < 0.5
ops["is_bottleneck"] = ops["is_long"] & ops["cond_mm"] & ops["cond_std"]

ops_show = ops.copy()
for c in ["mean_dur", "median_dur", "std_dur"]:
    ops_show[c] = ops_show[c].apply(fmt_time)

st.dataframe(
    ops_show[["activity","count","mean_dur","median_dur","std_dur","mm_ratio","norm_std","is_bottleneck"]],
    use_container_width=True
)

bn_ops = ops[ops["is_bottleneck"]].sort_values("mean_dur", ascending=False)
st.success(f"Найдено узких операций: **{bn_ops.shape[0]}** из {ops.shape[0]}")

download_df(ops, "bottleneck_operations.csv", "⬇️ CSV — метрики по операциям")

# Оценка эффекта
st.subheader("💰 Потенциальный эффект")
S = st.number_input("Стоимость минуты процесса S", min_value=0.0, value=10.0, step=1.0)
currency = st.text_input("Валюта/единица", value="₽")

mean_map = ops.set_index("activity")["mean_dur"].to_dict()
ev["over_mean_sec"] = ev.apply(
    lambda r: max(0.0, float(r["delta_sec"]) - float(mean_map.get(r["activity"], np.nan)))
    if pd.notna(mean_map.get(r["activity"], np.nan)) else np.nan, axis=1)
ev_eff = ev.dropna(subset=["over_mean_sec"]).copy()
ev_eff["over_mean_min"] = ev_eff["over_mean_sec"] / 60.0
ev_eff["cost"] = ev_eff["over_mean_min"] * S

eff_by_act = (ev_eff.groupby("activity")
              .agg(appearances=("activity","count"),
                   over_minutes=("over_mean_min","sum"),
                   cost=("cost","sum"))
              .reset_index().sort_values("cost", ascending=False))
st.write(f"Итоговый потенциальный эффект: **{eff_by_act['cost'].sum():,.0f} {currency}**")
st.dataframe(eff_by_act.assign(over_minutes=lambda d: d["over_minutes"].round(1),
                               cost=lambda d: d["cost"].round(0)), use_container_width=True)
download_df(eff_by_act, "bottleneck_effect_by_operation.csv", "⬇️ CSV — эффект по операциям")

# =========================================================
# 4) ЗАЦИКЛЕННОСТЬ — функции (каждая отдельно)
# =========================================================
st.header("🔄 Зацикленность — все подтипы, покейсно")

def prepare_event_order(df_in: pd.DataFrame) -> pd.DataFrame:
    return (
        df_in.sort_values(["case_id","timestamp"])
             .assign(pos=lambda d: d.groupby("case_id").cumcount())
             [["case_id","activity","timestamp","pos"]]
    )

def loops_self(df_sorted: pd.DataFrame) -> pd.DataFrame:
    res=[]
    for cid,g in df_sorted.groupby("case_id", sort=False):
        acts=g["activity"].tolist(); poss=g["pos"].tolist()
        cnt=0; ex=[]
        for i in range(len(acts)-1):
            if acts[i]==acts[i+1]:
                cnt+=1
                if len(ex)<10: ex.append((poss[i], acts[i]))
        res.append({"case_id":cid,"self_loops":cnt,"self_examples":ex})
    return pd.DataFrame(res)

def loops_return(df_sorted: pd.DataFrame, min_gap:int=2)->pd.DataFrame:
    res=[]
    for cid,g in df_sorted.groupby("case_id", sort=False):
        acts=g["activity"].tolist(); poss=g["pos"].tolist()
        last={}; cnt=0; ex=[]
        for i,a in enumerate(acts):
            if a in last and (i-last[a])>=min_gap:
                cnt+=1
                if len(ex)<10: ex.append((a, poss[last[a]], poss[i]))
            last[a]=i
        res.append({"case_id":cid,"returns_nonadj":cnt,"return_examples":ex})
    return pd.DataFrame(res)

def loops_pingpong(df_sorted: pd.DataFrame, allow_overlap_halfstep:bool=True)->pd.DataFrame:
    res=[]
    for cid,g in df_sorted.groupby("case_id", sort=False):
        acts=g["activity"].tolist(); poss=g["pos"].tolist()
        i=0; cnt=0; ex=[]
        while i+3<len(acts):
            a,b,c,d=acts[i:i+4]
            if a!=b and a==c and b==d:
                cnt+=1
                if len(ex)<10: ex.append((poss[i], a, b))
                i+=2 if allow_overlap_halfstep else 4
            else:
                i+=1
        res.append({"case_id":cid,"ping_pong":cnt,"pingpong_examples":ex})
    return pd.DataFrame(res)

def loops_back_to_start(df_sorted: pd.DataFrame)->pd.DataFrame:
    res=[]
    for cid,g in df_sorted.groupby("case_id", sort=False):
        acts=g["activity"].tolist(); poss=g["pos"].tolist()
        if not acts:
            res.append({"case_id":cid,"back_to_start":0,"start_examples":[]}); continue
        start=acts[0]; cnt=0; ex=[]
        for i in range(1,len(acts)):
            if acts[i]==start:
                cnt+=1
                if len(ex)<10: ex.append(poss[i])
        res.append({"case_id":cid,"back_to_start":cnt,"start_examples":ex})
    return pd.DataFrame(res)

def loops_backjump(df_sorted: pd.DataFrame)->pd.DataFrame:
    res=[]
    for cid,g in df_sorted.groupby("case_id", sort=False):
        acts=g["activity"].tolist(); poss=g["pos"].tolist()
        if not acts:
            res.append({"case_id":cid,"jump_to_prev_any":0,"backjump_examples":[]}); continue
        seen={acts[0]}; cnt=0; ex=[]
        for i in range(len(acts)-1):
            cur,nxt=acts[i],acts[i+1]
            if nxt in seen and nxt!=cur:
                cnt+=1
                if len(ex)<10: ex.append((poss[i], poss[i+1], nxt))
            seen.add(cur)
        res.append({"case_id":cid,"jump_to_prev_any":cnt,"backjump_examples":ex})
    return pd.DataFrame(res)

def compute_all_loops(df_in: pd.DataFrame) -> pd.DataFrame:
    base = prepare_event_order(df_in)
    a = loops_self(base)
    b = loops_return(base)
    c = loops_pingpong(base)
    d = loops_back_to_start(base)
    e = loops_backjump(base)
    out = (a.merge(b, on="case_id", how="outer")
             .merge(c, on="case_id", how="outer")
             .merge(d, on="case_id", how="outer")
             .merge(e, on="case_id", how="outer"))
    for col in ["self_loops","returns_nonadj","ping_pong","back_to_start","jump_to_prev_any"]:
        out[col] = out[col].fillna(0).astype(int)
    for col in ["self_examples","return_examples","pingpong_examples","start_examples","backjump_examples"]:
        out[col] = out[col].apply(lambda x: x if isinstance(x, list) else [])
    out["loop_score"] = out["self_loops"] + out["returns_nonadj"] + out["ping_pong"] + out["back_to_start"] + out["jump_to_prev_any"]
    return out

# считаем по всем кейсам
loops_df = compute_all_loops(work[["case_id","activity","timestamp"]])

# автопороги = q75 по распределениям
def q75_int(s): 
    s = pd.Series(s).fillna(0)
    return int(np.ceil(np.percentile(s, 75))) if len(s) else 1

thr_self  = q75_int(loops_df["self_loops"])
thr_ret   = q75_int(loops_df["returns_nonadj"])
thr_pp    = q75_int(loops_df["ping_pong"])
thr_start = q75_int(loops_df["back_to_start"])
thr_jump  = q75_int(loops_df["jump_to_prev_any"])
thr_sum   = q75_int(loops_df["loop_score"])

with st.expander("🧠 Пороговые значения (авто q75, можно править)", expanded=False):
    c1,c2,c3 = st.columns(3)
    with c1:
        thr_self  = st.number_input("В себя (A→A) ≥", 0, value=int(thr_self))
        thr_ret   = st.number_input("Возврат (A…A) ≥", 0, value=int(thr_ret))
    with c2:
        thr_pp    = st.number_input("Пинг-понг (ABAB) ≥", 0, value=int(thr_pp))
        thr_start = st.number_input("В начало ≥", 0, value=int(thr_start))
    with c3:
        thr_jump  = st.number_input("В произвольный этап ≥", 0, value=int(thr_jump))
        thr_sum   = st.number_input("Суммарный loop score ≥", 0, value=int(thr_sum))

# флаги
loops_df["flag_self"]  = loops_df["self_loops"]      >= thr_self
loops_df["flag_ret"]   = loops_df["returns_nonadj"]  >= thr_ret
loops_df["flag_pp"]    = loops_df["ping_pong"]       >= thr_pp
loops_df["flag_start"] = loops_df["back_to_start"]   >= thr_start
loops_df["flag_jump"]  = loops_df["jump_to_prev_any"]>= thr_jump
loops_df["flag_total"] = loops_df["loop_score"]      >= thr_sum

max_show = st.slider("Сколько примеров-«доказательств» показывать", 1, 20, 5)

def show_section(title, mask, sort_col, fmt_row, csv_name):
    st.subheader(title)
    bad = loops_df.loc[mask].copy()
    st.write(f"Наличие неэффективности: **{'Да' if len(bad)>0 else 'Нет'}** • кейсов: **{len(bad)}**")
    if not bad.empty:
        top = bad.sort_values(sort_col, ascending=False).head(max_show)
        st.markdown("**Показательные кейсы:**")
        for _, r in top.iterrows():
            st.markdown(fmt_row(r))
        with st.expander("⬇️ Выгрузка всего списка"):
            download_df(bad.sort_values(sort_col, ascending=False), csv_name, "Скачать CSV")

# суммарно
show_section(
    "Суммарный loop score",
    loops_df["flag_total"],
    "loop_score",
    lambda r: f"- **{r['case_id']}** — score={int(r['loop_score'])} "
              f"(в себя={int(r['self_loops'])}, возврат={int(r['returns_nonadj'])}, "
              f"пинг-понг={int(r['ping_pong'])}, в начало={int(r['back_to_start'])}, "
              f"в произвольный этап={int(r['jump_to_prev_any'])})",
    "loops_total.csv"
)
# подпункты
show_section(
    "В себя (A→A)",
    loops_df["flag_self"],
    "self_loops",
    lambda r: f"- **{r['case_id']}** — A→A: {int(r['self_loops'])} (примеры: {r['self_examples'][:3]})",
    "loops_self.csv"
)
show_section(
    "Возврат к пройденному шагу (A…A, не соседние)",
    loops_df["flag_ret"],
    "returns_nonadj",
    lambda r: f"- **{r['case_id']}** — возвратов: {int(r['returns_nonadj'])} (примеры: {r['return_examples'][:3]})",
    "loops_returns.csv"
)
show_section(
    "Пинг-понг (ABAB)",
    loops_df["flag_pp"],
    "ping_pong",
    lambda r: f"- **{r['case_id']}** — ABAB: {int(r['ping_pong'])} (примеры: {r['pingpong_examples'][:3]})",
    "loops_pingpong.csv"
)
show_section(
    "В начало (повтор стартовой активности)",
    loops_df["flag_start"],
    "back_to_start",
    lambda r: f"- **{r['case_id']}** — возвратов в начало: {int(r['back_to_start'])} (примеры: {r['start_examples'][:3]})",
    "loops_back_to_start.csv"
)
show_section(
    "В произвольный ранний этап (прыжки к ранним шагам)",
    loops_df["flag_jump"],
    "jump_to_prev_any",
    lambda r: f"- **{r['case_id']}** — «откатов»: {int(r['jump_to_prev_any'])} (примеры: {r['backjump_examples'][:3]})",
    "loops_backjump.csv"
)

any_loops = int(loops_df[["flag_self","flag_ret","flag_pp","flag_start","flag_jump","flag_total"]].any(axis=1).sum())
st.success(f"ИТОГО: кейсов с какой-либо зацикленностью — **{any_loops}** из {loops_df.shape[0]}.")

with st.expander("ℹ️ Определения подпунктов"):
    st.markdown(
        "- **В себя (A→A)** — два одинаковых шага подряд.\n"
        "- **Возврат (A…A)** — повтор ранее пройденного шага (не соседний).\n"
        "- **Пинг-понг (ABAB)** — чередование пары шагов A→B→A→B.\n"
        "- **В начало** — повтор стартовой активности позже в кейсе.\n"
        "- **В произвольный ранний этап** — переход к шагу, который уже встречался ранее (не соседний)."
    )

# =========================================================
# 5) DFG — вертикальная схема + экспорт PNG
# =========================================================
st.header("📌 Карта процесса (DFG) + экспорт")

if not HAS_PM4PY:
    st.info("PM4Py не установлен — секция DFG скрыта. Установи `pm4py` и (опц.) системный `graphviz` для PNG.")
else:
    # Формируем event log для pm4py
    fmt_kwargs = dict(case_id="case_id", activity_key="activity", timestamp_key="timestamp")
    evlog = pm4py.format_dataframe(work[["case_id","activity","timestamp"]], **fmt_kwargs)
    evlog = pm_df_utils.convert_timestamp_columns_in_df(evlog)
    if "time:timestamp" in evlog.columns:
        evlog = evlog.sort_values(["case:concept:name", "time:timestamp"])

    # Выбор метрики
    dfg_mode = st.radio("Метрика карты", ["Frequency", "Performance"], horizontal=True)
    if dfg_mode == "Frequency":
        dfg, sa, ea = pm4py.discover_dfg(evlog); variant = dfg_visualizer.Variants.FREQUENCY
    else:
        dfg, sa, ea = pm4py.discover_performance_dfg(evlog); variant = dfg_visualizer.Variants.PERFORMANCE

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
    gviz = dfg_visualizer.apply(dfg, log=evlog, variant=variant, parameters=params)
    gviz.graph_attr.update(rankdir=("TB" if rankdir.startswith("TB") else "LR"),
                           ranksep=str(ranksep), nodesep=str(nodesep), ratio=ratio)

    st.graphviz_chart(gviz.source, use_container_width=True)

    # DOT
    st.download_button("⬇️ DOT", gviz.source.encode("utf-8"),
                       file_name="process_dfg.dot", mime="text/plain")

    # PNG через graphviz (если есть)
    has_dot = shutil.which("dot") is not None
    if has_dot:
        try:
            with tempfile.TemporaryDirectory() as tmpd:
                outpath = os.path.join(tmpd, "process_dfg")
                gviz.render(filename=outpath, format="png", cleanup=True)
                with open(outpath + ".png", "rb") as f:
                    st.download_button("⬇️ PNG (graphviz)", f.read(),
                                       file_name="process_dfg.png", mime="image/png")
        except Exception as e:
            st.warning(f"PNG через graphviz не удалось: {e}")

    # Fallback PNG — вертикальный
    if not has_dot:
        st.info("Graphviz ('dot') не найден — рисую вертикальный PNG (fallback).")
        try:
            G = nx.DiGraph()
            for (u, v), w in dfg.items():
                G.add_edge(str(u), str(v), weight=w)
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
            fig, ax = plt.subplots(figsize=(8, 14))
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
            st.download_button("⬇️ PNG (fallback, вертикальный)", buf.getvalue(),
                               file_name="process_dfg_fallback_vertical.png", mime="image/png")
            st.pyplot(fig, use_container_width=True)
        except Exception as e:
            st.warning(f"Фоллбэк PNG не удался: {e}")
