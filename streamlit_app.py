# streamlit_app.py
# Process Mining — продвинутые неэффективности с подтипами (покейсно), PNG с фоллбэком

import os
import io
import tempfile
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# --- мягкая проверка pm4py, чтобы апп не падал на Cloud ---
try:
    import pm4py
    from pm4py.objects.log.util import dataframe_utils
    from pm4py.visualization.dfg import visualizer as dfg_visualizer
except ModuleNotFoundError:
    st.set_page_config(page_title="Process Mining — зависимости")
    st.error(
        "Модуль **pm4py** не установлен. Для Streamlit Cloud добавь в `requirements.txt` "
        "`pm4py>=2.7.11` (под Python 3.13) и опционально `graphviz` + `packages.txt: graphviz`."
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
# Маппинг + время (+ ресурс опционально для ping-pong по исполнителям)
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
# PM4Py event log
event_log = pm4py.format_dataframe(
    df,
    case_id="case_id",
    activity_key="activity",
    timestamp_key="timestamp",
    resource_key=("resource" if use_resource else None)
)
from pm4py.objects.log.util import dataframe_utils
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
# ИНДИКАТОРЫ И ПОДТИПЫ
# =========================
# ---------- 1) ЦИКЛЫ / ЗАЦИКЛЕННОСТЬ ----------
def loop_subtypes_for_case(sub: pd.DataFrame, use_res: bool) -> Dict[str, int]:
    """
    Подтипы:
      - self_loops: A→A (в себя)
      - ping_pong: ABAB (или BABA). Если есть ресурс — также «ресурсный пинг-понг» (A@R1 → A@R2 → A@R1 → A@R2)
      - returns_nonadj: возвраты к пройденным шагам (A … A, не соседние)
      - jump_to_prev_any: переход в любой ранее встречавшийся шаг (edge-«откат»)
      - back_to_start: возвраты к стартовой активности (… → A0)
    """
    acts = sub["activity"].tolist()
    res  = sub["resource"].tolist() if use_res else None
    if not acts:
        return {k: 0 for k in
                ["self_loops", "ping_pong", "ping_pong_res", "returns_nonadj", "jump_to_prev_any", "back_to_start", "loop_score_advanced"]}

    # 1) self loops
    self_loops = sum(1 for i in range(len(acts)-1) if acts[i] == acts[i+1])

    # 2) ping-pong по активностям (ABAB, длина окна >=4)
    ping_pong = 0
    i = 0
    while i+3 < len(acts):
        a, b, c, d = acts[i:i+4]
        if a != b and a == c and b == d:
            ping_pong += 1
            i += 2  # сдвиг на пол-шага, чтобы считать перекрытия умеренно
        else:
            i += 1

    # 2b) ресурсный пинг-понг (одна и та же активность, но R1↔R2↔R1↔R2)
    ping_pong_res = 0
    if use_res:
        i = 0
        while i+3 < len(acts):
            a1, r1 = acts[i],   res[i]
            a2, r2 = acts[i+1], res[i+1]
            a3, r3 = acts[i+2], res[i+2]
            a4, r4 = acts[i+3], res[i+3]
            # одна активность, но ресурсы чередуются
            if a1 == a2 == a3 == a4 and len({r1, r2}) == 2 and r1 == r3 and r2 == r4:
                ping_pong_res += 1
                i += 2
            else:
                i += 1

    # 3) возвраты к пройденным шагам (не соседние A…A)
    returns_nonadj = 0
    last_pos = {}
    for i, a in enumerate(acts):
        if a in last_pos and i - last_pos[a] > 1:
            returns_nonadj += 1
        last_pos[a] = i

    # 4) «прыжок в произвольный ранний этап»: переход, где next_activity уже встречалась ранее в трассе
    jump_to_prev_any = 0
    seen = set()
    for i in range(len(acts)-1):
        seen.add(acts[i])
        if acts[i+1] in seen and acts[i+1] != acts[i]:
            jump_to_prev_any += 1

    # 5) возврат в начало (повтор стартовой активности)
    start_act = acts[0]
    back_to_start = sum(1 for i in range(1, len(acts)) if acts[i] == start_act)

    loop_score_adv = self_loops + ping_pong + ping_pong_res + returns_nonadj + jump_to_prev_any + back_to_start
    return {
        "self_loops": self_loops,
        "ping_pong": ping_pong,
        "ping_pong_res": ping_pong_res,
        "returns_nonadj": returns_nonadj,
        "jump_to_prev_any": jump_to_prev_any,
        "back_to_start": back_to_start,
        "loop_score_advanced": loop_score_adv,
    }

loop_rows = []
for cid, g in df.groupby("case_id"):
    sc = loop_subtypes_for_case(g, use_resource)
    sc["case_id"] = cid
    loop_rows.append(sc)
loops_df = pd.DataFrame(loop_rows)

# авто-пороги по подтипам (q75), общий — по сумме (q75)
def q75(s): return int(np.ceil(safe_pct(s, 75, default=1)))
thr_self   = q75(loops_df["self_loops"])
thr_pp     = q75(loops_df["ping_pong"])
thr_ppr    = q75(loops_df["ping_pong_res"]) if use_resource else 1
thr_ret    = q75(loops_df["returns_nonadj"])
thr_jump   = q75(loops_df["jump_to_prev_any"])
thr_start  = q75(loops_df["back_to_start"])
thr_loop_total = q75(loops_df["loop_score_advanced"])

# ---------- 2) ДЛИТЕЛЬНОСТЬ ОПЕРАЦИЙ ----------
# Подтипы:
#   - single_spike: есть Δ > p99(edge) (хотя бы одно экстремально долгое ожидание)
#   - many_moderate: суммарный overrun относительно median(edge) велик (q90)
#   - queue_before_activity: систематический «хвост» перед конкретной активностью (много больших Δ на вход в один и тот же шаг)
edge_median_map = {e: s["median"] for e, s in edge_stats_dict.items()}

def per_case_overruns(sub_edges: pd.DataFrame):
    single_spike = 0
    overrun_sum  = 0.0
    entry_waits  = {}  # вход в активность: … -> X
    for _, row in sub_edges.iterrows():
        e = (row["activity"], row["next_activity"])
        d = float(row["delta_sec"])
        med = edge_median_map.get(e, np.nan)
        p99 = edge_p99_map.get(e, np.nan)
        if not np.isnan(med):
            overrun_sum += max(0.0, d - med)
        if not np.isnan(p99) and d > p99:
            single_spike += 1
        # копим ожидания по входу в next_activity
        entry_waits[row["next_activity"]] = entry_waits.get(row["next_activity"], 0.0) + d
    # «очередь» — если на 1 активность приходится ≥ 40% суммарного ожидания кейса
    total_wait = sum(entry_waits.values())
    queue_flag = False
    queue_target = None
    if total_wait > 0:
        best_act = max(entry_waits, key=entry_waits.get)
        if entry_waits[best_act] / total_wait >= 0.40:
            queue_flag = True
            queue_target = best_act
    return single_spike, overrun_sum, queue_flag, queue_target

over_rows = []
for cid, g in df_sorted.groupby("case_id"):
    sub_edges = g.dropna(subset=["next_activity", "delta_sec"])
    ss, osum, qf, qtarget = per_case_overruns(sub_edges)
    over_rows.append({"case_id": cid, "single_spike_cnt": ss, "overrun_sum_sec": osum,
                      "queue_before_flag": qf, "queue_target": qtarget})
over_df = pd.DataFrame(over_rows)

thr_over_sum   = float(np.ceil(safe_pct(over_df["overrun_sum_sec"], 90, default=0.0)))
thr_over_spike = max(1, int(np.ceil(safe_pct(over_df["single_spike_cnt"], 75, default=1))))

# ---------- 3) ВЛИЯНИЕ НА ПРОЦЕСС ----------
# Подтипы:
#   - impact_sum: Σ(max(0, Δ - p95(edge))) — вклад кейса сверх типовых ожиданий
#   - bottleneck_share: доля переходов по глобальным «узким» рёбрам (top-k по median)
#   - p95_exceed_count: сколько раз кейс превышал p95 своих рёбер
k_bottlenecks = st.sidebar.number_input("Top-k узких рёбер (для метрики влияния)", min_value=1, value=10, step=1)
top_edges = set(edge_stats.sort_values("median", ascending=False).head(k_bottlenecks)["edge"].tolist())

def impact_for_case(sub_edges: pd.DataFrame):
    if sub_edges.empty:
        return 0.0, 0.0, 0
    imp_sum = 0.0
    in_bneck = 0
    exceed_cnt = 0
    n_edges = 0
    for _, row in sub_edges.iterrows():
        e = (row["activity"], row["next_activity"])
        d = float(row["delta_sec"])
        p95 = edge_p95_map.get(e, np.nan)
        n_edges += 1
        if e in top_edges:
            in_bneck += 1
        if not np.isnan(p95):
            exc = d - p95
            if exc > 0:
                imp_sum += exc
                exceed_cnt += 1
    share = in_bneck / n_edges if n_edges else 0.0
    return imp_sum, share, exceed_cnt

impact_rows = []
for cid, g in df_sorted.groupby("case_id"):
    sub_edges = g.dropna(subset=["next_activity", "delta_sec"])
    isum, share, excnt = impact_for_case(sub_edges)
    impact_rows.append({"case_id": cid, "impact_sum_sec": isum, "bneck_share": share, "p95_exceed_cnt": excnt})
impact_df = pd.DataFrame(impact_rows)

thr_impact_sum   = float(np.ceil(safe_pct(impact_df["impact_sum_sec"], 90, default=0.0)))
thr_bneck_share  = float(np.round(safe_pct(impact_df["bneck_share"], 90, default=0.5), 2))
thr_exceed_cnt   = int(np.ceil(safe_pct(impact_df["p95_exceed_cnt"], 75, default=1)))

# =========================
# Режим: Авто / Ручной
# =========================
st.subheader("🧠 Пороговые значения")
mode = st.radio("Режим порогов", ["Авто", "Ручной"], horizontal=True, index=0)
if mode == "Ручной":
    c1, c2, c3 = st.columns(3)
    with c1:
        thr_self  = st.number_input("Циклы: в себя (≥)", 0, value=int(thr_self))
        thr_pp    = st.number_input("Циклы: пинг-понг (≥)", 0, value=int(thr_pp))
        if use_resource:
            thr_ppr = st.number_input("Циклы: ресурсный пинг-понг (≥)", 0, value=int(thr_ppr))
    with c2:
        thr_ret   = st.number_input("Циклы: возвраты A…A (≥)", 0, value=int(thr_ret))
        thr_jump  = st.number_input("Циклы: прыжки в ранние этапы (≥)", 0, value=int(thr_jump))
        thr_start = st.number_input("Циклы: возвраты в начало (≥)", 0, value=int(thr_start))
    with c3:
        thr_loop_total = st.number_input("Циклы: суммарный score (≥)", 0, value=int(thr_loop_total))
        thr_over_sum   = st.number_input("Длительность: Σ overrun, сек (≥)", 0, value=int(thr_over_sum), step=10)
        thr_over_spike = st.number_input("Длительность: экстремальные ожидания (шт ≥)", 0, value=int(thr_over_spike))
        thr_impact_sum = st.number_input("Влияние: impact_sum (сек ≥)", 0, value=int(thr_impact_sum), step=10)
        thr_bneck_share = st.number_input("Влияние: доля узких рёбер (≥)", 0.0, 1.0, value=float(thr_bneck_share), step=0.05)
        thr_exceed_cnt  = st.number_input("Влияние: превышений p95 (шт ≥)", 0, value=int(thr_exceed_cnt))

max_show = st.slider("Сколько примеров-«доказательств» показывать", 1, 20, 5)

# =========================
# ФЛАГИ (все кейсы считаются на бэке)
# =========================
loops_merge = loops_df.copy()
loops_merge["flag_self"]   = loops_merge["self_loops"]         >= thr_self
loops_merge["flag_pp"]     = loops_merge["ping_pong"]           >= thr_pp
loops_merge["flag_ppr"]    = (loops_merge["ping_pong_res"]     >= thr_ppr) if use_resource else False
loops_merge["flag_ret"]    = loops_merge["returns_nonadj"]      >= thr_ret
loops_merge["flag_jump"]   = loops_merge["jump_to_prev_any"]    >= thr_jump
loops_merge["flag_start"]  = loops_merge["back_to_start"]       >= thr_start
loops_merge["flag_loop"]   = loops_merge["loop_score_advanced"] >= thr_loop_total

over_merge = over_df.copy()
over_merge["flag_over_sum"]   = over_merge["overrun_sum_sec"]  >= thr_over_sum
over_merge["flag_over_spike"] = over_merge["single_spike_cnt"] >= thr_over_spike
over_merge["flag_queue"]      = over_merge["queue_before_flag"].fillna(False)

impact_merge = impact_df.copy()
impact_merge["flag_imp_sum"]  = impact_merge["impact_sum_sec"] >= thr_impact_sum
impact_merge["flag_imp_share"] = impact_merge["bneck_share"]   >= thr_bneck_share
impact_merge["flag_imp_exceed"] = impact_merge["p95_exceed_cnt"] >= thr_exceed_cnt

summary = (loops_merge[["case_id", "flag_self","flag_pp","flag_ppr","flag_ret","flag_jump","flag_start","flag_loop"]]
           .merge(over_merge[["case_id","flag_over_sum","flag_over_spike","flag_queue"]], on="case_id", how="outer")
           .merge(impact_merge[["case_id","flag_imp_sum","flag_imp_share","flag_imp_exceed"]], on="case_id", how="outer"))
summary = summary.fillna(False)
summary["any_flag"] = summary.drop(columns=["case_id"]).any(axis=1)

def download_df_button(df_to_dl: pd.DataFrame, filename: str, label: str):
    csv = df_to_dl.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv, file_name=filename, mime="text/csv")

# =========================
# ВЫВОД (только факты + короткие примеры)
# =========================
st.header("🧪 Результаты по индикаторам и подтипам")

def show_section(title: str, df_bad: pd.DataFrame, sort_cols, fmt_row, csv_name: str):
    st.subheader(title)
    n_bad = int(df_bad.shape[0])
    st.write(f"Наличие неэффективности: **{'Да' if n_bad>0 else 'Нет'}**  • Затронуто кейсов: **{n_bad}** из {n_cases}")
    if n_bad > 0:
        show = df_bad.sort_values(sort_cols, ascending=False).head(max_show)
        st.markdown("**Показательные кейсы (доказательства):**")
        for _, r in show.iterrows():
            st.markdown(fmt_row(r))
        with st.expander("⬇️ Выгрузка полного списка"):
            download_df_button(df_bad.sort_values(sort_cols, ascending=False), csv_name, "Скачать CSV")

# --- Циклы: общий флаг ---
show_section(
    "1) Циклы — суммарный score",
    loops_merge[loops_merge["flag_loop"]],
    ["loop_score_advanced"],
    lambda r: f"- **{r['case_id']}** — score={int(r['loop_score_advanced'])} (в себя={int(r['self_loops'])}, "
              f"пинг-понг={int(r['ping_pong'])}{', ресурсный='+str(int(r['ping_pong_res'])) if use_resource else ''}, "
              f"возвраты={int(r['returns_nonadj'])}, прыжки={int(r['jump_to_prev_any'])}, в начало={int(r['back_to_start'])})",
    "ineff_cycles_total.csv"
)

# --- Циклы: подтипы отдельно ---
show_section(
    "1.1) Циклы — в себя (A→A)",
    loops_merge[loops_merge["flag_self"]][["case_id","self_loops"]],
    ["self_loops"],
    lambda r: f"- **{r['case_id']}** — A→A: {int(r['self_loops'])}",
    "ineff_cycle_self.csv"
)
show_section(
    "1.2) Циклы — пинг-понг (ABAB)",
    loops_merge[loops_merge["flag_pp"]][["case_id","ping_pong"]],
    ["ping_pong"],
    lambda r: f"- **{r['case_id']}** — ABAB: {int(r['ping_pong'])}",
    "ineff_cycle_pingpong.csv"
)
if use_resource:
    show_section(
        "1.3) Циклы — ресурсный пинг-понг (A@R1↔A@R2)",
        loops_merge[loops_merge["flag_ppr"]][["case_id","ping_pong_res"]],
        ["ping_pong_res"],
        lambda r: f"- **{r['case_id']}** — ресурсный ABAB: {int(r['ping_pong_res'])}",
        "ineff_cycle_pingpong_resource.csv"
    )
show_section(
    "1.4) Циклы — возврат к пройденному шагу (A…A, не соседние)",
    loops_merge[loops_merge["flag_ret"]][["case_id","returns_nonadj"]],
    ["returns_nonadj"],
    lambda r: f"- **{r['case_id']}** — возвратов: {int(r['returns_nonadj'])}",
    "ineff_cycle_returns.csv"
)
show_section(
    "1.5) Циклы — прыжки в произвольный ранний этап",
    loops_merge[loops_merge["flag_jump"]][["case_id","jump_to_prev_any"]],
    ["jump_to_prev_any"],
    lambda r: f"- **{r['case_id']}** — «откатов» переходом: {int(r['jump_to_prev_any'])}",
    "ineff_cycle_backjumps.csv"
)
show_section(
    "1.6) Циклы — возврат в начало",
    loops_merge[loops_merge["flag_start"]][["case_id","back_to_start"]],
    ["back_to_start"],
    lambda r: f"- **{r['case_id']}** — возвратов к стартовой активности: {int(r['back_to_start'])}",
    "ineff_cycle_back_to_start.csv"
)

# --- Длительность операций: подтипы ---
show_section(
    "2) Длительность — суммарный перерасход",
    over_merge[over_merge["flag_over_sum"]][["case_id","overrun_sum_sec"]],
    ["overrun_sum_sec"],
    lambda r: f"- **{r['case_id']}** — Σ overrun={int(r['overrun_sum_sec'])} сек",
    "ineff_duration_overrun.csv"
)
show_section(
    "2.1) Длительность — экстремальные ожидания (Δ > p99(edge))",
    over_merge[over_merge["flag_over_spike"]][["case_id","single_spike_cnt"]],
    ["single_spike_cnt"],
    lambda r: f"- **{r['case_id']}** — экстремумов: {int(r['single_spike_cnt'])}",
    "ineff_duration_spikes.csv"
)
show_section(
    "2.2) Длительность — «очередь» перед активностью",
    over_merge[over_merge["flag_queue"]][["case_id","queue_target"]],
    ["queue_target"],
    lambda r: f"- **{r['case_id']}** — концентрация ожидания перед: **{r['queue_target']}**",
    "ineff_duration_queue.csv"
)

# --- Влияние на процесс: подтипы ---
show_section(
    "3) Влияние — impact_sum (сверх p95)",
    impact_merge[impact_merge["flag_imp_sum"]][["case_id","impact_sum_sec"]],
    ["impact_sum_sec"],
    lambda r: f"- **{r['case_id']}** — impact_sum={int(r['impact_sum_sec'])} сек",
    "ineff_impact_sum.csv"
)
show_section(
    "3.1) Влияние — доля рёбер из глобальных «узких»",
    impact_merge[impact_merge["flag_imp_share"]][["case_id","bneck_share"]],
    ["bneck_share"],
    lambda r: f"- **{r['case_id']}** — доля узких рёбер={r['bneck_share']:.2f}",
    "ineff_impact_share.csv"
)
show_section(
    "3.2) Влияние — количество превышений p95",
    impact_merge[impact_merge["flag_imp_exceed"]][["case_id","p95_exceed_cnt"]],
    ["p95_exceed_cnt"],
    lambda r: f"- **{r['case_id']}** — превышений p95: {int(r['p95_exceed_cnt'])}",
    "ineff_impact_exceed.csv"
)

# =========================
# Сводка фактов
# =========================
st.header("📋 Сводка фактов")
total_any = int(summary["any_flag"].sum())
st.write(
    f"- Циклы (любой подтип или суммарный score): **{int(loops_merge[['flag_self','flag_pp','flag_ppr','flag_ret','flag_jump','flag_start','flag_loop']].any(axis=1).sum())}** кейсов.\n"
    f"- Длительность (какой-либо подтип): **{int(over_merge[['flag_over_sum','flag_over_spike','flag_queue']].any(axis=1).sum())}** кейсов.\n"
    f"- Влияние (какой-либо подтип): **{int(impact_merge[['flag_imp_sum','flag_imp_share','flag_imp_exceed']].any(axis=1).sum())}** кейсов.\n"
    f"- Любая неэффективность: **{total_any}** из {n_cases}."
)
with st.expander("⬇️ Экспорт полной сводки по флагам"):
    download_df_button(summary, "ineff_summary_all_flags.csv", "Скачать CSV")

# =========================
# DFG + PNG экспорт (graphviz / fallback)
# =========================
with st.expander("📌 Карта процесса (DFG) и экспорт PNG", expanded=False):
    dfg_mode = st.radio("Метрика карты", ["Frequency", "Performance"], horizontal=True, key="dfg_mode")
    if dfg_mode == "Frequency":
        dfg, sa, ea = pm4py.discover_dfg(event_log)
        variant = dfg_visualizer.Variants.FREQUENCY
    else:
        dfg, sa, ea = pm4py.discover_performance_dfg(event_log)
        variant = dfg_visualizer.Variants.PERFORMANCE
    gviz = dfg_visualizer.apply(dfg, log=event_log, variant=variant, parameters={"start_activities": sa, "end_activities": ea})
    st.graphviz_chart(gviz.source, use_container_width=True)

    st.download_button("⬇️ DOT", gviz.source.encode("utf-8"), file_name="process_dfg.dot", mime="text/plain")

    has_dot = shutil.which("dot") is not None
    if has_dot:
        try:
            with tempfile.TemporaryDirectory() as tmpd:
                outpath = os.path.join(tmpd, "process_dfg")
                gviz.render(filename=outpath, format="png", cleanup=True)
                with open(outpath + ".png", "rb") as f:
                    st.download_button("⬇️ PNG (graphviz)", f, file_name="process_dfg.png", mime="image/png")
        except Exception as e:
            st.warning(f"PNG через graphviz не удалось: {e}")

    if not has_dot:
        st.info("Graphviz ('dot') не найден — рендерю упрощённый PNG.")
        try:
            # Готовим граф для фоллбэка
            G = nx.DiGraph()
            for (u, v), w in dfg.items():
                G.add_edge(str(u), str(v), weight=w)
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
            st.warning(f"Фоллбэк PNG не удался: {e}. Поставь `graphviz`.")
