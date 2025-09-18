# streamlit_app.py
# Bottle neck (по слайду) + ОТДЕЛЬНЫЙ БЛОК «Зацикленность» с подпунктами:
# - в себя (A→A)
# - возврат (A…A, не соседние)
# - пинг-понг (ABAB)
# - в начало (повтор стартовой активности)
# - в произвольный ранний этап (прыжки к ранее встречавшимся шагам)
#
# Всё считается для КАЖДОГО кейса на бэке. На экран — факт наличия и короткие примеры; полные списки — в CSV.

import os
import io
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

st.set_page_config(page_title="Process Mining — Bottlenecks & Loops", layout="wide")
st.title("🚦 Bottle neck + 🔄 Зацикленность (по кейсам)")

# -----------------------------
# 1) Загрузка данных
# -----------------------------
with st.expander("⚙️ Загрузка", expanded=False):
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

st.subheader("📊 Предпросмотр данных")
st.dataframe(df.head(), use_container_width=True)

# -----------------------------
# 2) Маппинг
# -----------------------------
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
# 3) ПРОКСИ-длительность операции (для bottleneck по слайду)
# =========================================================
work["next_ts"] = work.groupby("case_id")["timestamp"].shift(-1)
work["delta_sec"] = (work["next_ts"] - work["timestamp"]).dt.total_seconds()
ev = work.dropna(subset=["delta_sec"]).copy()

ops = (
    ev.groupby("activity")["delta_sec"]
      .agg(count="count", mean_dur="mean", median_dur="median", std_dur="std")
      .reset_index()
)

# =========================================================
# 4) Идентификация Bottle neck (по презентации)
# =========================================================
st.header("🔎 Bottle neck — идентификация (по слайду)")

pctl = st.slider("Порог «долгих» операций (перцентиль по mean_dur)", 50, 99, 90)
p_long = np.percentile(ops["mean_dur"].dropna(), pctl)
ops["is_long"] = ops["mean_dur"] >= p_long

ops["mm_ratio"] = ops["mean_dur"] / ops["median_dur"]
ops["cond_mm"] = (ops["mm_ratio"] > 0.9) & (ops["mm_ratio"] < 1.1)

mean_std_all = ops["std_dur"].replace([np.inf, -np.inf], np.nan).dropna().mean()
ops["norm_std"] = ops["std_dur"] / (mean_std_all if mean_std_all and not np.isnan(mean_std_all) else 1.0)
ops["cond_std"] = ops["norm_std"] < 0.5

ops["is_bottleneck"] = ops["is_long"] & ops["cond_mm"] & ops["cond_std"]

def fmt_time(sec):
    if pd.isna(sec): return ""
    sec = float(sec)
    if sec < 120: return f"{int(round(sec))} с"
    m = sec / 60
    if m < 120: return f"{m:.1f} мин"
    h = m / 60
    if h < 48: return f"{h:.1f} ч"
    d = h / 24
    return f"{d:.1f} д"

ops_show = ops.copy()
for col in ["mean_dur", "median_dur", "std_dur"]:
    ops_show[col] = ops_show[col].apply(fmt_time)

st.dataframe(
    ops_show[["activity", "count", "mean_dur", "median_dur", "std_dur", "mm_ratio", "norm_std", "is_bottleneck"]],
    use_container_width=True
)

bn_ops = ops[ops["is_bottleneck"]].sort_values("mean_dur", ascending=False)
st.success(f"Найдено узких операций: **{bn_ops.shape[0]}** из {ops.shape[0]}")

st.download_button(
    "⬇️ CSV — метрики и флаги по операциям",
    ops.to_csv(index=False).encode("utf-8"),
    file_name="bottleneck_operations.csv",
    mime="text/csv"
)

# =========================================================
# 5) Визуализация распределения по выбранной операции
# =========================================================
st.subheader("📈 Распределение длительностей по операции")
act_for_plot = st.selectbox(
    "Операция для графика",
    options=ops.sort_values("mean_dur", ascending=False)["activity"].tolist()
)
sub = ev[ev["activity"] == act_for_plot]["delta_sec"].dropna()
fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(sub / 60.0, bins=50)
ax.set_title(f"{act_for_plot} — распределение (мин)")
ax.set_xlabel("минуты"); ax.set_ylabel("кол-во")
st.pyplot(fig, use_container_width=True)

# =========================================================
# 6) Оценка потенциального эффекта (по слайду)
# =========================================================
st.header("💰 Оценка потенциального эффекта")
st.caption("Сумма превышений над средним по операции, умноженная на стоимость минуты S.")
S = st.number_input("Стоимость минуты процесса S", min_value=0.0, value=10.0, step=1.0)
currency = st.text_input("Валюта/единица", value="₽")

mean_map = ops.set_index("activity")["mean_dur"].to_dict()
ev["over_mean_sec"] = ev.apply(
    lambda r: max(0.0, float(r["delta_sec"]) - float(mean_map.get(r["activity"], np.nan)))
    if pd.notna(mean_map.get(r["activity"], np.nan)) else np.nan, axis=1)
ev_eff = ev.dropna(subset=["over_mean_sec"]).copy()
ev_eff["over_mean_min"] = ev_eff["over_mean_sec"] / 60.0
ev_eff["cost"] = ev_eff["over_mean_min"] * S

eff_by_act = ev_eff.groupby("activity").agg(
    appearances=("activity", "count"),
    over_minutes=("over_mean_min", "sum"),
    cost=("cost", "sum")
).reset_index().sort_values("cost", ascending=False)
total_cost = eff_by_act["cost"].sum()
st.success(f"Итоговый потенциальный эффект: **{total_cost:,.0f} {currency}**")
st.dataframe(eff_by_act.assign(over_minutes=lambda d: d["over_minutes"].round(1),
                               cost=lambda d: d["cost"].round(0)),
             use_container_width=True)
st.download_button(
    "⬇️ CSV — потенциальный эффект по операциям",
    eff_by_act.to_csv(index=False).encode("utf-8"),
    file_name="bottleneck_effect_by_operation.csv",
    mime="text/csv"
)

# =========================================================
# 7) 🔄 ЗАЦИКЛЕННОСТЬ — пункт и подпункты (по КАЖДОМУ кейсу)
# =========================================================
st.header("🔄 Зацикленность — покейсная диагностика")

# Подсчёты по каждому кейсу
def loop_subtypes_for_case(group: pd.DataFrame) -> dict:
    acts = group["activity"].tolist()
    if not acts:
        return dict(self_loops=0, returns_nonadj=0, ping_pong=0, back_to_start=0, jump_to_prev_any=0, loop_score=0)

    # 1) в себя (A→A)
    self_loops = sum(1 for i in range(len(acts)-1) if acts[i] == acts[i+1])

    # 2) возврат (A…A, не соседние)
    returns_nonadj, last = 0, {}
    for i, a in enumerate(acts):
        if a in last and i - last[a] > 1:
            returns_nonadj += 1
        last[a] = i

    # 3) пинг-понг (ABAB)
    ping_pong, i = 0, 0
    while i+3 < len(acts):
        a,b,c,d = acts[i:i+4]
        if a != b and a == c and b == d:
            ping_pong += 1
            i += 2
        else:
            i += 1

    # 4) в начало (повтор стартовой активности)
    start_act = acts[0]
    back_to_start = sum(1 for j in range(1, len(acts)) if acts[j] == start_act)

    # 5) в произвольный ранний этап (прыжок к ранее встреченному шагу, не соседний)
    jump_to_prev_any, seen = 0, set()
    for j in range(len(acts)-1):
        seen.add(acts[j])
        if acts[j+1] in seen and acts[j+1] != acts[j]:
            jump_to_prev_any += 1

    score = self_loops + returns_nonadj + ping_pong + back_to_start + jump_to_prev_any
    return dict(self_loops=self_loops, returns_nonadj=returns_nonadj, ping_pong=ping_pong,
                back_to_start=back_to_start, jump_to_prev_any=jump_to_prev_any, loop_score=score)

loop_rows = []
for cid, g in work.groupby("case_id"):
    row = loop_subtypes_for_case(g)
    row["case_id"] = cid
    loop_rows.append(row)
loops_df = pd.DataFrame(loop_rows)

def q75(s): 
    s = pd.Series(s).fillna(0)
    return int(np.ceil(np.percentile(s, 75))) if len(s) else 1

# Автопороги по подпунктам
thr_self   = q75(loops_df["self_loops"])
thr_ret    = q75(loops_df["returns_nonadj"])
thr_pp     = q75(loops_df["ping_pong"])
thr_start  = q75(loops_df["back_to_start"])
thr_jump   = q75(loops_df["jump_to_prev_any"])
thr_total  = q75(loops_df["loop_score"])

with st.expander("🧠 Пороговые значения (можно подкрутить)", expanded=False):
    c1, c2, c3 = st.columns(3)
    with c1:
        thr_self  = st.number_input("В себя (A→A) ≥", 0, value=int(thr_self))
        thr_ret   = st.number_input("Возврат (A…A) ≥", 0, value=int(thr_ret))
    with c2:
        thr_pp    = st.number_input("Пинг-понг (ABAB) ≥", 0, value=int(thr_pp))
        thr_start = st.number_input("В начало ≥", 0, value=int(thr_start))
    with c3:
        thr_jump  = st.number_input("В произвольный ранний этап ≥", 0, value=int(thr_jump))
        thr_total = st.number_input("Суммарный loop score ≥", 0, value=int(thr_total))

# Флаги по подпунктам
loops_df["flag_self"]  = loops_df["self_loops"]        >= thr_self
loops_df["flag_ret"]   = loops_df["returns_nonadj"]     >= thr_ret
loops_df["flag_pp"]    = loops_df["ping_pong"]          >= thr_pp
loops_df["flag_start"] = loops_df["back_to_start"]      >= thr_start
loops_df["flag_jump"]  = loops_df["jump_to_prev_any"]   >= thr_jump
loops_df["flag_total"] = loops_df["loop_score"]         >= thr_total

def download(df_to_dl, fname, label):
    st.download_button(label, df_to_dl.to_csv(index=False).encode("utf-8"),
                       file_name=fname, mime="text/csv")

max_show = st.slider("Сколько примеров-«доказательств» на подпункт", 1, 20, 5)

# Вспомогательная печать секции
def show_loop_subsection(title, flt_col, sort_col, fmt_fn, csv_name):
    st.subheader(title)
    bad = loops_df[loops_df[flt_col]].copy()
    n = int(bad.shape[0])
    st.write(f"Наличие неэффективности: **{'Да' if n>0 else 'Нет'}** • кейсов: **{n}**")
    if n > 0:
        show = bad.sort_values(sort_col, ascending=False).head(max_show)
        st.markdown("**Показательные кейсы:**")
        for _, r in show.iterrows():
            st.markdown(fmt_fn(r))
        with st.expander("⬇️ Выгрузка полного списка"):
            download(bad.sort_values(sort_col, ascending=False), csv_name, "Скачать CSV")

# Главный подпункт — суммарный
show_loop_subsection(
    "7.1 Суммарный loop score",
    "flag_total", "loop_score",
    lambda r: f"- **{r['case_id']}** — score={int(r['loop_score'])} "
              f"(в себя={int(r['self_loops'])}, возврат={int(r['returns_nonadj'])}, "
              f"пинг-понг={int(r['ping_pong'])}, в начало={int(r['back_to_start'])}, "
              f"в ранний этап={int(r['jump_to_prev_any'])})",
    "loops_total.csv"
)

# Подпункты по отдельности
show_loop_subsection(
    "7.2 В себя (A→A)",
    "flag_self", "self_loops",
    lambda r: f"- **{r['case_id']}** — A→A: {int(r['self_loops'])}",
    "loops_self.csv"
)
show_loop_subsection(
    "7.3 Возврат к пройденному шагу (A…A, не соседние)",
    "flag_ret", "returns_nonadj",
    lambda r: f"- **{r['case_id']}** — возвратов: {int(r['returns_nonadj'])}",
    "loops_returns.csv"
)
show_loop_subsection(
    "7.4 Пинг-понг (ABAB)",
    "flag_pp", "ping_pong",
    lambda r: f"- **{r['case_id']}** — ABAB: {int(r['ping_pong'])}",
    "loops_pingpong.csv"
)
show_loop_subsection(
    "7.5 В начало (повтор стартовой активности)",
    "flag_start", "back_to_start",
    lambda r: f"- **{r['case_id']}** — возвратов в начало: {int(r['back_to_start'])}",
    "loops_back_to_start.csv"
)
show_loop_subsection(
    "7.6 В произвольный ранний этап (прыжки к ранним шагам)",
    "flag_jump", "jump_to_prev_any",
    lambda r: f"- **{r['case_id']}** — «откатов» переходом: {int(r['jump_to_prev_any'])}",
    "loops_backjump.csv"
)

# Итог по зацикленности
any_loops = loops_df[["flag_self","flag_ret","flag_pp","flag_start","flag_jump","flag_total"]].any(axis=1).sum()
st.success(f"ИТОГО: кейсов с какой-либо зацикленностью — **{int(any_loops)}** из {loops_df.shape[0]}.")

# Маленькая памятка
with st.expander("ℹ️ Как мы определяем подпункты зацикленности?"):
    st.markdown(
        "- **В себя (A→A):** два одинаковых подряд шага.\n"
        "- **Возврат (A…A):** повтор ранее пройденного шага (не соседний).\n"
        "- **Пинг-понг (ABAB):** чередование пары шагов минимум по схеме A→B→A→B.\n"
        "- **В начало:** повтор стартовой активности позже в кейсе.\n"
        "- **В произвольный ранний этап:** переход в шаг, который уже встречался ранее (не соседний)."
    )

st.success("Готово. Все кейсы просчитаны на бэке; примеры и CSV — выше.")
